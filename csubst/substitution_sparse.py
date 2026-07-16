import numpy as np
import scipy.sparse as sp
from types import MappingProxyType

from csubst._extensions import load_optional_extension

substitution_sparse_cy = load_optional_extension('substitution_sparse_cy')


def _sum_result_dtype(dtype):
    return np.asarray(np.zeros(shape=(0,), dtype=dtype).sum()).dtype


def _normalize_sum_axes(axis, ndim):
    if axis is None:
        return None
    axes = axis if isinstance(axis, tuple) else (axis,)
    normalized = []
    for ax in axes:
        ax = int(ax)
        if ax < 0:
            ax += ndim
        if (ax < 0) or (ax >= ndim):
            raise ValueError('axis {} is out of bounds for sparse substitution tensor with {} dimensions'.format(ax, ndim))
        normalized.append(ax)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate value in 'axis'")
    return tuple(normalized)


class SparseSubstitutionTensor:
    """Immutable 5-D substitution tensor backed by one packed CSR matrix.

    The CSR rows are branches and its columns are ``event_id * num_site + site``,
    where ``event_id = (group * num_state_from + from_state) * num_state_to
    + to_state``.  Empty/invalid event types consume no payload.  Keeping all
    events in one CSR removes the per-event ``indptr`` arrays and Python/SciPy
    object overhead of the former block dictionary representation.

    ``blocks`` remains as a lazy compatibility view.  Performance-sensitive
    code should use ``matrix``, ``get_block()``, or ``project()`` instead.
    """

    def __init__(self, shape, dtype, blocks=None, matrix=None):
        if len(shape) != 5:
            raise ValueError('SparseSubstitutionTensor shape should have 5 dimensions.')
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        if (blocks is None) == (matrix is None):
            raise ValueError('Exactly one of blocks or matrix should be supplied.')
        if matrix is None:
            matrix = self._pack_blocks(blocks)
        expected_shape = (self.shape[0], self.num_event * self.num_site)
        if not sp.issparse(matrix):
            raise TypeError('Packed substitution tensor storage should be a SciPy sparse matrix.')
        if tuple(matrix.shape) != expected_shape:
            raise ValueError('Packed matrix has shape {}; expected {}.'.format(matrix.shape, expected_shape))
        needs_copy = (
            (not sp.isspmatrix_csr(matrix))
            or (matrix.dtype != self.dtype)
            or (not matrix.has_canonical_format)
            or (not matrix.has_sorted_indices)
            or ((matrix.nnz > 0) and np.any(matrix.data == 0))
        )
        if needs_copy:
            matrix = sp.csr_matrix(matrix, dtype=self.dtype, copy=True)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
        matrix.data.flags.writeable = False
        matrix.indices.flags.writeable = False
        matrix.indptr.flags.writeable = False
        self.matrix = matrix
        self._blocks_cache = None

    @property
    def num_branch(self):
        return self.shape[0]

    @property
    def num_site(self):
        return self.shape[1]

    @property
    def num_group(self):
        return self.shape[2]

    @property
    def num_state_from(self):
        return self.shape[3]

    @property
    def num_state_to(self):
        return self.shape[4]

    @property
    def num_event(self):
        return self.num_group * self.num_state_from * self.num_state_to

    def event_id(self, sg, a, d):
        sg, a, d = int(sg), int(a), int(d)
        if not (0 <= sg < self.num_group and 0 <= a < self.num_state_from and 0 <= d < self.num_state_to):
            raise IndexError('Substitution event ({},{},{}) is out of bounds for shape {}.'.format(sg, a, d, self.shape))
        return (sg * self.num_state_from + a) * self.num_state_to + d

    def decode_event_ids(self, event_ids):
        event_ids = np.asarray(event_ids, dtype=np.int64)
        sg, rem = np.divmod(event_ids, self.num_state_from * self.num_state_to)
        a, d = np.divmod(rem, self.num_state_to)
        return sg, a, d

    def _pack_blocks(self, blocks):
        rows = []
        cols = []
        vals = []
        for raw_key, raw_mat in blocks.items():
            if (not isinstance(raw_key, tuple)) or len(raw_key) != 3:
                raise ValueError('Sparse substitution block keys should be (group, from, to) tuples.')
            sg, a, d = (int(v) for v in raw_key)
            event_id = self.event_id(sg, a, d)
            if not sp.issparse(raw_mat):
                raise TypeError('Sparse substitution blocks should be SciPy sparse matrices.')
            if tuple(raw_mat.shape) != self.shape[:2]:
                raise ValueError('Sparse substitution block {} has shape {}; expected {}.'.format(raw_key, raw_mat.shape, self.shape[:2]))
            coo = raw_mat.tocoo(copy=False)
            if coo.nnz == 0:
                continue
            keep = np.asarray(coo.data != 0)
            if not keep.any():
                continue
            rows.append(np.asarray(coo.row[keep], dtype=np.int64))
            cols.append(event_id * self.num_site + np.asarray(coo.col[keep], dtype=np.int64))
            vals.append(np.asarray(coo.data[keep], dtype=self.dtype))
        packed_shape = (self.num_branch, self.num_event * self.num_site)
        if not rows:
            return sp.csr_matrix(packed_shape, dtype=self.dtype)
        matrix = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=packed_shape,
            dtype=self.dtype,
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix

    @property
    def blocks(self):
        # Compatibility only.  Materializing this view recreates per-event CSR
        # objects, so core reducers deliberately never access it.
        if self._blocks_cache is None:
            event_ids = np.unique(self.matrix.indices // self.num_site)
            out = {}
            for event_id in event_ids.tolist():
                sg, a, d = self.decode_event_ids(event_id)
                out[(int(sg), int(a), int(d))] = self.get_block(sg, a, d)
            self._blocks_cache = MappingProxyType(out)
        return self._blocks_cache

    @property
    def nnz(self):
        return int(self.matrix.nnz)

    @property
    def nbytes(self):
        return int(self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes)

    @property
    def dense_nbytes(self):
        return int(self.size * self.dtype.itemsize)

    @property
    def compression_ratio(self):
        if self.nbytes == 0:
            return np.inf if self.dense_nbytes > 0 else 1.0
        return float(self.dense_nbytes) / float(self.nbytes)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def density(self):
        return 0.0 if self.size == 0 else self.nnz / self.size

    def _coordinates(self):
        coo = self.matrix.tocoo(copy=False)
        event_ids, site = np.divmod(np.asarray(coo.col, dtype=np.int64), self.num_site)
        sg, a, d = self.decode_event_ids(event_ids)
        return np.asarray(coo.row, dtype=np.int64), site, sg, a, d, coo.data

    def to_dense(self):
        out = np.zeros(shape=self.shape, dtype=self.dtype)
        branch, site, sg, a, d, data = self._coordinates()
        out[branch, site, sg, a, d] = data
        return out

    def sum(self, axis=None):
        axes = _normalize_sum_axes(axis=axis, ndim=5)
        out_dtype = _sum_result_dtype(self.dtype)
        if axes is None:
            return np.asarray(self.matrix.data, dtype=out_dtype).sum(dtype=out_dtype)
        if len(axes) == 0:
            raise ValueError('axis=() would require a dense 5D tensor; use to_dense() explicitly.')
        remaining_axes = tuple(ax for ax in range(5) if ax not in axes)
        out = np.zeros(tuple(self.shape[ax] for ax in remaining_axes), dtype=out_dtype)
        if self.nnz == 0:
            return out
        branch, site, sg, a, d, data = self._coordinates()
        coords = (branch, site, sg, a, d)
        values = np.asarray(data, dtype=out_dtype)
        if not remaining_axes:
            out[...] = values.sum(dtype=out_dtype)
        else:
            np.add.at(out, tuple(coords[ax] for ax in remaining_axes), values)
        return out

    def get_block(self, sg, a, d):
        event_id = self.event_id(sg, a, d)
        start = event_id * self.num_site
        block = self.matrix[:, start:start + self.num_site]
        block.data.flags.writeable = False
        block.indices.flags.writeable = False
        block.indptr.flags.writeable = False
        return block

    def project(self, stat):
        if stat == 'spe2spe':
            return self.matrix if self.matrix.dtype == np.float64 else self.matrix.astype(np.float64)
        event_ids = self.matrix.indices // self.num_site
        sites = self.matrix.indices % self.num_site
        sg, a, d = self.decode_event_ids(event_ids)
        if stat == 'any2any':
            feature = sg
            num_feature = self.num_group
        elif stat == 'spe2any':
            feature = sg * self.num_state_from + a
            num_feature = self.num_group * self.num_state_from
        elif stat == 'any2spe':
            feature = sg * self.num_state_to + d
            num_feature = self.num_group * self.num_state_to
        else:
            raise ValueError('Unsupported projection statistic: {}'.format(stat))
        indices = np.asarray(feature * self.num_site + sites, dtype=self.matrix.indices.dtype)
        projection = sp.csr_matrix(
            (
                np.asarray(self.matrix.data, dtype=np.float64).copy(),
                indices,
                self.matrix.indptr.copy(),
            ),
            shape=(self.num_branch, int(num_feature) * self.num_site),
            dtype=np.float64,
        )
        projection.sum_duplicates()
        projection.eliminate_zeros()
        projection.sort_indices()
        return projection

    def project_any2any(self, sg):
        return self.project('any2any')[:, int(sg) * self.num_site:(int(sg) + 1) * self.num_site]

    def project_spe2any(self, sg, a):
        feature = int(sg) * self.num_state_from + int(a)
        return self.project('spe2any')[:, feature * self.num_site:(feature + 1) * self.num_site]

    def project_any2spe(self, sg, d):
        feature = int(sg) * self.num_state_to + int(d)
        return self.project('any2spe')[:, feature * self.num_site:(feature + 1) * self.num_site]

    @classmethod
    def from_packed(cls, shape, dtype, data, indices, indptr):
        num_col = int(np.prod(shape[2:])) * int(shape[1])
        matrix = sp.csr_matrix((data, indices, indptr), shape=(int(shape[0]), num_col), dtype=dtype)
        return cls(shape=shape, dtype=dtype, matrix=matrix)

    @classmethod
    def from_dense(cls, sub_tensor, tol=0):
        arr = np.asarray(sub_tensor)
        if arr.ndim != 5:
            raise ValueError('sub_tensor should have 5 dimensions: [branch,site,group,from,to].')
        if tol < 0:
            raise ValueError('tol should be >= 0.')
        if tol == 0:
            mask = arr != 0
        else:
            with np.errstate(invalid='ignore'):
                mask = np.abs(arr) > tol
            mask |= np.isnan(arr)
        branch, site, sg, a, d = np.nonzero(mask)
        num_site, num_from, num_to = arr.shape[1], arr.shape[3], arr.shape[4]
        event_ids = (sg * num_from + a) * num_to + d
        cols = event_ids * num_site + site
        matrix = sp.csr_matrix(
            (arr[branch, site, sg, a, d], (branch, cols)),
            shape=(arr.shape[0], int(np.prod(arr.shape[2:])) * num_site),
            dtype=arr.dtype,
        )
        return cls(shape=arr.shape, dtype=arr.dtype, matrix=matrix)


def dense_to_sparse_substitution_tensor(sub_tensor, tol=0):
    return SparseSubstitutionTensor.from_dense(sub_tensor=sub_tensor, tol=tol)


def sparse_to_dense_substitution_tensor(sparse_tensor):
    return sparse_tensor.to_dense()


def summarize_sparse_sub_tensor(sparse_tensor, mode):
    num_branch, num_site, num_group, num_state_from, num_state_to = sparse_tensor.shape
    dtype = sparse_tensor.dtype
    out_dtype = np.int64 if np.issubdtype(dtype, np.bool_) else np.result_type(
        dtype, np.int64 if np.issubdtype(dtype, np.integer) else dtype
    )
    branch, site, sg, a, d, values = sparse_tensor._coordinates()
    values = np.asarray(values, dtype=out_dtype)
    if mode == 'spe2spe':
        sub_bg = np.zeros((num_branch, num_group, num_state_from, num_state_to), dtype=out_dtype)
        sub_sg = np.zeros((num_site, num_group, num_state_from, num_state_to), dtype=out_dtype)
        np.add.at(sub_bg, (branch, sg, a, d), values)
        np.add.at(sub_sg, (site, sg, a, d), values)
    elif mode == 'spe2any':
        sub_bg = np.zeros((num_branch, num_group, num_state_from), dtype=out_dtype)
        sub_sg = np.zeros((num_site, num_group, num_state_from), dtype=out_dtype)
        np.add.at(sub_bg, (branch, sg, a), values)
        np.add.at(sub_sg, (site, sg, a), values)
    elif mode == 'any2spe':
        sub_bg = np.zeros((num_branch, num_group, num_state_to), dtype=out_dtype)
        sub_sg = np.zeros((num_site, num_group, num_state_to), dtype=out_dtype)
        np.add.at(sub_bg, (branch, sg, d), values)
        np.add.at(sub_sg, (site, sg, d), values)
    elif mode == 'any2any':
        sub_bg = np.zeros((num_branch, num_group), dtype=out_dtype)
        sub_sg = np.zeros((num_site, num_group), dtype=out_dtype)
        np.add.at(sub_bg, (branch, sg), values)
        np.add.at(sub_sg, (site, sg), values)
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))
    return sub_bg, sub_sg


# Retained as a monkeypatch seam for downstream tests/extensions that used the
# former block-by-block converter.  Dense conversion is now packed directly.
def _can_use_cython_dense_block_to_csr(*_args, **_kwargs):
    return False
