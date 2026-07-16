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
    if isinstance(axis, tuple):
        axes = axis
    else:
        axes = (axis,)
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
    def __init__(self, shape, dtype, blocks):
        if len(shape) != 5:
            raise ValueError('SparseSubstitutionTensor shape should have 5 dimensions.')
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        normalized_blocks = dict()
        for raw_key, raw_mat in blocks.items():
            if (not isinstance(raw_key, tuple)) or len(raw_key) != 3:
                raise ValueError('Sparse substitution block keys should be (group, from, to) tuples.')
            key = tuple(int(v) for v in raw_key)
            sg, a, d = key
            if not (0 <= sg < self.shape[2] and 0 <= a < self.shape[3] and 0 <= d < self.shape[4]):
                raise ValueError('Sparse substitution block key {} is out of bounds for shape {}.'.format(key, self.shape))
            if not sp.issparse(raw_mat):
                raise TypeError('Sparse substitution blocks should be SciPy sparse matrices.')
            if raw_mat.shape != self.shape[:2]:
                txt = 'Sparse substitution block {} has shape {}; expected {}.'
                raise ValueError(txt.format(key, raw_mat.shape, self.shape[:2]))
            mat = raw_mat
            needs_copy = (
                (not sp.isspmatrix_csr(mat))
                or (mat.dtype != self.dtype)
                or (not mat.has_canonical_format)
                or (not mat.has_sorted_indices)
                or ((mat.nnz > 0) and np.any(mat.data == 0))
            )
            if needs_copy:
                mat = sp.csr_matrix(mat, dtype=self.dtype, copy=True)
                mat.sum_duplicates()
                mat.eliminate_zeros()
                mat.sort_indices()
            if mat.nnz == 0:
                continue
            # Tensors are immutable after construction. Reducer projections
            # may therefore be cached safely without stale-data hazards.
            mat.data.flags.writeable = False
            mat.indices.flags.writeable = False
            mat.indptr.flags.writeable = False
            normalized_blocks[key] = mat
        self.blocks = MappingProxyType(normalized_blocks)

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
    def nnz(self):
        return int(sum(mat.nnz for mat in self.blocks.values()))

    @property
    def nbytes(self):
        return int(
            sum(
                mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
                for mat in self.blocks.values()
            )
        )

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
        if self.size == 0:
            return 0.0
        return self.nnz / self.size

    def to_dense(self):
        out = np.zeros(shape=self.shape, dtype=self.dtype)
        for (sg, a, d), mat in self.blocks.items():
            out[:, :, sg, a, d] = mat.toarray()
        return out

    def sum(self, axis=None):
        axes = _normalize_sum_axes(axis=axis, ndim=len(self.shape))
        out_dtype = _sum_result_dtype(self.dtype)
        if axes is None:
            total = np.zeros(shape=(), dtype=out_dtype)
            for mat in self.blocks.values():
                if mat.nnz == 0:
                    continue
                total[...] = total + np.asarray(mat.data, dtype=out_dtype).sum(dtype=out_dtype)
            return total[()]
        if len(axes) == 0:
            raise ValueError('axis=() would require a dense 5D tensor; use to_dense() explicitly.')

        remaining_axes = tuple(ax for ax in range(len(self.shape)) if ax not in axes)
        out_shape = tuple(self.shape[ax] for ax in remaining_axes)
        out = np.zeros(shape=out_shape, dtype=out_dtype)
        for (sg, a, d), mat in self.blocks.items():
            if mat.nnz == 0:
                continue
            coo = mat.tocoo()
            data = np.asarray(coo.data, dtype=out_dtype)
            if len(remaining_axes) == 0:
                out[...] = out + data.sum(dtype=out_dtype)
                continue
            coords_by_axis = {
                0: coo.row,
                1: coo.col,
                2: int(sg),
                3: int(a),
                4: int(d),
            }
            indices = tuple(coords_by_axis[ax] for ax in remaining_axes)
            has_branch_or_site_axis = (0 in remaining_axes) or (1 in remaining_axes)
            if has_branch_or_site_axis:
                np.add.at(out, indices, data)
            else:
                out[indices] = out[indices] + data.sum(dtype=out_dtype)
        return out

    def get_block(self, sg, a, d):
        key = (int(sg), int(a), int(d))
        mat = self.blocks.get(key)
        if mat is None:
            return sp.csr_matrix((self.num_branch, self.num_site), dtype=self.dtype)
        return mat

    def project_any2any(self, sg):
        mats = list()
        for a in range(self.num_state_from):
            for d in range(self.num_state_to):
                mat = self.blocks.get((int(sg), a, d))
                if mat is not None:
                    mats.append(mat)
        return _sum_sparse_mats(mats, self.num_branch, self.num_site, self.dtype)

    def project_spe2any(self, sg, a):
        mats = list()
        for d in range(self.num_state_to):
            mat = self.blocks.get((int(sg), int(a), d))
            if mat is not None:
                mats.append(mat)
        return _sum_sparse_mats(mats, self.num_branch, self.num_site, self.dtype)

    def project_any2spe(self, sg, d):
        mats = list()
        for a in range(self.num_state_from):
            mat = self.blocks.get((int(sg), a, int(d)))
            if mat is not None:
                mats.append(mat)
        return _sum_sparse_mats(mats, self.num_branch, self.num_site, self.dtype)

    @classmethod
    def from_dense(cls, sub_tensor, tol=0):
        arr = np.asarray(sub_tensor)
        if arr.ndim != 5:
            raise ValueError('sub_tensor should have 5 dimensions: [branch,site,group,from,to].')
        if tol < 0:
            raise ValueError('tol should be >= 0.')
        blocks = dict()
        if _can_use_cython_dense_to_sparse_direct_scan(arr=arr, tol=tol):
            for sg in range(arr.shape[2]):
                for a in range(arr.shape[3]):
                    for d in range(arr.shape[4]):
                        block = arr[:, :, int(sg), int(a), int(d)]
                        mat = _dense_block_to_csr(block=block, tol=tol)
                        if mat.nnz > 0:
                            blocks[(int(sg), int(a), int(d))] = mat
            return cls(shape=arr.shape, dtype=arr.dtype, blocks=blocks)
        if tol == 0:
            candidate_mask = np.any(arr != 0, axis=(0, 1))
        else:
            with np.errstate(invalid='ignore'):
                candidate_mask = np.any(np.abs(arr) > tol, axis=(0, 1))
            candidate_mask |= np.any(np.isnan(arr), axis=(0, 1))
        candidate_indices = np.argwhere(candidate_mask)
        for sg, a, d in candidate_indices:
            block = arr[:, :, int(sg), int(a), int(d)]
            mat = _dense_block_to_csr(block=block, tol=tol)
            if mat.nnz > 0:
                blocks[(int(sg), int(a), int(d))] = mat
        return cls(shape=arr.shape, dtype=arr.dtype, blocks=blocks)


def _can_use_cython_dense_to_sparse_direct_scan(arr, tol):
    if substitution_sparse_cy is None:
        return False
    if not isinstance(arr, np.ndarray):
        return False
    if arr.dtype != np.float64:
        return False
    if arr.ndim != 5:
        return False
    if not np.isfinite(tol):
        return False
    return True


def _can_use_cython_dense_block_to_csr(block, tol):
    if substitution_sparse_cy is None:
        return False
    if not isinstance(block, np.ndarray):
        return False
    if block.dtype != np.float64:
        return False
    if block.ndim != 2:
        return False
    if not np.isfinite(tol):
        return False
    return True


def _dense_block_to_csr(block, tol):
    if _can_use_cython_dense_block_to_csr(block=block, tol=tol):
        data, indices, indptr = substitution_sparse_cy.dense_block_to_csr_arrays_double(
            block,
            float(tol),
        )
        if data.size == 0:
            return sp.csr_matrix(block.shape, dtype=block.dtype)
        return sp.csr_matrix(
            (data, indices, indptr),
            shape=block.shape,
            dtype=block.dtype,
        )
    if tol > 0:
        dense_block = np.array(block, copy=True)
        dense_block[np.abs(dense_block) <= tol] = 0
        return sp.csr_matrix(dense_block)
    return sp.csr_matrix(block)


def _sum_sparse_mats(mats, nrow, ncol, dtype):
    if len(mats) == 0:
        return sp.csr_matrix((nrow, ncol), dtype=dtype)
    out = mats[0].copy()
    for mat in mats[1:]:
        out = out + mat
    return out


def dense_to_sparse_substitution_tensor(sub_tensor, tol=0):
    return SparseSubstitutionTensor.from_dense(sub_tensor=sub_tensor, tol=tol)


def sparse_to_dense_substitution_tensor(sparse_tensor):
    return sparse_tensor.to_dense()


def summarize_sparse_sub_tensor(sparse_tensor, mode):
    num_branch, num_site, num_group, num_state_from, num_state_to = sparse_tensor.shape
    dtype = sparse_tensor.dtype
    # Bool sparse tensors (e.g., --ml_anc yes) must be accumulated in an
    # integer/float container because scipy sum() returns integer counts.
    if np.issubdtype(dtype, np.bool_):
        out_dtype = np.int64
    else:
        out_dtype = np.result_type(dtype, np.int64 if np.issubdtype(dtype, np.integer) else dtype)
    if mode == 'spe2spe':
        sub_bg = np.zeros(shape=(num_branch, num_group, num_state_from, num_state_to), dtype=out_dtype)
        sub_sg = np.zeros(shape=(num_site, num_group, num_state_from, num_state_to), dtype=out_dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg, a, d] = np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg, a, d] = np.asarray(mat.sum(axis=0)).reshape(-1)
    elif mode == 'spe2any':
        sub_bg = np.zeros(shape=(num_branch, num_group, num_state_from), dtype=out_dtype)
        sub_sg = np.zeros(shape=(num_site, num_group, num_state_from), dtype=out_dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg, a] += np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg, a] += np.asarray(mat.sum(axis=0)).reshape(-1)
    elif mode == 'any2spe':
        sub_bg = np.zeros(shape=(num_branch, num_group, num_state_to), dtype=out_dtype)
        sub_sg = np.zeros(shape=(num_site, num_group, num_state_to), dtype=out_dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg, d] += np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg, d] += np.asarray(mat.sum(axis=0)).reshape(-1)
    elif mode == 'any2any':
        sub_bg = np.zeros(shape=(num_branch, num_group), dtype=out_dtype)
        sub_sg = np.zeros(shape=(num_site, num_group), dtype=out_dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg] += np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg] += np.asarray(mat.sum(axis=0)).reshape(-1)
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))
    return sub_bg, sub_sg
