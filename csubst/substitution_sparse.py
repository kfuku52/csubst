import numpy as np
import scipy.sparse as sp
try:
    from csubst import substitution_sparse_cy
except Exception:  # pragma: no cover - Cython extension is optional
    substitution_sparse_cy = None


class SparseSubstitutionTensor:
    def __init__(self, shape, dtype, blocks):
        if len(shape) != 5:
            raise ValueError('SparseSubstitutionTensor shape should have 5 dimensions.')
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.blocks = dict(blocks)

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
        # Keep numpy-like behavior for legacy call sites that aggregate dense tensors.
        return self.to_dense().sum(axis=axis)

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
    if mode == 'spe2spe':
        sub_bg = np.zeros(shape=(num_branch, num_group, num_state_from, num_state_to), dtype=dtype)
        sub_sg = np.zeros(shape=(num_site, num_group, num_state_from, num_state_to), dtype=dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg, a, d] = np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg, a, d] = np.asarray(mat.sum(axis=0)).reshape(-1)
    elif mode == 'spe2any':
        sub_bg = np.zeros(shape=(num_branch, num_group, num_state_from), dtype=dtype)
        sub_sg = np.zeros(shape=(num_site, num_group, num_state_from), dtype=dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg, a] += np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg, a] += np.asarray(mat.sum(axis=0)).reshape(-1)
    elif mode == 'any2spe':
        sub_bg = np.zeros(shape=(num_branch, num_group, num_state_to), dtype=dtype)
        sub_sg = np.zeros(shape=(num_site, num_group, num_state_to), dtype=dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg, d] += np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg, d] += np.asarray(mat.sum(axis=0)).reshape(-1)
    elif mode == 'any2any':
        sub_bg = np.zeros(shape=(num_branch, num_group), dtype=dtype)
        sub_sg = np.zeros(shape=(num_site, num_group), dtype=dtype)
        for (sg, a, d), mat in sparse_tensor.blocks.items():
            sub_bg[:, sg] += np.asarray(mat.sum(axis=1)).reshape(-1)
            sub_sg[:, sg] += np.asarray(mat.sum(axis=0)).reshape(-1)
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))
    return sub_bg, sub_sg
