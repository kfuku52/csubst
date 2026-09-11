"""Exact site-major temporary arrays with bounded reads and worker sharing.

These arrays are deliberately not NumPy array-likes: converting them implicitly
would materialize the complete payload. Consumers read a site or a small block.
The creating object owns its temporary directory; process workers borrow it
until the parent finishes their work. Completed arrays can be shared by copies.
"""

import copy
import os
import tempfile

import numpy as np

from csubst import runtime


class SiteArray:
    def __init__(self, shape, dtype, site_axis):
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self.site_axis = int(site_axis)
        self.site_shape = self.shape[:site_axis] + self.shape[site_axis + 1:]
        self.site_size = int(np.prod(self.site_shape))
        self._directory = tempfile.TemporaryDirectory(
            prefix='csubst-site-array-', dir=runtime.get_run_tempdir(create=False))
        self.filename = os.path.join(self._directory.name, 'values.bin')
        with open(self.filename, 'wb'):
            pass
        self._sealed = False
        self._cached_site = None
        self._cached_values = None

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def write_block(self, start, values):
        """Write a site-first block; every non-site axis keeps its original order."""
        if self._sealed:
            raise ValueError('Completed site arrays are read-only.')
        values = np.asarray(values, dtype=self.dtype)
        if (values.shape[1:] != self.site_shape or start < 0
                or start + len(values) > self.shape[self.site_axis]):
            raise ValueError('Invalid site-array write shape or range.')
        with open(self.filename, 'r+b') as handle:
            handle.seek(int(start) * self.site_size * self.dtype.itemsize)
            values.tofile(handle)

    def seal(self):
        if os.path.getsize(self.filename) != self.nbytes:
            raise ValueError('Incomplete site-array payload.')
        self._sealed = True

    def read_sites(self, start, stop):
        if not self._sealed:
            raise ValueError('Cannot read an unfinished site array.')
        if not 0 <= start <= stop <= self.shape[self.site_axis]:
            raise IndexError('Site range is outside the stored array.')
        count = (int(stop) - int(start)) * self.site_size
        with open(self.filename, 'rb') as handle:
            handle.seek(int(start) * self.site_size * self.dtype.itemsize)
            values = np.fromfile(handle, dtype=self.dtype, count=count)
        if len(values) != count:
            raise ValueError('Truncated site-array payload.')
        return values.reshape((int(stop) - int(start),) + self.site_shape)

    def read_site(self, site):
        site = int(site)
        if site != self._cached_site:
            self._cached_values = self.read_sites(site, site + 1)[0]
            self._cached_values.flags.writeable = False
            self._cached_site = site
        return self._cached_values

    def iter_site_blocks(self, max_bytes=8 * 1024**2):
        count = max(1, int(max_bytes) // max(1, self.site_size * self.dtype.itemsize))
        for start in range(0, self.shape[self.site_axis], count):
            stop = min(start + count, self.shape[self.site_axis])
            yield start, self.read_sites(start, stop)

    def __getstate__(self):
        if not self._sealed:
            raise ValueError('Cannot share an unfinished site array.')
        return {key: value for key, value in self.__dict__.items()
                if key not in ('_directory', '_cached_site', '_cached_values')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached_site = None
        self._cached_values = None

    def __deepcopy__(self, memo):
        if not self._sealed:
            raise ValueError('Cannot share an unfinished site array.')
        memo[id(self)] = self
        return self


class SiteEventTensor(SiteArray):
    """Branch/site/group/from/to events plus small branch/site totals."""

    def __init__(self, shape, dtype=np.float64):
        if len(shape) != 5:
            raise ValueError('Stored events require five axes.')
        super().__init__(shape, dtype, site_axis=1)
        self._branch_site = np.zeros(self.shape[:2], dtype=np.float64)
        self.min_sub_pp = 0.0

    def write_block(self, start, values):
        super().write_block(start, values)
        self._branch_site[:, start:start + len(values)] = values.sum(axis=(2, 3, 4)).T

    def read_sites(self, start, stop):
        values = super().read_sites(start, stop)
        if self.min_sub_pp:
            values[values < self.min_sub_pp] = 0
        return values

    @property
    def branch_site(self):
        if self._branch_site is None:
            self._branch_site = np.zeros(self.shape[:2], dtype=np.float64)
            for start, values in self.iter_site_blocks():
                self._branch_site[:, start:start + len(values)] = values.sum(axis=(2, 3, 4)).T
        return self._branch_site

    def thresholded(self, threshold):
        # Keep the owning directory alive in the parent; workers only borrow it.
        result = copy.copy(self)
        if hasattr(self, '_directory'):
            result._directory = self._directory
        result.min_sub_pp = max(float(threshold), self.min_sub_pp)
        result._branch_site = None
        result._cached_site = None
        result._cached_values = None
        return result
