from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.shape cimport Shape


cdef class _Shape:

    def __cinit__(self, dims = None, unsigned batch = 1):
        if dims == None:
            self.ptr = Shape()
        else:
            self.ptr = Shape(<vector[unsigned]> dims, <unsigned> batch)

    def __dealloc__(self):
        return

    def depth(self):
        return self.ptr.depth()

    def batch(self):
        return self.ptr.batch()

    def volume(self):
        return self.ptr.volume()

    def lower_volume(self, unsigned dim):
        return self.ptr.lower_volume(dim)

    def size(self):
        return self.ptr.size()

    def __str__(self):
        return self.ptr.to_string()

    def __getitem__(self, unsigned i):
        return self.ptr[i]

    def __eq__(_Shape self, _Shape rhs):
        return self.ptr == rhs.ptr

    def __ne__(_Shape self, _Shape rhs):
        return self.ptr != rhs.ptr

    def has_batch(self):
        return self.ptr.has_batch()

    def has_compatible_batch(self, _Shape rhs):
        return self.ptr.has_compatible_batch(rhs.ptr)

    def is_scalar(self):
        return self.ptr.is_scalar()

    def is_row_vector(self):
        return self.ptr.is_row_vector()

    def is_matrix(self):
        return self.ptr.is_matrix()

    def has_same_dims(self, _Shape rhs):
        return self.ptr.has_same_dims(rhs.ptr)

    def has_same_loo_dims(self, _Shape rhs, unsigned dim):
        return self.ptr.has_same_loo_dims(rhs.ptr, dim)

    def resize_dim(self, unsigned dim, unsigned m):
        self.ptr.resize_dim(dim, m)
        return self

    def resize_batch(self, unsigned batch):
        self.ptr.resize_batch(batch)
        return self

    def update_dim(self, unsigned dim, unsigned m):
        self.ptr.update_dim(dim, m)

    def update_batch(self, unsigned batch):
        self.ptr.update_batch(batch)
