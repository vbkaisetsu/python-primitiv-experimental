from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.cast cimport const_cast

from primitiv.device cimport wrapDevice
from primitiv.shape import _Shape
from primitiv.shape cimport _Shape, wrapShape

from primitiv.tensor cimport (Tensor,
                     tensor_inplace_multiply_const,
                     tensor_inplace_add,
                     tensor_inplace_subtract
)


cdef class _Tensor:

    def __cinit__(self, src = None):
        if src == None:
            self.ptr = Tensor()
        else:
            self.ptr = Tensor((<_Tensor> src).ptr)
        return

    def __dealloc__(self):
        return

    def valid(self):
        return self.ptr.valid()

    def shape(self):
        return wrapShape(self.ptr.shape())

    def device(self):
        return wrapDevice(&self.ptr.device())

    #def data(self):
        #return self.ptr.data()

    def to_vector(self):
        return self.ptr.to_vector()

    def __iter__(self):
        return iter(self.ptr.to_vector())

    def reset(self, float k):
        self.ptr.reset(k)

    #def reset_by_array(self, vector[float] values):
        #self.ptr.reset_by_array(values)

    def reset_by_vector(self, vector[float] values):
        self.ptr.reset_by_vector(values)

    def reshape(self, _Shape new_shape):
        self.ptr = self.ptr.reshape(new_shape.ptr)
        return self

    def flatten(self):
        return wrapTensor(self.ptr.flatten())

    def __imul__(self, float k):
        self.ptr = tensor_inplace_multiply_const(self.ptr, k)
        return self

    def __iadd__(self, _Tensor x):
        self.ptr = tensor_inplace_add(self.ptr, x.ptr)
        return self

    def __isub__(self, _Tensor x):
        self.ptr = tensor_inplace_subtract(self.ptr, x.ptr)
        return self
