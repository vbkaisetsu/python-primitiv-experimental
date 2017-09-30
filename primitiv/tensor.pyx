from libcpp.vector cimport vector

from primitiv.device cimport wrapDevice
from primitiv.shape cimport _Shape, wrapShape, normShape


cdef class _Tensor:

    def __init__(self, src = None):
        if src == None:
            self.wrapped = Tensor()
        else:
            self.wrapped = Tensor((<_Tensor> src).wrapped)
        return

    def valid(self):
        return self.wrapped.valid()

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return wrapDevice(&self.wrapped.device())

    #def data(self):
        #return self.wrapped.data()

    def to_vector(self):
        return self.wrapped.to_vector()

    def __iter__(self):
        return iter(self.wrapped.to_vector())

    def reset(self, float k):
        self.wrapped.reset(k)

    #def reset_by_array(self, vector[float] values):
        #self.wrapped.reset_by_array(values)

    def reset_by_vector(self, vector[float] values):
        self.wrapped.reset_by_vector(values)

    def reshape(self, _Shape new_shape):
        self.wrapped = self.wrapped.reshape(normShape(new_shape).wrapped)
        return self

    def flatten(self):
        return wrapTensor(self.wrapped.flatten())

    def __imul__(self, float k):
        self.wrapped = tensor_inplace_multiply_const(self.wrapped, k)
        return self

    def __iadd__(self, _Tensor x):
        self.wrapped = tensor_inplace_add(self.wrapped, x.wrapped)
        return self

    def __isub__(self, _Tensor x):
        self.wrapped = tensor_inplace_subtract(self.wrapped, x.wrapped)
        return self
