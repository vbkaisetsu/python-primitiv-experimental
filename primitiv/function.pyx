from libcpp.vector cimport vector

from primitiv.shape cimport Shape, _Shape, wrapShape
from primitiv.tensor cimport wrapTensor, _Tensor
from primitiv.device cimport wrapDevice


cdef class _Function:

    def forward_shape(self, args):
        cdef vector[const Shape *] vec
        cdef _Shape x
        for x in args:
            vec.push_back(&x.ptr)
        return wrapShape(self.ptr.forward_shape(vec))

    def get_device(self):
        return wrapDevice(self.ptr.get_device())

    def get_inner_value(self):
        cdef const Tensor *tensor = self.ptr.get_inner_value()
        if tensor == NULL:
            return None
        return wrapTensor(tensor[0])

    def forward(self, args):
        cdef vector[const Tensor *] vec
        cdef _Tensor x
        for x in args:
            vec.push_back(&x.ptr)
        return wrapTensor(self.ptr.forward(vec))

    def backward(self, _Tensor cur_value, _Tensor cur_grad, arg_values, arg_grads):
        cdef vector[const Tensor *] vec_arg_values
        cdef vector[Tensor *] vec_arg_grads
        cdef _Tensor x
        for x in arg_values:
            vec_arg_values.push_back(&x.ptr)
        for x in arg_grads:
            vec_arg_grads.push_back(&x.ptr)
        self.ptr.backward(cur_value.ptr, cur_grad.ptr, vec_arg_values, vec_arg_grads)
        return

    def name(self):
        return self.ptr.name().decode("utf-8")
