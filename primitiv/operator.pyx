from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.cast cimport const_cast

from primitiv.device cimport wrapDevice, _Device
from primitiv.tensor cimport Tensor, wrapTensor
from primitiv.shape cimport _Shape
from primitiv.graph cimport _Graph, wrapNode, Node
from primitiv.parameter cimport _Parameter
from primitiv.trainer cimport wrapTrainer


cdef class opNode:

    @staticmethod
    def input_vector(_Shape shape, vector[float] data, _Device dev = _Device.get_default_device(), _Graph g = None):
        if g != None:
            return wrapNode(Node_input_vector(shape.ptr, data, dev.ptr[0], g.ptr[0]))
        else:
            return wrapNode(Node_input_vector(shape.ptr, data, dev.ptr[0]))

    @staticmethod
    def input_parameter(_Parameter param, _Graph g = None):
        if g != None:
            return wrapNode(Node_input_parameter(param.ptr[0], g.ptr[0]))
        else:
            return wrapNode(Node_input_parameter(param.ptr[0]))


cdef class opTensor:

    @staticmethod
    def input_vector(_Shape shape, vector[float] data, _Device dev = _Device.get_default_device()):
        return wrapTensor(Tensor_input_vector(shape.ptr, data, dev.ptr[0]))

    @staticmethod
    def input_parameter(_Parameter param):
        return wrapTensor(Tensor_input_parameter(param.ptr[0]))
