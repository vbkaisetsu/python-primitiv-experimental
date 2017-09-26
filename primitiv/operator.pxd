from libcpp.string cimport string
from libcpp.vector cimport vector

from primitiv.device cimport Device
from primitiv.graph cimport Graph, Node
from primitiv.tensor cimport Tensor
from primitiv.shape cimport Shape
from primitiv.parameter cimport Parameter


cdef extern from "operator_template_wrapper.h" namespace "python_primitiv":

    inline Node Node_input_vector(const Shape &shape, const vector[float] &data, Device &dev, Graph &g)
    inline Node Node_input_vector(const Shape &shape, const vector[float] &data, Device &dev)
    inline Tensor Tensor_input_vector(const Shape &shape, const vector[float] &data, Device &dev)
    inline Node Node_input_parameter(Parameter &param, Graph &g)
    inline Node Node_input_parameter(Parameter &param)
    inline Tensor Tensor_input_parameter(Parameter &param)
