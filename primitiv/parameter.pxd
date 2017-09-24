from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.tensor cimport Tensor, _Tensor
from primitiv.shape cimport Shape, _Shape
from primitiv.device cimport Device
from primitiv.initializer cimport Initializer, _Initializer


cdef extern from "primitiv/parameter.h" namespace "primitiv":
    cdef cppclass Parameter:
        Parameter(Parameter &&src)
        Parameter()
        Parameter(const string &name, const Shape &shape, Device &device)
        Parameter(const string &name, const Shape &shape, const vector[float] &value, Device &device)
        Parameter(const string &name, const Shape &shape, const Initializer &init, Device &device)
        bool valid()
        void reset_value(const vector[float] &value)
        void reset_value(const Initializer &init)
        void reset_gradient()
        void add_stats(const string &name, const Shape &shape)
        bool has_stats(const string &name)
        const string &name()
        const Shape &shape()
        Device &device()
        Tensor &value()
        Tensor &gradient()
        Tensor &stats(const string &name)
        void save(const string &path, bool with_stats)

cdef extern from "primitiv/parameter.h" namespace "primitiv::Parameter":
    Parameter load(const string &path, bool with_stats, Device &device)


cdef class _Parameter:
    cdef Parameter *ptr


cdef inline _Parameter wrapParameter(Parameter *ptr):
    cdef _Parameter parameter = _Parameter.__new__(_Parameter)
    parameter.ptr = ptr
    return parameter
