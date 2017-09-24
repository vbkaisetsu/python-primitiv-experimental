from libcpp.vector cimport vector
from libcpp.string cimport string

from primitiv.tensor cimport Tensor, _Tensor
from primitiv.shape cimport Shape, _Shape
from primitiv.device cimport Device
from primitiv.function cimport Function, _Function
from primitiv.parameter cimport Parameter


cdef extern from "primitiv/function_impl.h" namespace "primitiv::functions":
    cdef cppclass Input(Function):
        Input(const Shape &shape, const vector[float] &data, Device &device);

    cdef cppclass ParameterInput(Function):
        ParameterInput(Parameter &param)

    cdef cppclass Copy(Function):
        Copy(Device &device)

    cdef cppclass Constant(Function):
        Constant(const Shape &shape, float k, Device &device)

    cdef cppclass IdentityMatrix(Function):
        IdentityMatrix(unsigned size, Device &device)

    cdef cppclass RandomBernoulli(Function):
        RandomBernoulli(const Shape &shape, float p, Device &device)

    cdef cppclass RandomUniform(Function):
        RandomUniform(const Shape &shape, float lower, float upper, Device &device)

    cdef cppclass RandomNormal(Function):
        RandomNormal(const Shape &shape, float mean, float sd, Device &device)

    cdef cppclass RandomLogNormal(Function):
        RandomLogNormal(const Shape &shape, float mean, float sd, Device &device)

    cdef cppclass Pick(Function):
        Pick(const vector[unsigned] &ids, unsigned dim)

    cdef cppclass Slice(Function):
        Slice(unsigned dim, unsigned lower, unsigned upper)

    cdef cppclass Concat(Function):
        Concat(unsigned dim)

    cdef cppclass Reshape(Function):
        Reshape(const Shape &shape)

    cdef cppclass Sum(Function):
        Sum(unsigned dim)

    cdef cppclass LogSumExp(Function):
        LogSumExp(unsigned dim)

    cdef cppclass Broadcast(Function):
        Broadcast(unsigned dim, unsigned size)

    cdef cppclass SoftmaxCrossEntropy(Function):
        SoftmaxCrossEntropy(unsigned dim)

    cdef cppclass SparseSoftmaxCrossEntropy(Function):
        SparseSoftmaxCrossEntropy(const vector[unsigned] ids, unsigned dim)


cdef class _Input(_Function):
    cdef Input *ptr_input

cdef class _ParameterInput(_Function):
    cdef ParameterInput *ptr_parameterinput

cdef class _Copy(_Function):
    cdef Copy *ptr_copy

cdef class _Constant(_Function):
    cdef Constant *ptr_constant

cdef class _IdentityMatrix(_Function):
    cdef IdentityMatrix *ptr_identitymatrix

cdef class _RandomBernoulli(_Function):
    cdef RandomBernoulli *ptr_randombernoulli

cdef class _RandomUniform(_Function):
    cdef RandomUniform *ptr_randomuniform

cdef class _RandomNormal(_Function):
    cdef RandomNormal *ptr_randomnormal

cdef class _RandomLogNormal(_Function):
    cdef RandomLogNormal *ptr_randomlognormal

cdef class _Pick(_Function):
    cdef Pick *ptr_pick

cdef class _Slice(_Function):
    cdef Slice *ptr_slice

cdef class _Concat(_Function):
    cdef Concat *ptr_concat

cdef class _Reshape(_Function):
    cdef Reshape *ptr_reshape

cdef class _Sum(_Function):
    cdef Sum *ptr_sum

cdef class _LogSumExp(_Function):
    cdef LogSumExp *ptr_logsumexp

cdef class _Broadcast(_Function):
    cdef Broadcast *ptr_broadcast

cdef class _SoftmaxCrossEntropy(_Function):
    cdef SoftmaxCrossEntropy *ptr_softmaxcrossentropy

cdef class _SparseSoftmaxCrossEntropy(_Function):
    cdef SparseSoftmaxCrossEntropy *ptr_sparsesoftmaxcrossentropy
