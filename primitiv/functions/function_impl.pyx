from libcpp.vector cimport vector

from primitiv.shape cimport Shape, _Shape, wrapShape
from primitiv.tensor cimport wrapTensor, _Tensor
from primitiv.device cimport wrapDevice, _Device
from primitiv.function cimport _Function
from primitiv.parameter cimport _Parameter


cdef class _Input(_Function):
    def __cinit__(self, _Shape shape, vector[float] data, _Device device):
        self.ptr_input = new Input(shape.ptr, data, device.ptr[0])
        self.ptr = self.ptr_input

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_input.get_device())

    def name(self):
        return self.ptr_input.name().decode("utf-8")


cdef class _ParameterInput(_Function):
    def __cinit__(self, _Parameter param):
        self.ptr_parameterinput = new ParameterInput(param.ptr[0])
        self.ptr = self.ptr_parameterinput

    def get_device(self):
        return wrapDevice(self.ptr_parameterinput.get_device())

    def get_inner_value(self):
        return wrapTensor(self.ptr_parameterinput.get_inner_value()[0])

    def name(self):
        return self.ptr_parameterinput.name().decode("utf-8")


cdef class _Copy(_Function):
    def __init__(self, _Device device):
        self.ptr_copy = new Copy(device.ptr[0])
        self.ptr = self.ptr_copy

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_copy.get_device())

    def name(self):
        return self.ptr_input.name().decode("utf-8")


cdef class _Constant(_Function):
    def __cinit__(self, _Shape shape, float k, _Device device):
        self.ptr_constant = new Constant(shape.ptr, k, device.ptr[0])
        self.ptr = self.ptr_constant

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_constant.get_device())

    def name(self):
        return self.ptr_constant.name().decode("utf-8")


cdef class _IdentityMatrix(_Function):
    def __cinit__(self, unsigned size, _Device device):
        self.ptr_identitymatrix = new IdentityMatrix(size, device.ptr[0])
        self.ptr = self.ptr_identitymatrix

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_identitymatrix.get_device())

    def name(self):
        return self.ptr_identitymatrix.name().decode("utf-8")


cdef class _RandomBernoulli(_Function):
    def __cinit__(self, _Shape shape, float p, _Device device):
        self.ptr_randombernoulli = new RandomBernoulli(shape.ptr, p, device.ptr[0])
        self.ptr = self.ptr_randombernoulli

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_randombernoulli.get_device())

    def name(self):
        return self.ptr_randombernoulli.name().decode("utf-8")


cdef class _RandomUniform(_Function):
    def __cinit__(self, _Shape shape, float lower, float upper, _Device device):
        self.ptr_randomuniform = new RandomUniform(shape.ptr, lower, upper, device.ptr[0])
        self.ptr = self.ptr_randomuniform

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_randomuniform.get_device())

    def name(self):
        return self.ptr_randomuniform.name().decode("utf-8")


cdef class _RandomNormal(_Function):
    def __cinit__(self, _Shape shape, float mean, float sd, _Device device):
        self.ptr_randomnormal = new RandomNormal(shape.ptr, mean, sd, device.ptr[0])
        self.ptr = self.ptr_randomnormal

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_randomnormal.get_device())

    def name(self):
        return self.ptr_randomnormal.name().decode("utf-8")


cdef class _RandomLogNormal(_Function):
    def __cinit__(self, _Shape shape, float mean, float sd, _Device device):
        self.ptr_randomlognormal = new RandomLogNormal(shape.ptr, mean, sd, device.ptr[0])
        self.ptr = self.ptr_randomlognormal

    def __dealloc__(self):
        del self.ptr

    def get_device(self):
        return wrapDevice(self.ptr_randomlognormal.get_device())

    def name(self):
        return self.ptr_randomlognormal.name().decode("utf-8")


cdef class _Pick(_Function):
    def __cinit__(self, vector[unsigned] ids, unsigned dim):
        self.ptr_pick = new Pick(ids, dim)
        self.ptr = self.ptr_pick

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_pick.name().decode("utf-8")


cdef class _Slice(_Function):
    def __cinit__(self, unsigned dim, unsigned lower, unsigned upper):
        self.ptr_slice = new Slice(dim, lower, upper)
        self.ptr = self.ptr_slice

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_slice.name().decode("utf-8")


cdef class _Concat(_Function):
    def __cinit__(self, unsigned dim):
        self.ptr_concat = new Concat(dim)
        self.ptr = self.ptr_concat

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_concat.name().decode("utf-8")


cdef class _Reshape(_Function):
    def __cinit__(self, _Shape shape):
        self.ptr_reshape = new Reshape(shape.ptr)
        self.ptr = self.ptr_reshape

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_reshape.name().decode("utf-8")


cdef class _Sum(_Function):
    def __cinit__(self, unsigned dim):
        self.ptr_sum = new Sum(dim)
        self.ptr = self.ptr_sum

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_sum.name().decode("utf-8")


cdef class _LogSumExp(_Function):
    def __cinit__(self, unsigned dim):
        self.ptr_logsumexp = new LogSumExp(dim)
        self.ptr = self.ptr_logsumexp

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_logsumexp.name().decode("utf-8")


cdef class _Broadcast(_Function):
    def __cinit__(self, unsigned dim, unsigned size):
        self.ptr_broadcast = new Broadcast(dim, size)
        self.ptr = self.ptr_broadcast

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_broadcast.name().decode("utf-8")


cdef class _SoftmaxCrossEntropy(_Function):
    def __cinit__(self, unsigned dim):
        self.ptr_softmaxcrossentropy = new SoftmaxCrossEntropy(dim)
        self.ptr = self.ptr_softmaxcrossentropy

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_softmaxcrossentropy.name().decode("utf-8")


cdef class _SparseSoftmaxCrossEntropy(_Function):
    def __cinit__(self, vector[unsigned] ids, unsigned dim):
        self.ptr_sparsesoftmaxcrossentropy = new SparseSoftmaxCrossEntropy(ids, dim)
        self.ptr = self.ptr_sparsesoftmaxcrossentropy

    def __dealloc__(self):
        del self.ptr

    def name(self):
        return self.ptr_sparsesoftmaxcrossentropy.name().decode("utf-8")
