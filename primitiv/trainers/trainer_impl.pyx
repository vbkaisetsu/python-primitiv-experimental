from libcpp.string cimport string


cdef class _SGD(_Trainer):

    def __cinit__(self, float eta = 0.1):
        self.ptr_sgd = new SGD(eta)
        self.ptr = self.ptr_sgd

    def __dealloc__(self):
        del self.ptr_sgd

    def name(self):
        return self.ptr_sgd.name()

    def eta(self):
        return self.ptr_sgd.eta()


cdef class _Adam(_Trainer):
    def __cinit__(self, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8):
        self.ptr_adam = new Adam(alpha, beta1, beta2, eps)
        self.ptr = self.ptr_adam

    def __dealloc__(self):
        del self.ptr_adam

    def name(self):
        return self.ptr_adam.name()

    def alpha(self):
        return self.ptr_sgd.alpha()

    def beta1(self):
        return self.ptr_sgd.beta1()

    def beta2(self):
        return self.ptr_sgd.beta2()

    def eps(self):
        return self.ptr_sgd.eps()
