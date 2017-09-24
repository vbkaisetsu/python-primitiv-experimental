from primitiv.tensor cimport _Tensor


cdef class _Constant(_Initializer):
    def __cinit__(self, float k):
        self.ptr_constant = new Constant(k)
        self.ptr = self.ptr_constant

    def __dealloc__(self):
        del self.ptr_constant

    def apply(self, _Tensor x):
        self.ptr_constant.apply(x.ptr)
        return


cdef class _Uniform(_Initializer):
    def __cinit__(self, float lower, float upper):
        self.ptr_uniform = new Uniform(lower, upper)
        self.ptr = self.ptr_uniform

    def __dealloc__(self):
        del self.ptr_uniform

    def apply(self, _Tensor x):
        self.ptr_uniform.apply(x.ptr)
        return


cdef class _Normal(_Initializer):
    def __cinit__(self, float mean, float sd):
        self.ptr_normal = new Normal(mean, sd)
        self.ptr = self.ptr_normal

    def __dealloc__(self):
        del self.ptr_normal

    def apply(self, _Tensor x):
        self.ptr_normal.apply(x.ptr)
        return


cdef class _Identity(_Initializer):
    def __cinit__(self):
        self.ptr_identity = new Identity()
        self.ptr = self.ptr_identity

    def __dealloc__(self):
        del self.ptr_identity

    def apply(self, _Tensor x):
        self.ptr_identity.apply(x.ptr)
        return


cdef class _XavierUniform(_Initializer):
    def __cinit__(self, scale = 1.0):
        self.ptr_xavieruniform = new XavierUniform(scale)
        self.ptr = self.ptr_xavieruniform

    def __dealloc__(self):
        del self.ptr_xavieruniform

    def apply(self, _Tensor x):
        self.ptr_xavieruniform.apply(x.ptr)
        return


cdef class _XavierNormal(_Initializer):
    def __cinit__(self, float scale = 1.0):
        self.ptr_xaviernormal = new XavierNormal(scale)
        self.ptr = self.ptr_xaviernormal

    def __dealloc__(self):
        del self.ptr_xaviernormal

    def apply(self, _Tensor x):
        self.ptr_xaviernormal.apply(x.ptr)
        return
