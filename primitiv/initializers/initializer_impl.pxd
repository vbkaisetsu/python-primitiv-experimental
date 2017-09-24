from primitiv.initializer cimport Initializer, _Initializer


cdef extern from "primitiv/initializer_impl.h" namespace "primitiv::initializers":
    cdef cppclass Constant(Initializer):
        Constant(float k)

    cdef cppclass Uniform(Initializer):
        Uniform(float lower, float upper)

    cdef cppclass Normal(Initializer):
        Normal(float mean, float sd)

    cdef cppclass Identity(Initializer):
        Identity()

    cdef cppclass XavierUniform(Initializer):
        XavierUniform(float scale)

    cdef cppclass XavierNormal(Initializer):
        XavierNormal(float scale)


cdef class _Constant(_Initializer):
    cdef Constant *ptr_constant

cdef class _Uniform(_Initializer):
    cdef Uniform *ptr_uniform

cdef class _Normal(_Initializer):
    cdef Normal *ptr_normal

cdef class _Identity(_Initializer):
    cdef Identity *ptr_identity

cdef class _XavierUniform(_Initializer):
    cdef XavierUniform *ptr_xavieruniform

cdef class _XavierNormal(_Initializer):
    cdef XavierNormal *ptr_xaviernormal
