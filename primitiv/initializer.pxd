from primitiv.tensor cimport Tensor


cdef extern from "primitiv/initializer.h" namespace "primitiv":
    cdef cppclass Initializer:
        Initializer() except +
        void apply(Tensor &x) except +


cdef class _Initializer:
    cdef Initializer *ptr


cdef inline _Initializer wrapInitializer(Initializer *ptr) except +:
    cdef _Initializer initializer = _Initializer.__new__(_Initializer)
    initializer.ptr = ptr
    return initializer
