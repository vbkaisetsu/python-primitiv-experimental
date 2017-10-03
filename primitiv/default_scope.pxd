from primitiv.device cimport Device, _Device
from primitiv.graph cimport Graph, _Graph


cdef extern from "primitiv/default_scope.h" namespace "primitiv":
    cdef cppclass DefaultScope[T]:
        DefaultScope() except +
        DefaultScope(T &obj) except +


cdef extern from "default_scope_wrapper.h" namespace "python_primitiv":
    cdef Device &DefaultScopeDevice_get() except +
    cdef size_t DefaultScopeDevice_size() except +
    cdef Graph &DefaultScopeGraph_get() except +
    cdef size_t DefaultScopeGraph_size() except +


cdef class _DefaultScopeDevice:
    cdef DefaultScope[Device] *wrapped
    cdef _Device obj


cdef class _DefaultScopeGraph:
    cdef DefaultScope[Graph] *wrapped
    cdef _Graph obj
