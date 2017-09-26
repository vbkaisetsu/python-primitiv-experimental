from libcpp.string cimport string

from primitiv.device cimport Device
from primitiv.trainer cimport Trainer, _Trainer


cdef extern from "primitiv/trainer_impl.h" namespace "primitiv::trainers":
    cdef cppclass SGD(Trainer):
        SGD(const float eta)
        float eta()

    cdef cppclass Adam(Trainer):
        Adam(float alpha, float beta1, float beta2, float eps)
        float alpha()
        float beta1()
        float beta2()
        float eps()


cdef class _SGD(_Trainer):
    cdef SGD *ptr_sgd


cdef class _Adam(_Trainer):
    cdef Adam *ptr_adam
