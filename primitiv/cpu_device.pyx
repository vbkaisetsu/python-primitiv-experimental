from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.cast cimport const_cast

from primitiv.cpu_device cimport CPUDevice
from primitiv.device cimport _Device, Device


cdef class _CPUDevice(_Device):

    def __cinit__(self, rng_seed = None):
        if rng_seed == None:
            self.ptr_cpudevice = new CPUDevice()
        else:
            self.ptr_cpudevice = new CPUDevice(<unsigned> rng_seed)
        self.ptr = <Device*> self.ptr_cpudevice

    def __dealloc__(self):
        del self.ptr_cpudevice

    def dump_description(self):
        self.ptr_cpudevice.dump_description()
        return

    def type(self):
        return self.ptr_cpudevice.type()
