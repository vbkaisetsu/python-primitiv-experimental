from primitiv.device cimport Device, _Device


cdef extern from "primitiv/cpu_device.h" namespace "primitiv":
    cdef cppclass CPUDevice(Device):
        CPUDevice()
        CPUDevice(unsigned rng_seed)
        void dump_description()
        Device.DeviceType type()


cdef class _CPUDevice(_Device):
    cdef CPUDevice *ptr_cpudevice


cdef inline _CPUDevice wrapCPUDevice(CPUDevice *ptr):
    cdef _CPUDevice cpu_device = _CPUDevice.__new__(_CPUDevice)
    cpu_device.ptr = ptr
    return cpu_device
