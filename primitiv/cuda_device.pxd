from primitiv.device cimport Device, _Device


cdef extern from "primitiv/cuda_device.h" namespace "primitiv":
    cdef cppclass CUDADevice(Device):
        CUDADevice(unsigned device_id)
        CUDADevice(unsigned device_id, unsigned rng_seed)
        void dump_description()
        Device.DeviceType type()


cdef extern from "primitiv/cpu_device.h" namespace "primitiv::CUDADevice":
    cdef unsigned num_devices()


cdef class _CUDADevice(_Device):
    cdef CUDADevice *ptr_cudadevice


cdef inline _CUDADevice wrapCUDADevice(CUDADevice *ptr):
    cdef _CUDADevice cuda_device = _CUDADevice.__new__(_CUDADevice)
    cuda_device.ptr = ptr
    return cuda_device
