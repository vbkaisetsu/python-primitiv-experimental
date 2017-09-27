from primitiv.device cimport Device, _Device


cdef extern from "primitiv/cuda_device.h" namespace "primitiv":
    cdef cppclass CUDADevice(Device):
        CUDADevice(unsigned device_id) except +
        CUDADevice(unsigned device_id, unsigned rng_seed) except +
        void dump_description() except +
        Device.DeviceType type() except +


cdef extern from "primitiv/cpu_device.h" namespace "primitiv::CUDADevice":
    cdef unsigned num_devices() except +


cdef class _CUDADevice(_Device) except +:
    cdef CUDADevice *ptr_cudadevice


cdef inline _CUDADevice wrapCUDADevice(CUDADevice *ptr) except +:
    cdef _CUDADevice cuda_device = _CUDADevice.__new__(_CUDADevice)
    cuda_device.ptr = ptr
    return cuda_device
