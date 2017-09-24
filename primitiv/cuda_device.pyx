from primitiv.cuda_device cimport num_devices as CUDADevice_num_devices
from primitiv.cuda_device cimport CUDADevice
from primitiv.device cimport _Device, Device


cdef class _CUDADevice(_Device):

    @staticmethod
    def num_devices():
        CUDADevice_num_devices()

    def __cinit__(self, unsigned device_id, rng_seed = None):
        if rng_seed == None:
            self.ptr_cudadevice = new CUDADevice(device_id)
        else:
            self.ptr_cudadevice = new CUDADevice(device_id, <unsigned> rng_seed)
            self.ptr = <Device*> self.ptr_cudadevice

    def __dealloc__(self):
        del self.ptr_cudadevice

    def dump_description(self):
        self.ptr_cudadevice.dump_description()
        return

    def type(self):
        return self.ptr_cudadevice.type()
