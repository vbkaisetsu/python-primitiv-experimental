from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.cpu_device cimport CPUDevice
from primitiv.device cimport _Device, wrapDevice
from primitiv.tensor cimport wrapTensor
from primitiv.shape cimport wrapShape
from primitiv.parameter cimport load as Parameter_load


cdef class _Parameter:
    def __cinit__(self):
        self.ptr = new Parameter()

    def __dealloc__(self):
        del self.ptr

    def copy(self):
        return wrapParameter(new Parameter(<Parameter&&> self.ptr[0]))

    @staticmethod
    def new(string name, _Shape shape, _Device device = None):
        if device == None:
            return wrapParameter(new Parameter(name, shape.ptr, (<_Device> _Device.get_default_device()).ptr[0]))
        else:
            return wrapParameter(new Parameter(name, shape.ptr, device.ptr[0]))

    @staticmethod
    def new_from_vector(string name, _Shape shape,  vector[float] value, _Device device = None):
        if device == None:
            return wrapParameter(new Parameter(name, shape.ptr, value, (<_Device> _Device.get_default_device()).ptr[0]))
        else:
            return wrapParameter(new Parameter(name, shape.ptr, value, device.ptr[0]))

    @staticmethod
    def new_from_initializer(string name, _Shape shape, _Initializer init, _Device device = None):
        if device == None:
            return wrapParameter(new Parameter(name, shape.ptr, init.ptr[0], (<_Device> _Device.get_default_device()).ptr[0]))
        else:
            return wrapParameter(new Parameter(name, shape.ptr, init.ptr[0], device.ptr[0]))

    def valid(self):
        return self.ptr.valid()

    def reset_value_by_vector(self, vector[float] &value):
        self.ptr.reset_value(value)
        return

    def reset_value_by_initializer(self, _Initializer init):
        self.ptr.reset_value(init.ptr[0])
        return

    def reset_gradient(self):
        self.ptr.reset_gradient()
        return

    def add_stats(self, string name, _Shape shape):
        self.ptr.add_stats(name, shape.ptr)
        return

    def has_stats(self, string name):
        return self.ptr.has_stats(name)

    def name(self):
        return self.ptr.name()

    def shape(self):
        return wrapShape(self.ptr.shape())

    def device(self):
        return wrapDevice(&self.ptr.device())

    def value(self):
        return wrapTensor(self.ptr.value())

    def gradient(self):
        return wrapTensor(self.ptr.gradient())

    def stats(self, string name):
        return wrapTensor(self.ptr.stats(name))

    def save(self, string path, bool with_stats = True):
        self.ptr.save(path, with_stats)
        return

    @staticmethod
    def load(string path, bool with_stats = True, _Device device = None):
        if device == None:
            return wrapParameter(new Parameter(Parameter_load(path, with_stats, (<_Device> _Device.get_default_device()).ptr[0])))
        else:
            return wrapParameter(new Parameter(Parameter_load(path, with_stats, device.ptr[0])))
