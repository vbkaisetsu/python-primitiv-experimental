from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.cpu_device cimport CPUDevice
from primitiv.device cimport _Device, wrapDevice, get_default_device
from primitiv.tensor cimport wrapTensor
from primitiv.shape cimport wrapShape
from primitiv.parameter cimport Parameter_load, Parameter


cdef class _Parameter:
    def __init__(self, str name, shape, init = None, _Device device = None):
        cdef _Shape _shape
        if isinstance(shape, list):
            _shape = _Shape(shape)
        elif isinstance(shape, _Shape):
            _shape = shape
        else:
            raise TypeError("Argument 'shape' has incorrect type (_Shape or list)")
        if init == None:
            if device == None:
                self.ptr = new Parameter(<string> name.encode("utf-8"), _shape.ptr, get_default_device())
            else:
                self.ptr = new Parameter(<string> name.encode("utf-8"), _shape.ptr, device.ptr[0])
        elif isinstance(init, list):
            if device == None:
                self.ptr = new Parameter(<string> name.encode("utf-8"), _shape.ptr, <vector[float]> init, get_default_device())
            else:
                self.ptr = new Parameter(<string> name.encode("utf-8"), _shape.ptr, <vector[float]> init, device.ptr[0])
        elif isinstance(init, _Initializer):
            if device == None:
                self.ptr = new Parameter(<string> name.encode("utf-8"), _shape.ptr, (<_Initializer> init).ptr[0], get_default_device())
            else:
                self.ptr = new Parameter(<string> name.encode("utf-8"), _shape.ptr, (<_Initializer> init).ptr[0], device.ptr[0])
        else:
            raise TypeError("Argument 'init' has incorrect type (list or Initializer)")

    def __dealloc__(self):
        del self.ptr

    #def copy(self):
        #return wrapParameter(new Parameter(self.ptr[0]))

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

    def add_stats(self, str name, _Shape shape):
        self.ptr.add_stats(<string> name.encode("utf-8"), shape.ptr)
        return

    def has_stats(self, str name):
        return self.ptr.has_stats(name)

    def name(self):
        return self.ptr.name().decode("utf-8")

    def shape(self):
        return wrapShape(self.ptr.shape())

    def device(self):
        return wrapDevice(&self.ptr.device())

    def value(self):
        return wrapTensor(self.ptr.value())

    def gradient(self):
        return wrapTensor(self.ptr.gradient())

    def stats(self, str name):
        return wrapTensor(self.ptr.stats(<string> name.encode("utf-8")))

    def save(self, str path, bool with_stats = True):
        self.ptr.save(<string> path.encode("utf-8"), with_stats)
        return

    @staticmethod
    def load(str path, bool with_stats = True, _Device device = None):
        if device == None:
            return wrapParameter(Parameter_load(<string> path.encode("utf-8"), with_stats, get_default_device()))
        else:
            return wrapParameter(Parameter_load(<string> path.encode("utf-8"), with_stats, device.ptr[0]))
