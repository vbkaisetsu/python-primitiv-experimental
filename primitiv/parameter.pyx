from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.device cimport _Device, wrapDevice, get_default_device
from primitiv.tensor cimport wrapTensor
from primitiv.shape cimport wrapShape
from primitiv.parameter cimport Parameter_load, Parameter


cdef class _Parameter:
    def __cinit__(self, str name, shape, init = None, _Device device = None):
        cdef _Shape _shape = shape if isinstance(shape, _Shape) else _Shape(shape)
        if init == None:
            if device == None:
                self.wrapped = new Parameter(<string> name.encode("utf-8"), _shape.wrapped, get_default_device())
            else:
                self.wrapped = new Parameter(<string> name.encode("utf-8"), _shape.wrapped, device.wrapped[0])
        elif isinstance(init, list):
            if device == None:
                self.wrapped = new Parameter(<string> name.encode("utf-8"), _shape.wrapped, <vector[float]> init, get_default_device())
            else:
                self.wrapped = new Parameter(<string> name.encode("utf-8"), _shape.wrapped, <vector[float]> init, device.wrapped[0])
        elif isinstance(init, _Initializer):
            if device == None:
                self.wrapped = new Parameter(<string> name.encode("utf-8"), _shape.wrapped, (<_Initializer> init).wrapped[0], get_default_device())
            else:
                self.wrapped = new Parameter(<string> name.encode("utf-8"), _shape.wrapped, (<_Initializer> init).wrapped[0], device.wrapped[0])
        else:
            raise TypeError("Argument 'init' has incorrect type (list or Initializer)")
        if self.wrapped is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    #def copy(self):
        #return wrapParameter(new Parameter(self.wrapped[0]))

    def valid(self):
        return self.wrapped.valid()

    def reset_value_by_vector(self, vector[float] &value):
        self.wrapped.reset_value(value)
        return

    def reset_value_by_initializer(self, _Initializer init):
        self.wrapped.reset_value(init.wrapped[0])
        return

    def reset_gradient(self):
        self.wrapped.reset_gradient()
        return

    def add_stats(self, str name, _Shape shape):
        cdef _Shape _shape = shape if isinstance(shape, _Shape) else _Shape(shape)
        self.wrapped.add_stats(<string> name.encode("utf-8"), _shape.wrapped)
        return

    def has_stats(self, str name):
        return self.wrapped.has_stats(name)

    def name(self):
        return self.wrapped.name().decode("utf-8")

    def shape(self):
        return wrapShape(self.wrapped.shape())

    def device(self):
        return wrapDevice(&self.wrapped.device())

    def value(self):
        return wrapTensor(self.wrapped.value())

    def gradient(self):
        return wrapTensor(self.wrapped.gradient())

    def stats(self, str name):
        return wrapTensor(self.wrapped.stats(<string> name.encode("utf-8")))

    def save(self, str path, bool with_stats = True):
        self.wrapped.save(<string> path.encode("utf-8"), with_stats)
        return

    @staticmethod
    def load(str path, bool with_stats = True, _Device device = None):
        if device == None:
            return wrapParameter(Parameter_load(<string> path.encode("utf-8"), with_stats, get_default_device()))
        else:
            return wrapParameter(Parameter_load(<string> path.encode("utf-8"), with_stats, device.wrapped[0]))
