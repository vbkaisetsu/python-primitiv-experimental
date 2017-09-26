from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.cast cimport const_cast

from primitiv.device cimport wrapDevice
from primitiv.shape import _Shape
from primitiv.trainer cimport wrapTrainer
from primitiv.trainer cimport load as Trainer_load


cdef class _Trainer:

    def load(self, string &path):
        return wrapTrainer(Trainer_load(path).get())

    def save(self, string &path):
        self.ptr.save(path)
        return

    def name(self):
        return self.ptr.name()

    def get_epoch(self):
        return self.ptr.get_epoch()

    def set_epoch(self, unsigned epoch):
        self.ptr.set_epoch(epoch)
        return

    def get_learning_rate_scaling(self):
        return self.ptr.get_learning_rate_scaling()

    def set_learning_rate_scaling(self, float scale):
        self.ptr.set_learning_rate_scaling(scale)
        return

    def get_weight_decay(self):
        return self.ptr.get_weight_decay()

    def set_weight_decay(self, float strength):
        self.ptr.set_weight_decay(strength)
        return

    def get_gradient_clipping(self):
        return self.ptr.get_gradient_clipping()

    def set_gradient_clipping(self, float threshold):
        self.ptr.set_gradient_clipping(threshold)
        return

    def add_parameter(self, _Parameter param):
        self.ptr.add_parameter(param.ptr[0])
        return

    def reset_gradients(self):
        self.ptr.reset_gradients()
        return

    def update(self):
        self.ptr.update()
        return
