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
        self.wrapped.save(path)
        return

    def name(self):
        return self.wrapped.name()

    def get_epoch(self):
        return self.wrapped.get_epoch()

    def set_epoch(self, unsigned epoch):
        self.wrapped.set_epoch(epoch)
        return

    def get_learning_rate_scaling(self):
        return self.wrapped.get_learning_rate_scaling()

    def set_learning_rate_scaling(self, float scale):
        self.wrapped.set_learning_rate_scaling(scale)
        return

    def get_weight_decay(self):
        return self.wrapped.get_weight_decay()

    def set_weight_decay(self, float strength):
        self.wrapped.set_weight_decay(strength)
        return

    def get_gradient_clipping(self):
        return self.wrapped.get_gradient_clipping()

    def set_gradient_clipping(self, float threshold):
        self.wrapped.set_gradient_clipping(threshold)
        return

    def add_parameter(self, _Parameter param):
        self.wrapped.add_parameter(param.wrapped[0])
        return

    def reset_gradients(self):
        self.wrapped.reset_gradients()
        return

    def update(self):
        self.wrapped.update()
        return
