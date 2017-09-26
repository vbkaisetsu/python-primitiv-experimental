from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from primitiv.device cimport Device
from primitiv.shape cimport Shape
from primitiv.parameter cimport Parameter, _Parameter


cdef extern from "primitiv/trainer.h" namespace "primitiv":
    cdef cppclass Trainer:
        Trainer(Trainer &&)
        Trainer()
        void save(const string &path)
        string name()
        unsigned get_epoch()
        void set_epoch(unsigned epoch)
        float get_learning_rate_scaling()
        void set_learning_rate_scaling(float scale)
        float get_weight_decay()
        void set_weight_decay(float strength)
        float get_gradient_clipping()
        void set_gradient_clipping(float threshold)
        void add_parameter(Parameter &param)
        void reset_gradients()
        void update()


cdef extern from "primitiv/trainer.h" namespace "primitiv::Trainer":
    shared_ptr[Trainer] load(const string &path)


cdef class _Trainer:
    cdef Trainer *ptr


cdef inline _Trainer wrapTrainer(Trainer *ptr):
    cdef _Trainer trainer = _Trainer.__new__(_Trainer)
    trainer.ptr = ptr
    return trainer
