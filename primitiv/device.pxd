from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.tensor cimport Tensor
from primitiv.shape cimport Shape


cdef extern from "primitiv/device.h" namespace "primitiv":
    cdef cppclass Device:
        enum DeviceType:
            DEVICE_TYPE_CPU = 0x0
            DEVICE_TYPE_CUDA = 0x10000
        Tensor new_tensor(const Shape &shape)
        Tensor new_tensor(const Shape &shape, float k)
        # Tensor new_tensor_by_array(const Shape &shape, const float values[])
        Tensor new_tensor_by_vector(const Shape &shape, const vector[float] &values)
        Tensor copy_tensor(const Tensor &x)
        Tensor identity(unsigned size)
        Tensor random_bernoulli(const Shape &shape, float p)
        Tensor random_uniform(const Shape &shape, float lower, float upper)
        Tensor random_normal(const Shape &shape, float mean, float sd)
        Tensor random_log_normal(const Shape &shape, float mean, float sd)
        Tensor pick_fw(const Tensor &x, const vector[unsigned] &ids, unsigned dim)
        Tensor slice_fw(const Tensor &x, unsigned dim, unsigned lower, unsigned upper)
        Tensor concat_fw(const vector[const Tensor *] &xs, unsigned dim)
        void pick_bw(const Tensor &gy, const vector[unsigned] &ids, unsigned dim, Tensor &gx)
        void slice_bw(const Tensor &gy, unsigned dim, unsigned offset, Tensor &gx)
        Tensor negate_fw(const Tensor &x)
        Tensor sqrt_fw(const Tensor &x)
        Tensor exp_fw(const Tensor &x)
        Tensor log_fw(const Tensor &x)
        Tensor tanh_fw(const Tensor &x)
        Tensor sigmoid_fw(const Tensor &x)
        Tensor softplus_fw(const Tensor &x)
        Tensor sin_fw(const Tensor &x)
        Tensor cos_fw(const Tensor &x)
        Tensor tan_fw(const Tensor &x)
        Tensor transpose_fw(const Tensor &x)
        void sqrt_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void exp_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void log_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void tanh_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void sigmoid_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void softplus_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void sin_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void cos_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void tan_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        void transpose_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx)
        Tensor add_const_fw(const Tensor &x, float k)
        Tensor subtract_const_r_fw(const Tensor &x, float k)
        Tensor subtract_const_l_fw(const Tensor &x, float k)
        Tensor multiply_const_fw(const Tensor &x, float k)
        Tensor divide_const_r_fw(const Tensor &x, float k)
        Tensor divide_const_l_fw(const Tensor &x, float k)
        Tensor prelu_fw(const Tensor &x, float k)
        Tensor elu_fw(const Tensor &x, float k)
        void add_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void subtract_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void subtract_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void multiply_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void divide_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void divide_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void prelu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        void elu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx)
        Tensor add_scalar_fw(const Tensor &x, const Tensor &k)
        Tensor subtract_scalar_r_fw(const Tensor &x, const Tensor &k)
        Tensor subtract_scalar_l_fw(const Tensor &x, const Tensor &k)
        Tensor multiply_scalar_fw(const Tensor &x, const Tensor &k)
        Tensor divide_scalar_r_fw(const Tensor &x, const Tensor &k)
        Tensor divide_scalar_l_fw(const Tensor &x, const Tensor &k)
        Tensor add_fw(const Tensor &a, const Tensor &b)
        Tensor subtract_fw(const Tensor &a, const Tensor &b)
        Tensor multiply_fw(const Tensor &a, const Tensor &b)
        Tensor divide_fw(const Tensor &a, const Tensor &b)
        Tensor matmul_fw(const Tensor &a, const Tensor &b)
        void add_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb)
        void subtract_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb)
        void multiply_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb)
        void divide_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb)
        void matmul_bw(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb)
        Tensor sum_fw(const Tensor &x, unsigned dim)
        Tensor logsumexp_fw(const Tensor &x, unsigned dim)
        Tensor broadcast_fw(const Tensor &x, unsigned dim, unsigned size)
        Tensor batch_sum_fw(const Tensor &x)
        void inplace_multiply_const(float k, Tensor &x)
        void inplace_add(const Tensor &x, Tensor &y)
        void inplace_subtract(const Tensor &x, Tensor &y)


cdef extern from "primitiv/device.h" namespace "primitiv::Device":
    cdef Device &get_default_device()
    cdef void set_default_device(Device &dev)


cdef class _Device:
    cdef Device *ptr


cdef inline _Device wrapDevice(Device *ptr):
    cdef _Device device = _Device.__new__(_Device)
    device.ptr = ptr
    return device
