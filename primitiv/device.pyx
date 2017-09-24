from libcpp.vector cimport vector

from primitiv.device cimport get_default_device as Device_get_default_device
from primitiv.device cimport set_default_device as Device_set_default_device
from primitiv.shape cimport _Shape
from primitiv.tensor cimport wrapTensor, _Tensor


cdef class _Device:

    @staticmethod
    def get_default_device():
        return wrapDevice(&Device_get_default_device())

    @staticmethod
    def set_default_device(_Device dev):
        Device_set_default_device(dev.ptr[0])
        return

    def new_tensor(self, _Shape shape, float k = 0):
        return wrapTensor(self.ptr.new_tensor(shape.ptr, k))

        # Tensor new_tensor(const Shape &shape)
        # Tensor new_tensor(const Shape &shape, float k)

    #def new_tensor_by_array(self, _Shape shape, float values[])
        #return wrapTensor(self.ptr.new_tensor_by_array(const Shape &shape, const float values[]))

    def new_tensor_by_vector(self, _Shape shape, vector[float] values):
        return wrapTensor(self.ptr.new_tensor_by_vector(shape.ptr, values))

    def copy_tensor(self, _Tensor x):
        return wrapTensor(self.ptr.copy_tensor(x.ptr))

    def identity(self, unsigned size):
        return wrapTensor(self.ptr.identity(size))

    def random_bernoulli(self, _Shape shape, float p):
        return wrapTensor(self.ptr.random_bernoulli(shape.ptr, p))

    def random_uniform(self, _Shape shape, float lower, float upper):
        return wrapTensor(self.ptr.random_uniform(shape.ptr, lower, upper))

    def random_normal(self, _Shape shape, float mean, float sd):
        return wrapTensor(self.ptr.random_normal(shape.ptr, mean, sd))

    def random_log_normal(self, _Shape shape, float mean, float sd):
        return wrapTensor(self.ptr.random_log_normal(shape.ptr, mean, sd))

    def pick_fw(self, _Tensor x, vector[unsigned] ids, unsigned dim):
        return wrapTensor(self.ptr.pick_fw(x.ptr, ids, dim))

    def slice_fw(self, _Tensor x, unsigned dim, unsigned lower, unsigned upper):
        return wrapTensor(self.ptr.slice_fw(x.ptr, dim, lower, upper))

    #def concat_fw(self, vector[const Tensor *] xs, unsigned dim):
    #    return wrapTensor(self.ptr.concat_fw(xs, dim))

    def pick_bw(self, _Tensor gy, vector[unsigned] ids, unsigned dim, _Tensor gx):
        self.ptr.pick_bw(gy.ptr, ids, dim, gx.ptr)
        return

    def slice_bw(self, _Tensor gy, unsigned dim, unsigned offset, _Tensor gx):
        self.ptr.slice_bw(gy.ptr, dim, offset, gx.ptr)
        return

    def negate_fw(self, _Tensor x):
        return wrapTensor(self.ptr.negate_fw(x.ptr))

    def sqrt_fw(self, _Tensor x):
        return wrapTensor(self.ptr.sqrt_fw(x.ptr))

    def exp_fw(self, _Tensor x):
        return wrapTensor(self.ptr.exp_fw(x.ptr))

    def log_fw(self, _Tensor x):
        return wrapTensor(self.ptr.log_fw(x.ptr))

    def tanh_fw(self, _Tensor x):
        return wrapTensor(self.ptr.tanh_fw(x.ptr))

    def sigmoid_fw(self, _Tensor x):
        return wrapTensor(self.ptr.sigmoid_fw(x.ptr))

    def softplus_fw(self, _Tensor x):
        return wrapTensor(self.ptr.softplus_fw(x.ptr))

    def sin_fw(self, _Tensor x):
        return wrapTensor(self.ptr.sin_fw(x.ptr))

    def cos_fw(self, _Tensor x):
        return wrapTensor(self.ptr.cos_fw(x.ptr))

    def tan_fw(self, _Tensor x):
        return wrapTensor(self.ptr.tan_fw(x.ptr))

    def transpose_fw(self, _Tensor x):
        return wrapTensor(self.ptr.transpose_fw(x.ptr))

    def sqrt_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.sqrt_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def exp_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.exp_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def log_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.log_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def tanh_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.tanh_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def sigmoid_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.sigmoid_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def softplus_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.softplus_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def sin_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.sin_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def cos_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.cos_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def tan_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.tan_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def transpose_bw(self, _Tensor x, _Tensor y, _Tensor gy, _Tensor gx):
        self.ptr.transpose_bw(x.ptr, y.ptr, gy.ptr, gx.ptr)
        return

    def add_const_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.add_const_fw(x.ptr, k))

    def subtract_const_r_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.subtract_const_r_fw(x.ptr, k))

    def subtract_const_l_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.subtract_const_l_fw(x.ptr, k))

    def multiply_const_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.multiply_const_fw(x.ptr, k))

    def divide_const_r_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.divide_const_r_fw(x.ptr, k))

    def divide_const_l_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.divide_const_l_fw(x.ptr, k))

    def prelu_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.prelu_fw(x.ptr, k))

    def elu_fw(self, _Tensor x, float k):
        return wrapTensor(self.ptr.elu_fw(x.ptr, k))

    def add_const_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.add_const_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def subtract_const_r_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.subtract_const_r_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def subtract_const_l_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.subtract_const_l_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def multiply_const_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.multiply_const_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def divide_const_r_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.divide_const_r_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def divide_const_l_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.divide_const_l_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def prelu_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.prelu_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def elu_bw(self, _Tensor x, _Tensor y, _Tensor gy, float k, _Tensor gx):
        self.ptr.elu_bw(x.ptr, y.ptr, gy.ptr, k, gx.ptr)
        return

    def add_scalar_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.ptr.add_scalar_fw(x.ptr, k.ptr))

    def subtract_scalar_r_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.ptr.subtract_scalar_r_fw(x.ptr, k.ptr))

    def subtract_scalar_l_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.ptr.subtract_scalar_l_fw(x.ptr, k.ptr))

    def multiply_scalar_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.ptr.multiply_scalar_fw(x.ptr, k.ptr))

    def divide_scalar_r_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.ptr.divide_scalar_r_fw(x.ptr, k.ptr))

    def divide_scalar_l_fw(self, _Tensor x, _Tensor k):
        return wrapTensor(self.ptr.divide_scalar_l_fw(x.ptr, k.ptr))

    def add_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.ptr.add_fw(a.ptr, b.ptr))

    def subtract_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.ptr.subtract_fw(a.ptr, b.ptr))

    def multiply_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.ptr.multiply_fw(a.ptr, b.ptr))

    def divide_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.ptr.divide_fw(a.ptr, b.ptr))

    def matmul_fw(self, _Tensor a, _Tensor b):
        return wrapTensor(self.ptr.matmul_fw(a.ptr, b.ptr))

    def add_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.ptr.add_bw(a.ptr, b.ptr, y.ptr, gy.ptr, ga.ptr, gb.ptr)
        return

    def subtract_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.ptr.subtract_bw(a.ptr, b.ptr, y.ptr, gy.ptr, ga.ptr, gb.ptr)
        return

    def multiply_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.ptr.multiply_bw(a.ptr, b.ptr, y.ptr, gy.ptr, ga.ptr, gb.ptr)
        return

    def divide_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.ptr.divide_bw(a.ptr, b.ptr, y.ptr, gy.ptr, ga.ptr, gb.ptr)
        return

    def matmul_bw(self, _Tensor a, _Tensor b, _Tensor y, _Tensor gy, _Tensor ga, _Tensor gb):
        self.ptr.matmul_bw(a.ptr, b.ptr, y.ptr, gy.ptr, ga.ptr, gb.ptr)
        return

    def sum_fw(self, _Tensor x, unsigned dim):
        return wrapTensor(self.ptr.sum_fw(x.ptr, dim))

    def logsumexp_fw(self, _Tensor x, unsigned dim):
        return wrapTensor(self.ptr.logsumexp_fw(x.ptr, dim))

    def broadcast_fw(self, _Tensor x, unsigned dim, unsigned size):
        return wrapTensor(self.ptr.broadcast_fw(x.ptr, dim, size))

    def batch_sum_fw(self, _Tensor x):
        return wrapTensor(self.ptr.batch_sum_fw(x.ptr))

    def inplace_multiply_const(self, float k, _Tensor x):
        self.ptr.inplace_multiply_const(k, x.ptr)
        return

    def inplace_add(self, _Tensor x, _Tensor y):
        self.ptr.inplace_add(x.ptr, y.ptr)
        return

    def inplace_subtract(self, _Tensor x, _Tensor y):
        self.ptr.inplace_subtract(x.ptr, y.ptr)
        return
