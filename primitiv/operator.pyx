from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.device cimport wrapDevice, _Device
from primitiv.tensor cimport Tensor, wrapTensor
from primitiv.shape cimport _Shape
from primitiv.graph cimport _Graph, wrapNode, Node, _Node
from primitiv.parameter cimport _Parameter
from primitiv.trainer cimport wrapTrainer


class _operators:

    @staticmethod
    def input_vector(_Shape shape, vector[float] data, _Device device = None, _Graph g = None):
        if device == None:
            device = _Device.get_default_device()
        if g != None:
            return wrapNode(Node_input_vector(shape.wrapped, data, device.wrapped[0], g.wrapped[0]))
        else:
            return wrapNode(Node_input_vector(shape.wrapped, data, device.wrapped[0]))

    @staticmethod
    def input_parameter(_Parameter param, _Graph g = None):
        if g != None:
            return wrapNode(Node_input_parameter(param.wrapped[0], g.wrapped[0]))
        else:
            return wrapNode(Node_input_parameter(param.wrapped[0]))

    @staticmethod
    def pick(_Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(Node_pick(x.wrapped, ids, dim))

    @staticmethod
    def slice(_Node x, unsigned dim, unsigned lower, unsigned upper):
        return wrapNode(Node_slice(x.wrapped, dim, lower, upper))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[Node] vec
        cdef _Node x
        for x in xs:
            vec.push_back(x.wrapped)
            return wrapNode(Node_concat(vec, dim))

    @staticmethod
    def reshape(_Node x, _Shape new_shape):
        return wrapNode(Node_reshape(x.wrapped, new_shape.wrapped))

    @staticmethod
    def flatten(_Node x):
        return wrapNode(Node_flatten(x.wrapped))

    @staticmethod
    def transpose(_Node x):
        return wrapNode(Node_transpose(x.wrapped))

    @staticmethod
    def matmul(_Node a, _Node b):
        return wrapNode(Node_matmul(a.wrapped, b.wrapped))

    @staticmethod
    def sqrt(_Node x):
        return wrapNode(Node_sqrt(x.wrapped))

    @staticmethod
    def exp(_Node x):
        return wrapNode(Node_exp(x.wrapped))

    @staticmethod
    def log(_Node x):
        return wrapNode(Node_log(x.wrapped))

    @staticmethod
    def tanh(_Node x):
        return wrapNode(Node_tanh(x.wrapped))

    @staticmethod
    def sigmoid(_Node x):
        return wrapNode(Node_sigmoid(x.wrapped))

    @staticmethod
    def softplus(_Node x):
        return wrapNode(Node_softplus(x.wrapped))

    @staticmethod
    def sin(_Node x):
        return wrapNode(Node_sin(x.wrapped))

    @staticmethod
    def cos(_Node x):
        return wrapNode(Node_cos(x.wrapped))

    @staticmethod
    def tan(_Node x):
        return wrapNode(Node_tan(x.wrapped))

    @staticmethod
    def relu(_Node x):
        return wrapNode(Node_relu(x.wrapped))

    @staticmethod
    def lrelu(_Node x):
        return wrapNode(Node_lrelu(x.wrapped))

    @staticmethod
    def prelu(_Node x, float a):
        return wrapNode(Node_prelu(x.wrapped, a))

    @staticmethod
    def elu(_Node x, float a):
        return wrapNode(Node_elu(x.wrapped, a))

    @staticmethod
    def selu(_Node x, float a, float s):
        return wrapNode(Node_selu(x.wrapped, a, s))

    @staticmethod
    def sum(_Node x, unsigned dim):
        return wrapNode(Node_sum(x.wrapped, dim))

    @staticmethod
    def mean(_Node x, unsigned dim):
        return wrapNode(Node_mean(x.wrapped, dim))

    @staticmethod
    def broadcast(_Node x, unsigned dim, unsigned size):
        return wrapNode(Node_broadcast(x.wrapped, dim, size))

    @staticmethod
    def logsumexp(_Node x, unsigned dim):
        return wrapNode(Node_logsumexp(x.wrapped, dim))

    @staticmethod
    def log_softmax(_Node x, unsigned dim):
        return wrapNode(Node_log_softmax(x.wrapped, dim))

    @staticmethod
    def softmax(_Node x, unsigned dim):
        return wrapNode(Node_softmax(x.wrapped, dim))

    @staticmethod
    def softmax_cross_entropy(_Node x, _Node t, unsigned dim):
        return wrapNode(Node_softmax_cross_entropy(x.wrapped, t.wrapped, dim))

    @staticmethod
    def softmax_cross_entropy(_Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(Node_softmax_cross_entropy(x.wrapped, ids, dim))

    @staticmethod
    def constant(_Shape shape, float k, _Device device, _Graph g):
        return wrapNode(Node_constant(shape.wrapped, k, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def zeros(_Shape shape, _Device device, _Graph g):
        return wrapNode(Node_zeros(shape.wrapped, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def ones(_Shape shape, _Device device, _Graph g):
        return wrapNode(Node_ones(shape.wrapped, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def identity(unsigned size, _Device device, _Graph g):
        return wrapNode(Node_identity(size, device.wrapped[0], g.wrapped[0]))

    @staticmethod
    def constant(_Shape shape, float k, _Device device):
        return wrapNode(Node_constant(shape.wrapped, k, device.wrapped[0]))

    @staticmethod
    def zeros(_Shape shape, _Device device):
        return wrapNode(Node_zeros(shape.wrapped, device.wrapped[0]))

    @staticmethod
    def ones(_Shape shape, _Device device):
        return wrapNode(Node_ones(shape.wrapped, device.wrapped[0]))

    @staticmethod
    def identity(unsigned size, _Device device):
        return wrapNode(Node_identity(size, device.wrapped[0]))


    class batch:
        @staticmethod
        def sum(_Node x):
            return wrapNode(Node_batch_sum(x.wrapped))

        @staticmethod
        def mean(_Node x):
            return wrapNode(Node_batch_mean(x.wrapped))

        @staticmethod
        def normalize(_Node x):
            return wrapNode(Node_batch_normalize(x.wrapped))

    class random:
        @staticmethod
        def bernoulli(_Shape shape, float p, _Device device, _Graph g):
            return wrapNode(Node_random_bernoulli(shape.wrapped, p, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def bernoulli(_Shape shape, float p, _Device device):
            return wrapNode(Node_random_bernoulli(shape.wrapped, p, device.wrapped[0]))

        @staticmethod
        def uniform(_Shape shape, float lower, float upper, _Device device, _Graph g):
            return wrapNode(Node_random_uniform(shape.wrapped, lower, upper, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def uniform(_Shape shape, float lower, float upper, _Device device):
            return wrapNode(Node_random_uniform(shape.wrapped, lower, upper, device.wrapped[0]))

        @staticmethod
        def normal(_Shape shape, float mean, float sd, _Device device, _Graph g):
            return wrapNode(Node_random_normal(shape.wrapped, mean, sd, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def normal(_Shape shape, float mean, float sd, _Device device):
            return wrapNode(Node_random_normal(shape.wrapped, mean, sd, device.wrapped[0]))

        @staticmethod
        def log_normal(_Shape shape, float mean, float sd, _Device device, _Graph g):
            return wrapNode(Node_random_log_normal(shape.wrapped, mean, sd, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def log_normal(_Shape shape, float mean, float sd, _Device device):
            return wrapNode(Node_random_log_normal(shape.wrapped, mean, sd, device.wrapped[0]))

        @staticmethod
        def gumbel(_Shape shape, float mu, float beta, _Device device, _Graph g):
            return wrapNode(Node_random_gumbel(shape.wrapped, mu, beta, device.wrapped[0], g.wrapped[0]))

        @staticmethod
        def gumbel(_Shape shape, float mu, float beta, _Device device):
            return wrapNode(Node_random_gumbel(shape.wrapped, mu, beta, device.wrapped[0]))

    @staticmethod
    def dropout(_Node x, float rate, bool enabled):
        return wrapNode(Node_dropout(x.wrapped, rate, enabled))
