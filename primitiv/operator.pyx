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
    def input_vector(_Shape shape, vector[float] data, _Device dev = None, _Graph g = None):
        if dev == None:
            dev = _Device.get_default_device()
        if g != None:
            return wrapNode(Node_input_vector(shape.ptr, data, dev.ptr[0], g.ptr[0]))
        else:
            return wrapNode(Node_input_vector(shape.ptr, data, dev.ptr[0]))

    @staticmethod
    def input_parameter(_Parameter param, _Graph g = None):
        if g != None:
            return wrapNode(Node_input_parameter(param.ptr[0], g.ptr[0]))
        else:
            return wrapNode(Node_input_parameter(param.ptr[0]))

    @staticmethod
    def pick(_Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(Node_pick(x.ptr, ids, dim))

    @staticmethod
    def slice(_Node x, unsigned dim, unsigned lower, unsigned upper):
        return wrapNode(Node_slice(x.ptr, dim, lower, upper))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[Node] vec
        cdef _Node x
        for x in xs:
            vec.push_back(x.ptr)
            return wrapNode(Node_concat(vec, dim))

    @staticmethod
    def reshape(_Node x, _Shape new_shape):
        return wrapNode(Node_reshape(x.ptr, new_shape.ptr))

    @staticmethod
    def flatten(_Node x):
        return wrapNode(Node_flatten(x.ptr))

    @staticmethod
    def transpose(_Node x):
        return wrapNode(Node_transpose(x.ptr))

    @staticmethod
    def matmul(_Node a, _Node b):
        return wrapNode(Node_matmul(a.ptr, b.ptr))

    @staticmethod
    def sqrt(_Node x):
        return wrapNode(Node_sqrt(x.ptr))

    @staticmethod
    def exp(_Node x):
        return wrapNode(Node_exp(x.ptr))

    @staticmethod
    def log(_Node x):
        return wrapNode(Node_log(x.ptr))

    @staticmethod
    def tanh(_Node x):
        return wrapNode(Node_tanh(x.ptr))

    @staticmethod
    def sigmoid(_Node x):
        return wrapNode(Node_sigmoid(x.ptr))

    @staticmethod
    def softplus(_Node x):
        return wrapNode(Node_softplus(x.ptr))

    @staticmethod
    def sin(_Node x):
        return wrapNode(Node_sin(x.ptr))

    @staticmethod
    def cos(_Node x):
        return wrapNode(Node_cos(x.ptr))

    @staticmethod
    def tan(_Node x):
        return wrapNode(Node_tan(x.ptr))

    @staticmethod
    def relu(_Node x):
        return wrapNode(Node_relu(x.ptr))

    @staticmethod
    def lrelu(_Node x):
        return wrapNode(Node_lrelu(x.ptr))

    @staticmethod
    def prelu(_Node x, float a):
        return wrapNode(Node_prelu(x.ptr, a))

    @staticmethod
    def elu(_Node x, float a):
        return wrapNode(Node_elu(x.ptr, a))

    @staticmethod
    def selu(_Node x, float a, float s):
        return wrapNode(Node_selu(x.ptr, a, s))

    @staticmethod
    def sum(_Node x, unsigned dim):
        return wrapNode(Node_sum(x.ptr, dim))

    @staticmethod
    def mean(_Node x, unsigned dim):
        return wrapNode(Node_mean(x.ptr, dim))

    @staticmethod
    def broadcast(_Node x, unsigned dim, unsigned size):
        return wrapNode(Node_broadcast(x.ptr, dim, size))

    @staticmethod
    def logsumexp(_Node x, unsigned dim):
        return wrapNode(Node_logsumexp(x.ptr, dim))

    @staticmethod
    def log_softmax(_Node x, unsigned dim):
        return wrapNode(Node_log_softmax(x.ptr, dim))

    @staticmethod
    def softmax(_Node x, unsigned dim):
        return wrapNode(Node_softmax(x.ptr, dim))

    @staticmethod
    def softmax_cross_entropy(_Node x, _Node t, unsigned dim):
        return wrapNode(Node_softmax_cross_entropy(x.ptr, t.ptr, dim))

    @staticmethod
    def softmax_cross_entropy(_Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(Node_softmax_cross_entropy(x.ptr, ids, dim))

    @staticmethod
    def constant(_Shape shape, float k, _Device dev, _Graph g):
        return wrapNode(Node_constant(shape.ptr, k, dev.ptr[0], g.ptr[0]))

    @staticmethod
    def zeros(_Shape shape, _Device dev, _Graph g):
        return wrapNode(Node_zeros(shape.ptr, dev.ptr[0], g.ptr[0]))

    @staticmethod
    def ones(_Shape shape, _Device dev, _Graph g):
        return wrapNode(Node_ones(shape.ptr, dev.ptr[0], g.ptr[0]))

    @staticmethod
    def identity(unsigned size, _Device dev, _Graph g):
        return wrapNode(Node_identity(size, dev.ptr[0], g.ptr[0]))

    @staticmethod
    def constant(_Shape shape, float k, _Device dev):
        return wrapNode(Node_constant(shape.ptr, k, dev.ptr[0]))

    @staticmethod
    def zeros(_Shape shape, _Device dev):
        return wrapNode(Node_zeros(shape.ptr, dev.ptr[0]))

    @staticmethod
    def ones(_Shape shape, _Device dev):
        return wrapNode(Node_ones(shape.ptr, dev.ptr[0]))

    @staticmethod
    def identity(unsigned size, _Device dev):
        return wrapNode(Node_identity(size, dev.ptr[0]))


    class batch:
        @staticmethod
        def sum(_Node x):
            return wrapNode(Node_batch_sum(x.ptr))

        @staticmethod
        def mean(_Node x):
            return wrapNode(Node_batch_mean(x.ptr))

        @staticmethod
        def normalize(_Node x):
            return wrapNode(Node_batch_normalize(x.ptr))

    class random:
        @staticmethod
        def bernoulli(_Shape shape, float p, _Device dev, _Graph g):
            return wrapNode(Node_random_bernoulli(shape.ptr, p, dev.ptr[0], g.ptr[0]))

        @staticmethod
        def bernoulli(_Shape shape, float p, _Device dev):
            return wrapNode(Node_random_bernoulli(shape.ptr, p, dev.ptr[0]))

        @staticmethod
        def uniform(_Shape shape, float lower, float upper, _Device dev, _Graph g):
            return wrapNode(Node_random_uniform(shape.ptr, lower, upper, dev.ptr[0], g.ptr[0]))

        @staticmethod
        def uniform(_Shape shape, float lower, float upper, _Device dev):
            return wrapNode(Node_random_uniform(shape.ptr, lower, upper, dev.ptr[0]))

        @staticmethod
        def normal(_Shape shape, float mean, float sd, _Device dev, _Graph g):
            return wrapNode(Node_random_normal(shape.ptr, mean, sd, dev.ptr[0], g.ptr[0]))

        @staticmethod
        def normal(_Shape shape, float mean, float sd, _Device dev):
            return wrapNode(Node_random_normal(shape.ptr, mean, sd, dev.ptr[0]))

        @staticmethod
        def log_normal(_Shape shape, float mean, float sd, _Device dev, _Graph g):
            return wrapNode(Node_random_log_normal(shape.ptr, mean, sd, dev.ptr[0], g.ptr[0]))

        @staticmethod
        def log_normal(_Shape shape, float mean, float sd, _Device dev):
            return wrapNode(Node_random_log_normal(shape.ptr, mean, sd, dev.ptr[0]))

        @staticmethod
        def gumbel(_Shape shape, float mu, float beta, _Device dev, _Graph g):
            return wrapNode(Node_random_gumbel(shape.ptr, mu, beta, dev.ptr[0], g.ptr[0]))

        @staticmethod
        def gumbel(_Shape shape, float mu, float beta, _Device dev):
            return wrapNode(Node_random_gumbel(shape.ptr, mu, beta, dev.ptr[0]))

    @staticmethod
    def dropout(_Node x, float rate, bool enabled):
        return wrapNode(Node_dropout(x.ptr, rate, enabled))
