from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp cimport bool
from primitiv.device cimport Device
from primitiv.shape cimport Shape
from primitiv.tensor cimport Tensor
from primitiv.function cimport Function


cdef extern from "primitiv/graph.h" namespace "primitiv":
    cdef cppclass Node:
        Node(Node &&src)
        Node()
        bool valid()
        Graph &graph()
        unsigned function_id()
        unsigned value_id()
        const Shape &shape()
        Device &device()
        vector[float] to_vector()


cdef extern from "primitiv/graph.h" namespace "primitiv":
    cdef cppclass Graph:
        Graph()
        Graph &get_default_graph()
        void set_default_graph(Graph &g)
        Node add_function(unique_ptr[Function] &&func, vector[Node] &args)
        const Tensor &forward(const Node &node)
        void backward(const Node &node)
        const Shape &get_shape(const Node &node)
        Device &get_device(const Node &node)
        void dump()
        unsigned num_functions()


cdef class _Node:
    cdef Node ptr


cdef class _Graph:
    cdef Graph *ptr


cdef inline _Node wrapNode(Node ptr):
    cdef _Node node = _Node.__new__(_Node)
    node.ptr = ptr
    return node


cdef inline _Graph wrapGraph(Graph *ptr):
    cdef _Graph graph = _Graph.__new__(_Graph)
    graph.ptr = ptr
    return graph
