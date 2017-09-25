from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from primitiv.graph cimport Graph, _Graph, wrapGraph, Node, _Node, wrapNode
from primitiv.device cimport wrapDevice
from primitiv.shape cimport wrapShape
from primitiv.tensor cimport wrapTensor
from primitiv.function cimport _Function


cdef class _Node:

    def __init__(self, _Node node = None):
        if node == None:
            self.ptr = Node(node.ptr)
        else:
            self.ptr = Node()

    def valid(self):
        return self.ptr.valid()

    def graph(self):
        return wrapGraph(&self.ptr.graph())

    def function_id(self):
        return self.ptr.function_id()

    def value_id(self):
        return self.ptr.value_id()

    def shape(self):
        return wrapShape(self.ptr.shape())

    def device(self):
        return wrapDevice(&self.ptr.device())

    def to_vector(self):
        return self.ptr.to_vector()

    def __iter__(self):
        return iter(self.ptr.to_vector())


cdef class _Graph:

    def __init__(self):
        self.ptr = new Graph()

    def get_default_graph(self):
        return wrapGraph(&self.ptr.get_default_graph())

    def set_default_graph(self, _Graph g):
        self.ptr.set_default_graph(g.ptr[0])
        return

    def add_function(self, _Function func, args):
        cdef vector[Node] vec
        cdef _Node x
        for x in args:
            vec.push_back(x.ptr)
            return wrapNode(self.ptr.add_function(unique_ptr[Function](func.ptr), vec))

    def forward(self, _Node node):
        return wrapTensor(self.ptr.forward(node.ptr))

    def backward(self, _Node node):
        self.ptr.backward(node.ptr)
        return

    def get_shape(self, _Node node):
        return wrapShape(self.ptr.get_shape(node.ptr))

    def get_device(self, _Node node):
        return wrapDevice(&self.ptr.get_device(node.ptr))

    def dump(self):
        self.ptr.dump()
        return

    def num_functions(self):
        return self.ptr.num_functions()
