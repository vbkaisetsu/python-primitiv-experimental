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

    def __pos__(self):
        return wrapNode(op_node_pos(self.ptr))

    def __neg__(self):
        return wrapNode(op_node_neg(self.ptr))

    def __add__(_Node self, v):
        if isinstance(v, (int, float)):
            return wrapNode(op_node_add(self.ptr, <float> v))
        elif isinstance(v, _Node):
            return wrapNode(op_node_add(self.ptr, (<_Node> v).ptr))
        else:
            return NotImplemented

    def __radd__(self, v):
        if isinstance(v, _Node):
            return wrapNode(op_node_add(<float> v, self.ptr))
        else:
            return NotImplemented

    def __sub__(_Node self, v):
        if isinstance(v, (int, float)):
            return wrapNode(op_node_sub(self.ptr, <float> v))
        elif isinstance(v, _Node):
            return wrapNode(op_node_sub(self.ptr, (<_Node> v).ptr))
        else:
            return NotImplemented

    def __rsub__(self, v):
        if isinstance(v, _Node):
            return wrapNode(op_node_sub(<float> v, self.ptr))
        else:
            return NotImplemented

    def __mul__(_Node self, v):
        if isinstance(v, (int, float)):
            return wrapNode(op_node_mul(self.ptr, <float> v))
        elif isinstance(v, _Node):
            return wrapNode(op_node_mul(self.ptr, (<_Node> v).ptr))
        else:
            return NotImplemented

    def __rmul__(self, v):
        if isinstance(v, _Node):
            return wrapNode(op_node_mul(<float> v, self.ptr))
        else:
            return NotImplemented

    def __div__(_Node self, v):
        if isinstance(v, (int, float)):
            return wrapNode(op_node_div(self.ptr, <float> v))
        elif isinstance(v, _Node):
            return wrapNode(op_node_div(self.ptr, (<_Node> v).ptr))
        else:
            return NotImplemented

    def __rdiv__(self, v):
        if isinstance(v, _Node):
            return wrapNode(op_node_div(<float> v, self.ptr))
        else:
            return NotImplemented


cdef class _Graph:

    def __init__(self):
        self.ptr = new Graph()

    @staticmethod
    def get_default_graph():
        return wrapGraph(&get_default_graph())

    @staticmethod
    def set_default_graph(_Graph g):
        set_default_graph(g.ptr[0])
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
