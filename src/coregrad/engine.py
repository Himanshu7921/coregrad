"""
Core scalar-based automatic differentiation engine.

This module implements a minimal reverse-mode automatic differentiation
system for scalar-valued computations. The engine constructs a dynamic
computational graph during the forward pass and performs gradient
propagation through reverse traversal of the graph.

Each `Scalar` instance represents a node in the computational graph and
stores:

    - scalar numerical value (`data`)
    - accumulated gradient (`grad`)
    - parent dependencies (`_prev`)
    - generating operation metadata (`_op`)
    - local backward propagation function (`_backward`)

The backward pass is executed using topological ordering of the
computational graph, ensuring that gradient contributions are propagated
only after all dependent nodes have been processed.

Implemented differentiable operations:
    - Addition
    - Subtraction
    - Multiplication
    - Division
    - Power
    - Softmax()
    - Hyperbolic Tangent (tanh)

Features:
    - Dynamic graph construction
    - Reverse-mode automatic differentiation
    - Gradient accumulation
    - Operator overloading for mathematical expressions
    - Topologically ordered backpropagation

Notes:
    - This implementation operates on scalar values only.
    - Gradients are accumulated in-place.
    - The engine is designed for educational, experimental,
      and foundational deep learning system research purposes.

For implementation walkthroughs and derivations, refer to:
    notebook/manual_back_prop.ipynb
"""

import numpy as np

class Scalar:
    def __init__(self, data, _prev = (), _op = "", var_name = ""):
        self.data = data
        self.grad = 0
        self._prev = tuple(_prev)
        self._op = _op
        self.var_name = var_name
        self._backward = lambda: None

    def __repr__(self):
        return (
        f"Scalar(data={self.data}, "
        f"grad={self.grad}"
    )

    @staticmethod
    def ensure_scalar(x):
        return x if isinstance(x, Scalar) else Scalar(x)
    
    def __add__(self, other):
        other = Scalar.ensure_scalar(other)
        out = Scalar(self.data + other.data, _prev = (other, self), _op = "+")
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = Scalar.ensure_scalar(other)
        # self = self if isinstance(self, Scalar) else Scalar(self)
        out = Scalar(self.data - other.data, _prev = (other, self), _op = "-")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = Scalar.ensure_scalar(other)
        if other.data == 0:
            raise ZeroDivisionError("Division by zero")
        
        out = Scalar(self.data / other.data, _prev = (other, self), _op = "/")

        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = Scalar.ensure_scalar(other)
        out = Scalar(self.data * other.data, _prev = (other, self), _op = "*")
        # Store the Local gradients
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = Scalar.ensure_scalar(other)
        out = Scalar(self.data ** other.data, _prev = (other, self), _op = f"**{other}")
        # c = a ** b ---> dc/da = b * a ** (b - 1); dc/db = (a ** b) * (ln(a))
        def _backward():
            self.grad += (other.data * (self.data ** (other.data - 1.0))) * out.grad
            # other.grad += ((self.data ** other.data) * np.log(self.data)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        tanh_x = ((np.exp(2*x) - 1) / (np.exp(2*x) + 1))
        out = Scalar(tanh_x, var_name = "tanh", _prev = (self, ), _op = "tanh")
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = np.exp(self.data)
        out = Scalar(x, var_name="exp", _prev=(self,), _op="exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # To add this functionality we need to implement a topological sorting
        topo_sorted = []
        visited = set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child)
                topo_sorted.append(v) # add the parent only after adding the child
        topo_sort(self)

        # Doing backprop on whole computatinal graph
        self.grad = 1.0
        for node in topo_sorted[::-1]:
            node._backward() # the Topological sorting will ensure that we are not calling backward before computing all the dependencies after the calling node
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other - self
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other / self
    
    def __rpow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other ** self
    