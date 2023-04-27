import math
import numpy as np

global exp
exp = math.e

class Value():
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        
    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self,other):
        # other = other if isinstance(other, Value) else Value(other)
        # out = Value(self.data + (-other.data),(self,other),'-')
        # return out
        return self + (-other)
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __pow__(self,other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
       # other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        #e^2x -1/e^2x + 1
        top = exp**(2*self.data) - 1
        bot = exp**(2*self.data) + 1
        tanh = top/bot
        out = Value(tanh,(self,),'tanh')
        
        def _backward():
            self.grad += (1-tanh**2) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
            
        def _backward():
            self.grad += (out.grad > 0) * out.grad
            
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()