import numpy as np


class Value:

    def __init__(self, data, children=[], label=''):
        self.data = data
        self.grad = 0.0
        self.label = label
        self.prev = children
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], f"({self.label})+{other.label}")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data, [self, other], f"({self.label})+{other.label}")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], f"({self.label})*{other.label}")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other.data, [self, other], f"({self.label})^{other.label}")
        
        def _backward():
            self.grad += other.data * self.data * out.grad
            sign = 1 if self.data > 0 else -1
            other.grad += self.data ** other.data * sign * np.log(abs(self.data)) * out.grad
            
        out._backward = _backward
        return out
    
    def sig(self):
        y = np.exp(self.data) / (np.exp(self.data) + 1)
        out = Value(y, [self], f"(sig({self.label})")
        
        def _backward():
            self.grad += y * (1 - y)
            
        out._backward = _backward
        return out
    

    def backward(self):
        topo = []
        visited = set()
        def topo_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    topo_sort(child)
                topo.append(node)

        self.grad = 1
        topo_sort(self)
        nodes = reversed(topo)
        for node in nodes:
            node._backward()


class Neuron:

    def __init__(self, num_input):
        self.w = [Value(w) for w  in np.random.randn(num_input)]
        self.b = Value(np.random.randn(1)[0])

    def __call__(self, xs):
        out = np.dot(self.w, xs) + self.b
        return out.sig()
    

class Layer:

    def __init__(self, num_input, num_neurons):
        # number of neurons is equivalent to number of outputs
        self.num_input = num_input
        self.num_neurons = num_neurons
        self.neurons = [Neuron(num_input) for _ in range(num_neurons)]

    def __call__(self, xs):
        outputs = [n(xs) for n in self.neurons]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def __repr__(self):
        return  f"Layer({self.num_input=}, {self.num_neurons=})"
    

class MLP:

    def __init__(self, num_inputs, outs):
        # number of outputs is for each layers
        self.layers = [Layer(num_inputs, outs[0])] + [Layer(outs[i], outs[i+1]) for i in range(len(outs) - 1)]

    def __call__(self, xs):
        outs = xs
        for layer in self.layers:
            outs = layer(outs)

        return outs