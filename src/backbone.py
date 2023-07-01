import math


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
            other.grad += self.data ** other.data * sign * math.log(abs(self.data)) * out.grad
            
        out._backward = _backward
        return out
    
    def sig(self):
        y = math.exp(self.data) / (math.exp(self.data) + 1)
        out = Value(y, [self], f"(sig({self.label})")
        
        def _backward():
            self.grad += y * (1 - y)
            
        out._backward = _backward
        return out
    

    def __sub__(self, other):
            return self + (- other)


    def __radd__(self, other): # other + self
        return self + other
    

    def __rsub__(self, other): # other + self
        return self + (- other)
    

    def __rmul__(self, other):
        return self * other


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