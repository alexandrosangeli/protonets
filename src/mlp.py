from backbone import Value
from utils import rand_normal, dot


class Neuron:

    def __init__(self, num_input):
        self.w = [Value(w) for w  in rand_normal(num_input)]
        self.b = Value(rand_normal(1)[0])

    def __call__(self, xs):
        out = dot(self.w, xs) + self.b
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
        # number of outputs is for each layer
        self.layers = [Layer(num_inputs, outs[0])] + [Layer(outs[i], outs[i+1]) for i in range(len(outs) - 1)]

    def __call__(self, xs):
        outs = xs
        for layer in self.layers:
            outs = layer(outs)

        return outs