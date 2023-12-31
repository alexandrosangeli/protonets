from backbone import Value
from utils import rand_normal, dot, check_bottleneck


class Neuron:
    def __init__(self, num_input):
        self.w = [Value(w) for w in rand_normal(num_input)]
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
        return f"Layer({self.num_input=}, {self.num_neurons=})"


class MLP:
    def __init__(self, num_inputs, outs):
        # number of outputs is for each layer

        if check_bottleneck(outs):
            raise ValueError(
                "Hidden layers with one output is not supported (and not recommended)"
            )

        self.layers = [Layer(num_inputs, outs[0])] + [
            Layer(outs[i], outs[i + 1]) for i in range(len(outs) - 1)
        ]

    def __call__(self, xs):
        outs = xs
        for layer in self.layers:
            outs = layer(outs)

        return outs

    def parameters(self):
        params = []
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.w:
                    params.append(weight)
        return params

    def zero_grad(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.w:
                    weight.grad = 0.0

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def step(self, lr, n=1):
        for _ in range(n):
            for p in self.parameters():
                p.data -= lr * p.grad
