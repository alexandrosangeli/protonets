from mlp import MLP

def test_grads():
    xs = [5.1, 4.5, 8.8]
    h = 1e-5
    mlp = MLP(3, [2,1]) 
    for i in range(len(mlp.layers)):
        for j in range(len(mlp.layers[i].neurons)):
            for k in range(len(mlp.layers[i].neurons[j].w)):
                out_before = mlp(xs)
                out_before.backward()
                grad = mlp.layers[i].neurons[j].w[k].grad
                mlp.layers[i].neurons[j].w[k].data += h
                out_after = mlp(xs)

                a = out_after.data
                b = out_before.data

                approx_grad = (a - b)/h
                assert abs(approx_grad - grad) < 0.0001
                mlp.zero_grad()


if __name__ == "__main__":
    test_grads()