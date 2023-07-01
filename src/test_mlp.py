import random

from mlp import MLP


def test_grads():

    h = 1e-5

    xs_size = random.randint(1, 10)
    num_hidden_layers = random.randint(1, 5)

    xs = [random.uniform(0.0, 10.0) for _ in range(xs_size)]
    hidden_layers = [random.randint(2, 8) for _ in range(num_hidden_layers)] + [1]

    mlp = MLP(xs_size, hidden_layers)
    params = mlp.parameters()

    flag = False
    for p in params:
        out_before = mlp(xs)
        out_before.backward()
        grad = p.grad
        p.data += h
        out_after = mlp(xs)
        mlp.zero_grad()

        a = out_after.data
        b = out_before.data

        if a - b != 0:
            flag = True  # Indicated at least one change in the output

        approx_grad = (a - b) / h
        grad_diff = approx_grad - grad

        assert (
            abs(grad_diff) < 1e-4
        ), "The gradient is not close to the approximated gradient"
    assert flag, "The output never changed"


if __name__ == "__main__":
    for i in range(20):
        test_grads()
    print("All tests passed!")
