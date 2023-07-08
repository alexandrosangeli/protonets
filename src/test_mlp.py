import random

from mlp import MLP
from utils import mse_loss


def rand_mlp_and_input():
    xs_size = random.randint(1, 10)
    num_hidden_layers = random.randint(1, 5)

    xs = [random.uniform(0.0, 10.0) for _ in range(xs_size)]
    hidden_layers = [random.randint(2, 8) for _ in range(num_hidden_layers)] + [1]

    return MLP(xs_size, hidden_layers), xs


def test_grads():
    """
        Test that the gradients are calculated correctly
    """
    h = 1e-5
    mlp, xs = rand_mlp_and_input()
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
            flag = True  # Indicates at least one change in the output

        approx_grad = (a - b) / h
        grad_diff = approx_grad - grad

        assert (
            abs(grad_diff) < 1e-4
        ), "The gradient is not close to the approximated gradient"
    assert flag, "The output never changed"


def test_step():
    mlp, xs = rand_mlp_and_input()
    target = random.randint(1, 10)

    out_before = mlp(xs)
    loss_before = mse_loss(target, out_before)

    loss_before.backward()
    mlp.step(lr=1)

    out_after = mlp(xs)
    loss_after = mse_loss(target, out_after)

    assert loss_after.data < loss_before.data


if __name__ == "__main__":

    for i in range(20):
        test_grads()
        test_step()

    print("All tests passed!")
