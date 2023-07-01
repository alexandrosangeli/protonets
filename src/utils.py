import random
import math


def rand_normal(n):
    """
        Return n samples from the standard normal distribution.

        - Uses the Box-Muller transform method: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        - This is a fairly efficient method. See rand_normal_slow for a (slightly) simpler implementation
        - Takes approx. half the time to execute when drawing 10000 samples (from an avg. of 1000 executions)
    """
    if n < 1:
        return None

    rands = []
    i = 0
    for i in range(n // 2 + 1):
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)

        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

        rands.append(z0)
        rands.append(z1)

    return rands[:n]


def rand_normal_slow(n):
    if n < 1:
        return None

    rands = []
    for i in range(n):
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)

        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

        rands.append(z0)

    return rands


def dot(arr0, arr1):
    # Calculates the dot product between two arrays
    return sum(x * y for x, y in zip(arr0, arr1))


def check_bottleneck(outs):
    for i, n in enumerate(outs):
        if n == 1 and i != len(outs) - 1:
            return True
    return False
