import random
import math

def monte_carlo_pi(n_samples):
    inside_circle = 0

    for _ in range(n_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        distance = math.sqrt(x**2 + y**2)

        if distance <= 1:
            inside_circle += 1

    pi_estimate = 4 * inside_circle / n_samples
    return pi_estimate

if __name__ == "__main__":
    n_samples = 1000000
    pi_estimate = monte_carlo_pi(n_samples)
    print(f"円周率の近似値（モンテカルロ法, サンプル数: {n_samples}): {pi_estimate}")
