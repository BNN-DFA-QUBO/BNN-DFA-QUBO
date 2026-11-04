import itertools
import torch
import neal

from experiments.stroke_experiments.stroke_dfa_qubo import (
    build_ising_model,
)

torch.manual_seed(42)

n_samples = 20
n_weights = 8
alpha = 0.7

H = torch.randn(n_samples, n_weights)
target = torch.randn(n_samples)

h, J = build_ising_model(H, target, alpha)

H_centered = H - H.mean(dim=0, keepdim=True)
target_centered = target - target.mean()

def squared_error(bits):
    weights = torch.tensor(bits, dtype=torch.float32)
    return torch.sum(
        (alpha * (H_centered @ weights) - target_centered) ** 2
    ).item()

all_bits = list(itertools.product([-1.0, 1.0], repeat=n_weights))
true_best = min(all_bits, key=squared_error)
true_best_error = squared_error(true_best)

sampler = neal.SimulatedAnnealingSampler()

num_trials = 30
successes = 0
best_found_error = float("inf")

for seed in range(num_trials):
    response = sampler.sample_ising(
        h,
        J,
        num_reads=100,
        seed=seed,
    )

    sample = response.first.sample
    bits = tuple(float(sample[i]) for i in range(n_weights))
    error = squared_error(bits)

    best_found_error = min(best_found_error, error)

    if bits == true_best:
        successes += 1

print(f"Sampler: dwave-neal {__import__('importlib.metadata', fromlist=['version']).version('dwave-neal')}")
print(f"Exhaustive global minimum: {true_best_error:.8f}")
print(f"Sampler trials: {num_trials}")
print(f"Reads per trial: 100")
print(f"Trials reaching exact optimum: {successes}/{num_trials}")
print(f"Exact optimum hit rate: {100 * successes / num_trials:.1f}%")
print(f"Best squared error found: {best_found_error:.8f}")
print(f"Global optimum reached at least once: {abs(best_found_error - true_best_error) < 1e-4}")
