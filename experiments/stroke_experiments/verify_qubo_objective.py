import itertools
import torch

from experiments.stroke_experiments.stroke_dfa_qubo import (
    build_ising_model,
)

torch.manual_seed(42)

# Small synthetic problem: 20 examples, 8 binary weights.
n_samples = 20
n_weights = 8
alpha = 0.7

H = torch.randn(n_samples, n_weights)
target = torch.randn(n_samples)

h, J = build_ising_model(H, target, alpha)

H_centered = H - H.mean(dim=0, keepdim=True)
target_centered = target - target.mean()

results = []

# Exhaustively evaluate all 2^8 = 256 configurations.
for bits in itertools.product([-1.0, 1.0], repeat=n_weights):
    weights = torch.tensor(bits, dtype=torch.float32)

    squared_error = torch.sum(
        (alpha * (H_centered @ weights) - target_centered) ** 2
    ).item()

    ising_energy = sum(
        h[i] * bits[i]
        for i in range(n_weights)
    )

    ising_energy += sum(
        coefficient * bits[i] * bits[j]
        for (i, j), coefficient in J.items()
    )

    results.append((squared_error, ising_energy, bits))

# The original objective and Ising energy should differ only by a constant.
errors = [
    squared_error - ising_energy
    for squared_error, ising_energy, _ in results
]

max_difference_from_constant = max(
    abs(value - errors[0]) for value in errors
)

best_squared_error = min(results, key=lambda row: row[0])
best_ising_energy = min(results, key=lambda row: row[1])

print(f"Configurations checked: {len(results)}")
print(
    "Maximum variation in objective-minus-energy: "
    f"{max_difference_from_constant:.8f}"
)
print(f"Best squared error: {best_squared_error[0]:.8f}")
print(f"Best Ising energy: {best_ising_energy[1]:.8f}")
print(
    "Same optimal configuration: "
    f"{best_squared_error[2] == best_ising_energy[2]}"
)

tolerance = 1e-4

if max_difference_from_constant > tolerance:
    raise AssertionError(
        "FAIL: Ising energy does not match the squared-error "
        "objective up to an additive constant."
    )

if best_squared_error[2] != best_ising_energy[2]:
    raise AssertionError(
        "FAIL: The two objectives do not select the same optimum."
    )

print("PASS: Ising objective matches exhaustive squared-error search.")
