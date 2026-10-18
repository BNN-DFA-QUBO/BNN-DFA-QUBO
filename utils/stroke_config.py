import os

SEEDS = [42, 123, 2024, 7, 99]

SPLIT_SEED = int(os.environ.get("STROKE_SPLIT_SEED", "42"))
BALANCING_METHOD = os.environ.get("STROKE_BALANCING_METHOD", "random_oversample")
PREPROCESSING_VERSION = "finalboss-stroke-v3-scaled-inputs"

HIDDEN_SIZE = 64

BATCH_SIZE = 64

TRAIN_EPOCHS = 50
LEARNING_RATE = 0.001

STE_EPOCHS = 200
STE_LEARNING_RATE = 0.01
# Preserve the best validation checkpoint rather than blindly using the final STE sign pattern.
STE_PATIENCE = 25
STE_MIN_DELTA = 1e-4

QUBO_NUM_READS = 100
QUBO_NUM_SWEEPS = 80
# Training-only binary seeds get QUBO-energy coordinate refinement at every alpha scale.
QUBO_RANDOM_SEED_STARTS = 4
# Explore binary-head coupling strengths; validation selects the QUBO candidate.
QUBO_ALPHA_MULTIPLIERS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
QUBO_TOP_CANDIDATES_PER_SCALE = 10

DEVICE_PRIORITY = ["mps", "cuda", "cpu"]
