import os

SEEDS = [42, 123, 2024, 7, 99]

SPLIT_SEED = int(os.environ.get("STROKE_SPLIT_SEED", "42"))

# Active model input size comes from the fitted preprocessing schema.
INPUT_SIZE = None
HIDDEN_SIZE = 64

BATCH_SIZE = 64

TRAIN_EPOCHS = 50
LEARNING_RATE = 0.001

STE_EPOCHS = 200
STE_LEARNING_RATE = 0.01

QUBO_NUM_READS = 100

DEVICE_PRIORITY = ["mps", "cuda", "cpu"]
