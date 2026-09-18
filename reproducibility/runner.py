import argparse
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from reproducibility.deterministic import set_deterministic
from reproducibility.protocol_loader import (
    build_protocol_loaders,
    load_manifest,
)


ROOT = Path(__file__).resolve().parent.parent

MNIST_EXPERIMENTS_DIR = ROOT / "experiments" / "MNIST_experiments"
PROTOCOL_DIR = ROOT / "protocol"


def run_experiment(experiment_name):
    experiment_path = MNIST_EXPERIMENTS_DIR / f"{experiment_name}.py"

    if not experiment_path.exists():
        available = sorted(
            p.stem
            for p in MNIST_EXPERIMENTS_DIR.glob("*.py")
            if p.name != "__init__.py"
        )

        raise FileNotFoundError(
            f"Experiment not found: {experiment_path}\n"
            f"Available MNIST experiments: {', '.join(available)}"
        )

    manifest = load_manifest(PROTOCOL_DIR)

    seed = int(manifest["seed"])
    set_deterministic(seed)

    print("[REPRO] =============================================")
    print("[REPRO] Cross-platform reproducibility runner")
    print("[REPRO] =============================================")
    print(f"[REPRO] Experiment : {experiment_name}")
    print(f"[REPRO] Seed       : {seed}")
    print(f"[REPRO] Protocol   : {PROTOCOL_DIR}")
    print(f"[REPRO] Epochs     : {manifest['num_epochs']}")
    print()

    # Import the MNIST loader used by the MNIST experiments.
    import utils.MNIST.data

    original_get_loaders = utils.MNIST.data.get_mnist_loaders

    def protocol_get_loaders(batch_size=64):
        print(
            f"[REPRO] Replacing get_mnist_loaders(batch_size={batch_size}) "
            "with protocol-controlled loaders"
        )

        return build_protocol_loaders(
            original_get_loaders=original_get_loaders,
            protocol_dir=PROTOCOL_DIR,
            batch_size=batch_size,
        )

    print("[REPRO] Running original experiment without modifying it.")
    print()

    # Give the experiment the protocol's seed through --seed.
    original_argv = sys.argv

    try:
        sys.argv = [
            str(experiment_path),
            "--seed",
            str(seed),
        ]

        with patch(
            "utils.MNIST.data.get_mnist_loaders",
            new=protocol_get_loaders,
        ):
            runpy.run_path(
                str(experiment_path),
                run_name="__main__",
            )

    finally:
        sys.argv = original_argv

    print()
    print("[REPRO] =============================================")
    print("[REPRO] Experiment finished")
    print("[REPRO] =============================================")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run an MNIST experiment with the fixed "
            "reproducibility protocol."
        )
    )

    parser.add_argument(
        "experiment",
        help="MNIST experiment filename without .py, e.g. bnn_dfa_qubo",
    )

    args = parser.parse_args()

    run_experiment(args.experiment)


if __name__ == "__main__":
    main()