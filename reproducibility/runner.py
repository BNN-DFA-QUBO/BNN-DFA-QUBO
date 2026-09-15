import argparse
import runpy
from pathlib import Path
from unittest.mock import patch

from reproducibility.deterministic import set_deterministic
from reproducibility.protocol_loader import (
    build_protocol_loaders,
    load_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
PROTOCOL_DIR = ROOT / "protocol"


def run_experiment(experiment_name):
    experiment_path = EXPERIMENTS_DIR / f"{experiment_name}.py"

    if not experiment_path.exists():
        available = sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.py"))
        raise FileNotFoundError(
            f"Experiment not found: {experiment_path}\n"
            f"Available experiments: {', '.join(available)}"
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

    # Import the original utility only here. The experiment itself remains
    # unchanged; its `from utils.data import get_mnist_loaders` will receive
    # the patched function when runpy executes it.
    import utils.data

    original_get_mnist_loaders = utils.data.get_mnist_loaders

    def protocol_get_mnist_loaders(batch_size=64):
        print(
            f"[REPRO] Replacing get_mnist_loaders(batch_size={batch_size}) "
            "with protocol-controlled loaders"
        )
        return build_protocol_loaders(
            original_get_mnist_loaders=original_get_mnist_loaders,
            protocol_dir=PROTOCOL_DIR,
            batch_size=batch_size,
        )

    print("[REPRO] Running original experiment without modifying it.")
    print()

    with patch(
        "utils.data.get_mnist_loaders",
        new=protocol_get_mnist_loaders,
    ):
        runpy.run_path(
            str(experiment_path),
            run_name="__main__",
        )

    print()
    print("[REPRO] =============================================")
    print("[REPRO] Experiment finished")
    print("[REPRO] =============================================")


def main():
    parser = argparse.ArgumentParser(
        description="Run an existing experiment with the fixed reproducibility protocol."
    )
    parser.add_argument(
        "experiment",
        help="Experiment filename without .py, e.g. bnn_dfa_qubo",
    )
    args = parser.parse_args()

    run_experiment(args.experiment)


if __name__ == "__main__":
    main()
