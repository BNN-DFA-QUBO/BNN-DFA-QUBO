"""Create stratified stroke splits, fit preprocessing, and balance training only."""
import argparse
from pathlib import Path

from utils.stroke_data import DATA_DIR, prepare_stroke_data, load_stroke_metadata
from utils.stroke_config import SPLIT_SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR,
                        help="Directory for preprocessing artifacts (default: configured stroke data directory)")
    parser.add_argument("--balancing", choices=("random_oversample", "none"), default=None,
                        help="Training-only balancing method (defaults to STROKE_BALANCING_METHOD)")
    parser.add_argument("--overwrite", action="store_true", help="Replace stale preprocessing artifacts after validating the source CSV")
    args = parser.parse_args()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_stroke_data(
        split_seed=args.seed, balance_method=args.balancing, overwrite=args.overwrite,
        output_dir=args.output_dir
    )
    metadata = load_stroke_metadata(args.output_dir / "metadata.json")
    print(f"Feature schema: {len(metadata['feature_names'])} columns")
    print(f"Train: {len(y_train)} rows, counts={metadata['resampled_train_class_counts']}")
    print(f"Validation: {len(y_val)} rows, counts={metadata['validation_class_counts']}")
    print(f"Test: {len(y_test)} rows, counts={metadata['test_class_counts']}")
    print(f"Balancing: {metadata['balancing_method']}")
    print(f"Saved preprocessor and metadata under {args.output_dir}/ ({X_train.shape[1]} features)")


if __name__ == "__main__":
    main()
