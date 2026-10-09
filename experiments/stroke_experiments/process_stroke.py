"""Create stratified stroke splits, fit preprocessing, and balance training only."""
import argparse

from utils.stroke_data import DATA_DIR, prepare_stroke_data, load_stroke_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balance", action="store_true", help="Keep the original imbalanced training set")
    args = parser.parse_args()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_stroke_data(seed=args.seed, balance=not args.no_balance)
    metadata = load_stroke_metadata()
    print(f"Feature schema: {len(metadata['feature_names'])} columns")
    print(f"Train: {len(y_train)} rows, counts={metadata['resampled_train_class_counts']}")
    print(f"Validation: {len(y_val)} rows, counts={metadata['validation_class_counts']}")
    print(f"Test: {len(y_test)} rows, counts={metadata['test_class_counts']}")
    print(f"Saved preprocessor and metadata under {DATA_DIR}/ ({X_train.shape[1]} features)")


if __name__ == "__main__":
    main()
