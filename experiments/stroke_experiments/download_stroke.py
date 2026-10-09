"""Download the public Kaggle stroke dataset into the local ignored data directory."""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import kagglehub

from utils.stroke_data import REQUIRED_COLUMNS, DATA_DIR, RAW_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Local destination (default: data/stroke)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local CSV after downloading")
    args = parser.parse_args()
    destination = args.data_dir / RAW_PATH.name
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} already exists; keeping the user-provided file. Pass --overwrite to replace it.")
    cached = Path(kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset"))
    source = cached / "healthcare-dataset-stroke-data.csv"
    if not source.is_file():
        raise FileNotFoundError(f"KaggleHub download did not contain expected CSV: {source}")
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Downloaded stroke CSV is missing required columns: {missing}")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Saved {len(frame):,} rows to {destination.resolve()}")


if __name__ == "__main__":
    main()
