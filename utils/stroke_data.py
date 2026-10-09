"""Reproducible local stroke dataset, split, preprocessing and balancing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from utils.stroke_config import SPLIT_SEED

DATA_DIR = Path(os.environ.get("STROKE_DATA_DIR", "data/stroke"))
RAW_PATH = Path(os.environ.get("STROKE_CSV", DATA_DIR / "healthcare-dataset-stroke-data.csv"))
PREPROCESSOR_PATH = DATA_DIR / "preprocessor.joblib"
METADATA_PATH = DATA_DIR / "metadata.json"
REQUIRED_COLUMNS = {
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke",
}
NUMERIC = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "Residence_type", "work_type_grouped", "smoking_status"]


def _validate(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Stroke CSV is missing required columns: {missing}")
    if df.empty or df["stroke"].isna().any() or not set(df["stroke"].unique()) <= {0, 1}:
        raise ValueError("Stroke CSV must have rows and a non-missing binary 'stroke' target")
    if df["id"].duplicated().any():
        raise ValueError("Stroke CSV contains duplicate row identifiers")


class StrokePreprocessor:
    """Fit imputers, feature engineering, power transforms and encoder on train only."""
    def fit(self, frame: pd.DataFrame):
        x = self._engineer(frame)
        self.numeric_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self.numeric_imputer.fit(x[NUMERIC])
        imputed = pd.DataFrame(self.numeric_imputer.transform(x[NUMERIC]), columns=NUMERIC, index=x.index)
        self.category_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        self.category_imputer.fit(x[CATEGORICAL])
        cats = pd.DataFrame(self.category_imputer.transform(x[CATEGORICAL]), columns=CATEGORICAL, index=x.index)
        self.power = PowerTransformer(method="yeo-johnson", standardize=True)
        self.power.fit(self._power_inputs(imputed))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.encoder.fit(cats)
        self.feature_names = self._feature_names()
        return self

    @staticmethod
    def _engineer(frame):
        x = frame.drop(columns=["id", "stroke"], errors="ignore").copy()
        x["work_type_grouped"] = x["work_type"].replace({"children": "Non-Working/Dependent", "Never_worked": "Non-Working/Dependent"})
        x = x.drop(columns="work_type")
        return x

    @staticmethod
    def _power_inputs(imputed):
        return np.column_stack((imputed["avg_glucose_level"], imputed["bmi"], imputed["age"] * imputed["avg_glucose_level"]))

    def _feature_names(self):
        cat_names = self.encoder.get_feature_names_out(CATEGORICAL).tolist()
        return ["age", "hypertension", "heart_disease", "cardio_comorbidity_index", "glucose_risk_tier", "metabolic_syndrome_flag",
                "avg_glucose_level_normalized", "bmi_normalized", "age_glucose_interaction_normalized"] + cat_names

    def transform(self, frame):
        x = self._engineer(frame)
        numeric = pd.DataFrame(self.numeric_imputer.transform(x[NUMERIC]), columns=NUMERIC, index=x.index)
        cats = pd.DataFrame(self.category_imputer.transform(x[CATEGORICAL]), columns=CATEGORICAL, index=x.index)
        a, b, interaction = self.power.transform(self._power_inputs(numeric)).T
        features = pd.DataFrame({
            "age": numeric.age, "hypertension": numeric.hypertension.round().clip(0, 1),
            "heart_disease": numeric.heart_disease.round().clip(0, 1),
            "cardio_comorbidity_index": numeric.hypertension.round().clip(0, 1) + numeric.heart_disease.round().clip(0, 1),
            "glucose_risk_tier": pd.cut(numeric.avg_glucose_level, [-np.inf, 100, 140, np.inf], labels=[0, 1, 2]).astype(int),
            "metabolic_syndrome_flag": ((numeric.hypertension >= .5) & (numeric.avg_glucose_level >= 140) & (numeric.bmi >= 30)).astype(int),
            "avg_glucose_level_normalized": a, "bmi_normalized": b, "age_glucose_interaction_normalized": interaction,
        }, index=x.index)
        encoded = self.encoder.transform(cats)
        result = np.column_stack((features.to_numpy(dtype=np.float32), encoded)).astype(np.float32)
        if result.shape[1] != len(self.feature_names) or not np.isfinite(result).all():
            raise ValueError("Transformed stroke features have invalid schema or non-finite values")
        return pd.DataFrame(result, index=frame.index, columns=self.feature_names)


def _split(df, seed):
    indices = np.arange(len(df))
    train, remaining = train_test_split(indices, test_size=.30, stratify=df.stroke, random_state=seed)
    val, test = train_test_split(remaining, test_size=.50, stratify=df.stroke.iloc[remaining], random_state=seed)
    return train, val, test


def prepare_stroke_data(seed: int | None = None, balance: bool = True, csv_path=None, output_dir=None):
    """Return X/y in the legacy six-part format; balance means train duplication only."""
    source_path = Path(csv_path) if csv_path is not None else RAW_PATH
    artifact_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    if not source_path.is_file():
        raise FileNotFoundError(f"Stroke CSV not found at {source_path}. Run: python -m experiments.stroke_experiments.download_stroke")
    raw_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if seed is None:
        # Reuse only the seed associated with this exact source file. Indices are
        # always rebuilt below, so stale index arrays can never be applied.
        previous_metadata = artifact_dir / METADATA_PATH.name
        try:
            prior = json.loads(previous_metadata.read_text(encoding="utf-8"))
            seed = int(prior["seed"]) if prior.get("dataset_sha256") == raw_hash else SPLIT_SEED
        except (OSError, ValueError, KeyError, TypeError):
            seed = SPLIT_SEED
    else:
        seed = int(seed)
    df = pd.read_csv(source_path, na_values=["N/A", "NA", "Unknown_na", ""])
    _validate(df)
    # Match the reference pipeline's exclusion of the one 'Other' gender row, while preserving source ids.
    df = df.loc[df.gender.ne("Other")].reset_index(drop=True)
    train_idx, val_idx, test_idx = _split(df, seed)
    preprocessor = StrokePreprocessor().fit(df.iloc[train_idx])
    X_train = preprocessor.transform(df.iloc[train_idx])
    X_val = preprocessor.transform(df.iloc[val_idx])
    X_test = preprocessor.transform(df.iloc[test_idx])
    y_train = df.stroke.iloc[train_idx].astype(int).reset_index(drop=True)
    y_val = df.stroke.iloc[val_idx].astype(int).reset_index(drop=True)
    y_test = df.stroke.iloc[test_idx].astype(int).reset_index(drop=True)
    original_counts = y_train.value_counts().sort_index().to_dict()
    train_row_ids = df.id.iloc[train_idx].tolist()
    resampled_train_row_ids = list(train_row_ids)
    if balance and len(original_counts) == 2:
        # Random oversampling duplicates actual minority examples. This avoids SMOTE's invalid
        # fractional binary flags and multi-hot category groups after the ZIP's one-hot encoding.
        rng = np.random.default_rng(seed)
        majority = int(y_train.value_counts().max())
        minority_rows = np.flatnonzero(y_train.to_numpy() == y_train.value_counts().idxmin())
        extra = rng.choice(minority_rows, majority - len(minority_rows), replace=True)
        rows = np.concatenate((np.arange(len(y_train)), extra))
        rng.shuffle(rows)
        resampled_train_row_ids = [train_row_ids[int(i)] for i in rows]
        X_train = X_train.iloc[rows].reset_index(drop=True)
        y_train = y_train.iloc[rows].reset_index(drop=True)
    else:
        X_train = X_train.reset_index(drop=True)
    X_val, X_test = X_val.reset_index(drop=True), X_test.reset_index(drop=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / PREPROCESSOR_PATH.name)
    metadata = {
        "dataset_sha256": raw_hash, "seed": seed, "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_row_ids": {name: df.id.iloc[idx].tolist() for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))},
        "resampled_train_row_ids": resampled_train_row_ids,
        "feature_names": preprocessor.feature_names, "input_size": len(preprocessor.feature_names),
        "original_train_class_counts": {str(k): int(v) for k, v in original_counts.items()},
        "resampled_train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()},
        "validation_class_counts": {str(k): int(v) for k, v in y_val.value_counts().sort_index().to_dict().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()},
        "balancing_method": "random minority oversampling by exact row duplication" if balance else "none",
    }
    (artifact_dir / METADATA_PATH.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_stroke_preprocessor(path=PREPROCESSOR_PATH):
    return joblib.load(path)


def load_stroke_metadata(path=METADATA_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))
