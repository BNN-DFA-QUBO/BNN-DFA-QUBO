"""Reproducible local stroke dataset, split, preprocessing and balancing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from utils.stroke_config import BALANCING_METHOD, PREPROCESSING_VERSION, SPLIT_SEED

DATA_DIR = Path(os.environ.get("STROKE_DATA_DIR", "data/stroke"))
RAW_PATH = Path(os.environ.get("STROKE_CSV", DATA_DIR / "healthcare-dataset-stroke-data.csv"))
PREPROCESSOR_PATH = DATA_DIR / "preprocessor.joblib"
METADATA_PATH = DATA_DIR / "metadata.json"
REQUIRED_COLUMNS = {
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke",
}
NUMERIC = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
KNN_COLUMNS = ["age", "gender_encoded", "avg_glucose_level", "hypertension", "heart_disease", "bmi"]
CATEGORICAL = ["gender", "ever_married", "Residence_type", "work_type_grouped", "smoking_status"]
BINARY_CATEGORICAL_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "ever_married": {"No": 0, "Yes": 1},
    "Residence_type": {"Rural": 0, "Urban": 1},
}
ONE_HOT_CATEGORICAL = ["work_type_grouped", "smoking_status"]
SPLIT_POLICY = "stratified_train_test_split_70_15_15_v1"
PREPROCESSING_CONFIG = {
    "excluded_gender": "Other",
    "work_type_grouping": {"children": "Non-Working/Dependent", "Never_worked": "Non-Working/Dependent"},
    "numeric_imputer": {"method": "knn", "features": KNN_COLUMNS, "n_neighbors": 5, "weights": "distance"},
    "categorical_imputer": {"method": "most_frequent"},
    "power_transform": {"method": "yeo-johnson", "standardize": True,
                        "columns": ["avg_glucose_level", "bmi", "age_glucose_interaction"]},
    "encoder": {"method": "onehot", "columns": ONE_HOT_CATEGORICAL, "handle_unknown": "ignore"},
    "binary_categorical_maps": BINARY_CATEGORICAL_MAPS,
    "cardio_comorbidity_index": "rounded hypertension + rounded heart_disease",
    "glucose_risk_tier": {"cut_points": [100, 140], "labels": [0, 1, 2], "intervals": "right-closed, matching pandas.cut defaults"},
    "metabolic_syndrome_flag": {"hypertension": 1, "avg_glucose_level_min": 140, "bmi_min": 30},
    "feature_order_policy": "binary/clinical/raw-engineered, one-hot work/smoking, Yeo-Johnson values",
    "split_policy": SPLIT_POLICY,
}


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
        self.category_imputer = SimpleImputer(strategy="most_frequent")
        self.category_imputer.fit(x[CATEGORICAL])
        cats = pd.DataFrame(self.category_imputer.transform(x[CATEGORICAL]), columns=CATEGORICAL, index=x.index)
        self.binary_fill_values = {name: cats[name].mode().iloc[0] for name in BINARY_CATEGORICAL_MAPS}
        binary_cats = self._map_binary_categories(cats)
        knn_input = self._knn_inputs(x, binary_cats)
        self.numeric_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self.numeric_imputer.fit(knn_input)
        imputed = pd.DataFrame(self.numeric_imputer.transform(knn_input), columns=KNN_COLUMNS, index=x.index)
        self.power = PowerTransformer(method="yeo-johnson", standardize=True)
        self.power.fit(self._power_inputs(imputed))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.encoder.fit(cats[ONE_HOT_CATEGORICAL])
        self.feature_names = self._feature_names()
        return self

    @staticmethod
    def _engineer(frame):
        x = frame.drop(columns=["id", "stroke"], errors="ignore").copy()
        x["work_type_grouped"] = x["work_type"].replace({"children": "Non-Working/Dependent", "Never_worked": "Non-Working/Dependent"})
        x = x.drop(columns="work_type")
        return x

    def _map_binary_categories(self, cats):
        binary = pd.DataFrame(index=cats.index)
        for name, mapping in BINARY_CATEGORICAL_MAPS.items():
            mapped = cats[name].where(cats[name].isin(mapping), self.binary_fill_values[name]).map(mapping)
            binary[name] = mapped.astype(np.float64)
        return binary

    @staticmethod
    def _knn_inputs(x, binary_cats):
        return pd.DataFrame({
            "age": pd.to_numeric(x["age"], errors="coerce"),
            "gender_encoded": binary_cats["gender"],
            "avg_glucose_level": pd.to_numeric(x["avg_glucose_level"], errors="coerce"),
            "hypertension": pd.to_numeric(x["hypertension"], errors="coerce"),
            "heart_disease": pd.to_numeric(x["heart_disease"], errors="coerce"),
            "bmi": pd.to_numeric(x["bmi"], errors="coerce"),
        }, index=x.index)

    @staticmethod
    def _power_inputs(imputed):
        return np.column_stack((imputed["avg_glucose_level"], imputed["bmi"], imputed["age"] * imputed["avg_glucose_level"]))

    def _feature_names(self):
        binary_names = ["gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type",
                        "avg_glucose_level", "bmi", "cardio_comorbidity_index", "age_glucose_interaction",
                        "glucose_risk_tier", "metabolic_syndrome_flag"]
        cat_names = [name.replace("work_type_grouped_", "work_type_").replace("smoking_status_", "smoking_")
                     for name in self.encoder.get_feature_names_out(ONE_HOT_CATEGORICAL).tolist()]
        return binary_names + cat_names + ["avg_glucose_level_normalized", "bmi_normalized", "age_glucose_interaction_normalized"]

    def transform(self, frame):
        x = self._engineer(frame)
        cats = pd.DataFrame(self.category_imputer.transform(x[CATEGORICAL]), columns=CATEGORICAL, index=x.index)
        binary_cats = self._map_binary_categories(cats)
        numeric = pd.DataFrame(self.numeric_imputer.transform(self._knn_inputs(x, binary_cats)), columns=KNN_COLUMNS, index=x.index)
        numeric["hypertension"] = numeric.hypertension.round().clip(0, 1)
        numeric["heart_disease"] = numeric.heart_disease.round().clip(0, 1)
        a, b, interaction = self.power.transform(self._power_inputs(numeric)).T
        features = pd.DataFrame({
            "age": numeric.age, "hypertension": numeric.hypertension.round().clip(0, 1),
            "heart_disease": numeric.heart_disease.round().clip(0, 1),
            "cardio_comorbidity_index": numeric.hypertension.round().clip(0, 1) + numeric.heart_disease.round().clip(0, 1),
            "glucose_risk_tier": pd.cut(numeric.avg_glucose_level, [-np.inf, 100, 140, np.inf], labels=[0, 1, 2]).astype(int),
            "metabolic_syndrome_flag": ((numeric.hypertension >= .5) & (numeric.avg_glucose_level >= 140) & (numeric.bmi >= 30)).astype(int),
            "avg_glucose_level_normalized": a, "bmi_normalized": b, "age_glucose_interaction_normalized": interaction,
        }, index=x.index)
        binary = pd.DataFrame(index=frame.index)
        for name in BINARY_CATEGORICAL_MAPS:
            if name == "gender":
                binary[name] = numeric.gender_encoded.round().clip(0, 1).astype(np.float32).to_numpy()
            else:
                binary[name] = binary_cats[name].astype(np.float32).to_numpy()
        binary["age"] = numeric.age.to_numpy()
        binary["hypertension"] = numeric.hypertension.round().clip(0, 1).to_numpy()
        binary["heart_disease"] = numeric.heart_disease.round().clip(0, 1).to_numpy()
        binary["avg_glucose_level"] = numeric.avg_glucose_level.to_numpy()
        binary["bmi"] = numeric.bmi.to_numpy()
        binary["cardio_comorbidity_index"] = features.cardio_comorbidity_index.to_numpy()
        binary["age_glucose_interaction"] = (numeric.age * numeric.avg_glucose_level).to_numpy()
        binary["glucose_risk_tier"] = features.glucose_risk_tier.to_numpy()
        binary["metabolic_syndrome_flag"] = features.metabolic_syndrome_flag.to_numpy()
        binary = binary[self.feature_names[:12]]
        encoded = self.encoder.transform(cats[ONE_HOT_CATEGORICAL])
        normalized = np.column_stack((a, b, interaction)).astype(np.float32)
        result = np.column_stack((binary.to_numpy(dtype=np.float32), encoded, normalized)).astype(np.float32)
        if result.shape[1] != len(self.feature_names) or not np.isfinite(result).all():
            raise ValueError("Transformed stroke features have invalid schema or non-finite values")
        return pd.DataFrame(result, index=frame.index, columns=self.feature_names)


def _split(df, seed):
    indices = np.arange(len(df))
    train, remaining = train_test_split(indices, test_size=.30, stratify=df.stroke, random_state=seed)
    val, test = train_test_split(remaining, test_size=.50, stratify=df.stroke.iloc[remaining], random_state=seed)
    return train, val, test


def prepare_stroke_data(split_seed: int | None = None, balance_method: str | None = None, csv_path=None, output_dir=None, overwrite: bool = False):
    """Return X/y in the legacy six-part format with training-only balancing."""
    source_path = Path(csv_path) if csv_path is not None else RAW_PATH
    artifact_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    if not source_path.is_file():
        raise FileNotFoundError(f"Stroke CSV not found at {source_path}. Run: python -m experiments.stroke_experiments.download_stroke")
    raw_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if balance_method is None:
        balance_method = BALANCING_METHOD
    if balance_method not in {"random_oversample", "none"}:
        raise ValueError(f"Unsupported stroke balancing method: {balance_method!r}")
    if split_seed is None:
        # Reuse only the seed associated with this exact source file. Indices are
        # always rebuilt below, so stale index arrays can never be applied.
        previous_metadata = artifact_dir / METADATA_PATH.name
        try:
            prior = json.loads(previous_metadata.read_text(encoding="utf-8"))
            split_seed = int(prior["split_seed"]) if prior.get("dataset_sha256") == raw_hash else SPLIT_SEED
        except (OSError, ValueError, KeyError, TypeError):
            split_seed = SPLIT_SEED
    else:
        split_seed = int(split_seed)
    df = pd.read_csv(source_path, na_values=["N/A", "NA", "Unknown_na", ""])
    _validate(df)
    # Match the reference pipeline's exclusion of the one 'Other' gender row, while preserving source ids.
    df = df.loc[df.gender.ne("Other")].reset_index(drop=True)
    train_idx, val_idx, test_idx = _split(df, split_seed)
    preprocessor_path = artifact_dir / PREPROCESSOR_PATH.name
    metadata_path = artifact_dir / METADATA_PATH.name
    config_payload = {"version": PREPROCESSING_VERSION, "config": PREPROCESSING_CONFIG,
                     "split_seed": split_seed, "balancing_method": balance_method,
                     "libraries": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                                   "numpy": np.__version__, "python": platform.python_version()}}
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
    prior = None
    if metadata_path.exists() or preprocessor_path.exists():
        if not (metadata_path.exists() and preprocessor_path.exists()):
            if not overwrite:
                raise FileExistsError("Incomplete stroke preprocessing artifacts found; rerun process_stroke with --overwrite")
        else:
            try:
                prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                if not overwrite:
                    raise ValueError("Stroke metadata is unreadable; rerun process_stroke with --overwrite") from exc
                prior = None
            if prior is None:
                compatible = False
            else:
                compatible = (prior.get("dataset_sha256") == raw_hash
                              and prior.get("preprocessing_config_hash") == config_hash
                              and prior.get("preprocessing_version") == PREPROCESSING_VERSION)
            if not compatible and not overwrite:
                raise FileExistsError("Existing stroke preprocessing artifacts are stale or incompatible; rerun process_stroke with --overwrite")
    if prior and compatible and not overwrite:
        preprocessor = joblib.load(preprocessor_path)
        if preprocessor.feature_names != prior.get("feature_names"):
            raise ValueError("Saved stroke preprocessor schema does not match metadata; rerun process_stroke with --overwrite")
    else:
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
    if balance_method == "random_oversample" and len(original_counts) == 2:
        # Random oversampling duplicates actual minority examples. This avoids SMOTE's invalid
        # fractional binary flags and multi-hot category groups after the ZIP's one-hot encoding.
        rng = np.random.default_rng(split_seed)
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
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION, "preprocessing_config": PREPROCESSING_CONFIG,
        "preprocessing_config_hash": config_hash, "dataset_sha256": raw_hash, "split_seed": split_seed,
        "library_versions": config_payload["libraries"],
        "split_policy": SPLIT_POLICY, "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_row_ids": {name: df.id.iloc[idx].tolist() for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))},
        "resampled_train_row_ids": resampled_train_row_ids,
        "feature_names": preprocessor.feature_names, "input_size": len(preprocessor.feature_names),
        "original_train_class_counts": {str(k): int(v) for k, v in original_counts.items()},
        "resampled_train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()},
        "validation_class_counts": {str(k): int(v) for k, v in y_val.value_counts().sort_index().to_dict().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()},
        "balancing_method": balance_method,
    }
    if prior and compatible and not overwrite:
        # Validate every persisted provenance field, including the split seed/policy,
        # library versions, row identities, and balancing details. The config hash
        # alone cannot detect metadata edited independently of the fitted artifacts.
        for key, expected_value in metadata.items():
            if prior.get(key) != expected_value:
                raise ValueError(f"Saved stroke metadata field {key!r} does not match the current deterministic pipeline")
    if prior is None or overwrite:
        joblib.dump(preprocessor, preprocessor_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        # Do not rewrite matching artifacts; their provenance remains stable.
        metadata = prior
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_stroke_preprocessor(path=PREPROCESSOR_PATH):
    return joblib.load(path)


def load_stroke_metadata(path=METADATA_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))
