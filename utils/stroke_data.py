import os
import pandas as pd
import kagglehub
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


SPLIT_SEED = 42

SPLIT_PATH = "data/stroke_split.pt"

CATEGORICAL_COLUMNS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

NUMERICAL_COLUMNS = [
    "age",
    "avg_glucose_level",
    "bmi"
]


def load_stroke_data():
    path = kagglehub.dataset_download(
        "fedesoriano/stroke-prediction-dataset"
    )

    csv_path = os.path.join(
        path,
        "healthcare-dataset-stroke-data.csv"
    )

    return pd.read_csv(csv_path)


def create_split(df):
    indices = df.index.to_numpy()

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        stratify=df["stroke"],
        random_state=SPLIT_SEED
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=df.loc[temp_indices, "stroke"],
        random_state=SPLIT_SEED
    )

    split = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "split_seed": SPLIT_SEED
    }

    os.makedirs(
        os.path.dirname(SPLIT_PATH),
        exist_ok=True
    )

    torch.save(
        split,
        SPLIT_PATH
    )

    return split


def load_or_create_split(df):
    if os.path.exists(SPLIT_PATH):
        return torch.load(
            SPLIT_PATH,
            weights_only=False
        )

    return create_split(df)


def prepare_stroke_data():
    df = load_stroke_data()

    split = load_or_create_split(df)

    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    test_indices = split["test_indices"]

    df = df.drop(columns=["id"])

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train = X.loc[train_indices].copy()
    X_val = X.loc[val_indices].copy()
    X_test = X.loc[test_indices].copy()

    y_train = y.loc[train_indices].copy()
    y_val = y.loc[val_indices].copy()
    y_test = y.loc[test_indices].copy()

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_num = numerical_imputer.fit_transform(
        X_train[NUMERICAL_COLUMNS]
    )

    X_val_num = numerical_imputer.transform(
        X_val[NUMERICAL_COLUMNS]
    )

    X_test_num = numerical_imputer.transform(
        X_test[NUMERICAL_COLUMNS]
    )

    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(
        X_train_num
    )

    X_val_num = scaler.transform(
        X_val_num
    )

    X_test_num = scaler.transform(
        X_test_num
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = encoder.fit_transform(
        X_train[CATEGORICAL_COLUMNS]
    )

    X_val_cat = encoder.transform(
        X_val[CATEGORICAL_COLUMNS]
    )

    X_test_cat = encoder.transform(
        X_test[CATEGORICAL_COLUMNS]
    )

    categorical_feature_names = (
        encoder.get_feature_names_out(
            CATEGORICAL_COLUMNS
        )
    )

    feature_names = (
        NUMERICAL_COLUMNS
        + list(categorical_feature_names)
    )

    X_train = pd.DataFrame(
        X_train_num,
        index=X_train.index,
        columns=NUMERICAL_COLUMNS
    )

    X_train = pd.concat(
        [
            X_train,
            pd.DataFrame(
                X_train_cat,
                index=X_train.index,
                columns=categorical_feature_names
            )
        ],
        axis=1
    )

    X_val = pd.DataFrame(
        X_val_num,
        index=X_val.index,
        columns=NUMERICAL_COLUMNS
    )

    X_val = pd.concat(
        [
            X_val,
            pd.DataFrame(
                X_val_cat,
                index=X_val.index,
                columns=categorical_feature_names
            )
        ],
        axis=1
    )

    X_test = pd.DataFrame(
        X_test_num,
        index=X_test.index,
        columns=NUMERICAL_COLUMNS
    )

    X_test = pd.concat(
        [
            X_test,
            pd.DataFrame(
                X_test_cat,
                index=X_test.index,
                columns=categorical_feature_names
            )
        ],
        axis=1
    )

    X_train = X_train[feature_names]
    X_val = X_val[feature_names]
    X_test = X_test[feature_names]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )