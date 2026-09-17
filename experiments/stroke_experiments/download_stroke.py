import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download(
    "fedesoriano/stroke-prediction-dataset"
)

csv_path = os.path.join(
    path,
    "healthcare-dataset-stroke-data.csv"
)

df = pd.read_csv(csv_path)

print("Dataset path:", path)
print("Shape:", df.shape)
print(df.head())