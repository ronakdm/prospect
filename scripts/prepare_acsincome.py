import os
import urllib.request
import pandas as pd
from scipy.io import arff

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Download raw data.
data_path = "../data/"
url = "https://www.openml.org/data/download/22101666/ACSIncome_state_number.arff"
print(f"Downloading raw data from '{url}'...")
file_name = "ACSIncome_state_number.arff"
dir_name = os.path.join(data_path, "OpenML/")
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

fpath = os.path.join(dir_name, file_name)
if not os.path.exists(fpath):
    urllib.request.urlretrieve(url, fpath)

data = arff.loadarff(fpath)
df = pd.DataFrame(data[0])

# Take subset of 5,000 points.
n = 5000
seed = 42
dfs = df.sample(n=n, random_state=seed)

# Drop columns that have a large number of categories (e.g. place of birth and occuption).
dtypes = {
    "AGEP": "float",
    "COW": "string",
    "SCHL": "string",
    "MAR": "string",
    "RELP": "string",
    "WKHP": "float",
    "SEX": "string",
    "RAC1P": "string",
    "PINCP": "float",
}
feat_cols = ["AGEP", "COW", "SCHL", "MAR", "RELP", "WKHP"]  # Input features.
metadata_cols = ["SEX", "RAC1P"]  # Protected attribute.
target_col = "PINCP"
print("Selecting feature columns:", feat_cols)
print("Selecting metadata columns:", metadata_cols)
print("Selecting target:", target_col)

dfs = dfs[feat_cols + metadata_cols + [target_col]]
dfs = dfs.astype(dtypes)
dfs = pd.get_dummies(dfs, drop_first=True, columns=feat_cols)

# Standardize.
print("Performing train/test split and standardizing features and labels...")
train, test = train_test_split(dfs, test_size=0.2, random_state=seed)

X_train = train.drop(columns=metadata_cols + [target_col]).to_numpy()
y_train = np.log(train[target_col].to_numpy())
metadata_tr = train[metadata_cols]

X_test = test.drop(columns=metadata_cols + [target_col]).to_numpy()
y_test = np.log(test[target_col].to_numpy())
metadata_te = test[metadata_cols]

print("Train features shape:", X_train.shape)
print("Train labels shape:", y_train.shape)
print("Test features shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

scaler = StandardScaler().fit(X_train)
center = y_train.mean()
spread = y_train.std(ddof=1)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = (y_train - center) / spread
y_test = (y_test - center) / spread

print("Saving data files...")
dir_name = "acsincome"
os.makedirs(os.path.join(data_path, dir_name), exist_ok=True)

np.save(os.path.join(data_path, dir_name, "X_train"), X_train)
np.save(os.path.join(data_path, dir_name, "y_train"), y_train)
metadata_tr.to_csv(os.path.join(data_path, dir_name, "metadata_tr.csv"))

np.save(os.path.join(data_path, dir_name, "X_test"), X_test)
np.save(os.path.join(data_path, dir_name, "y_test"), y_test)
metadata_te.to_csv(os.path.join(data_path, dir_name, "metadata_te.csv"))
