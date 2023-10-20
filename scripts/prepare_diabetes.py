import os
import urllib.request
import pandas as pd
from scipy.io import arff

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Download raw data.
data_path = "../data/"
url = "https://github.com/fairlearn/talks/raw/main/2021_scipy_tutorial/data/diabetic_data.csv"
print(f"Downloading raw data from '{url}'...")
urllib.request.urlretrieve(url, os.path.join(data_path, "UCI", "diabetic_data.csv"))

# Repeat processing used in the tutorial.
print(
    "Applying preprocessing used in 'https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py'..."
)
df = pd.read_csv(os.path.join(data_path, "UCI", "diabetic_data.csv")).rename(
    columns={"diag_1": "primary_diagnosis"}
)
# Create Outcome variables
df.loc[:, "readmit_30_days"] = df["readmitted"] == "<30"
df.loc[:, "readmit_binary"] = df["readmitted"] != "NO"
# Replace missing values and re-code categories
df.loc[:, "age"] = df.age.replace({"?": ""})
df.loc[:, "payer_code"] = df["payer_code"].replace({"?", "Unknown"})
df.loc[:, "medical_specialty"] = df["medical_specialty"].replace({"?": "Missing"})
df.loc[:, "race"] = df["race"].replace({"?": "Unknown"})

df.loc[:, "admission_source_id"] = df["admission_source_id"].replace(
    {1: "Referral", 2: "Referral", 3: "Referral", 7: "Emergency"}
)
df.loc[:, "age"] = df["age"].replace(
    ["[0-10)", "[10-20)", "[20-30)"], "30 years or younger"
)
df.loc[:, "age"] = df["age"].replace(["[30-40)", "[40-50)", "[50-60)"], "30-60 years")
df.loc[:, "age"] = df["age"].replace(["[60-70)", "[70-80)", "[80-90)"], "Over 60 years")

# Clean various medical codes
df.loc[:, "discharge_disposition_id"] = df.discharge_disposition_id.apply(
    lambda x: "Discharged to Home" if x == 1 else "Other"
)
df.loc[:, "admission_source_id"] = df["admission_source_id"].apply(
    lambda x: x if x in ["Emergency", "Referral"] else "Other"
)
# Re-code Medical Specialties and Primary Diagnosis
specialties = [
    "Missing",
    "InternalMedicine",
    "Emergency/Trauma",
    "Family/GeneralPractice",
    "Cardiology",
    "Surgery",
]
df.loc[:, "medical_specialty"] = df["medical_specialty"].apply(
    lambda x: x if x in specialties else "Other"
)
#
df.loc[:, "primary_diagnosis"] = df["primary_diagnosis"].replace(
    regex={
        "[7][1-3][0-9]": "Musculoskeltal Issues",
        "250.*": "Diabetes",
        "[4][6-9][0-9]|[5][0-1][0-9]|786": "Respitory Issues",
        "[5][8-9][0-9]|[6][0-2][0-9]|788": "Genitourinary Issues",
    }
)
diagnoses = [
    "Respitory Issues",
    "Diabetes",
    "Genitourinary Issues",
    "Musculoskeltal Issues",
]
df.loc[:, "primary_diagnosis"] = df["primary_diagnosis"].apply(
    lambda x: x if x in diagnoses else "Other"
)

# Binarize and bin features
df.loc[:, "medicare"] = df.payer_code == "MC"
df.loc[:, "medicaid"] = df.payer_code == "MD"

df.loc[:, "had_emergency"] = df["number_emergency"] > 0
df.loc[:, "had_inpatient_days"] = df["number_inpatient"] > 0
df.loc[:, "had_outpatient_days"] = df["number_outpatient"] > 0

# Save DataFrame
cols_to_keep = [
    "race",
    "gender",
    "age",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "primary_diagnosis",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
    "medicare",
    "medicaid",
    "had_emergency",
    "had_inpatient_days",
    "had_outpatient_days",
    "readmitted",
    "readmit_binary",
    "readmit_30_days",
]

final_df = df.loc[:, cols_to_keep]
# final_df.to_csv(data_path / "diabetic_preprocessed.csv", index=False)
final_df.to_csv(
    os.path.join(data_path, "UCI", "diabetic_preprocessed.csv"), index=False
)


# Take subset of 5,000 points.
n = 5000
seed = 42
dfs = final_df.sample(n=n, random_state=seed)

# Drop columns that have a large number of categories (e.g. place of birth and occuption).
feat_cols = [
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "primary_diagnosis",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
    "medicare",
    "medicaid",
    "had_emergency",
    "had_inpatient_days",
    "had_outpatient_days",
]

metadata_cols = ["gender", "race"]
target_col = "readmit_30_days"
dfs = dfs[feat_cols + metadata_cols + [target_col]]
print("Selecting feature columns:", feat_cols)
print("Selecting metadata columns:", metadata_cols)
print("Selecting target:", target_col)

# Standardize.
print("Performing train/test split and standardizing features and labels...")
train, test = train_test_split(dfs, test_size=0.2, random_state=seed)

X_train = pd.get_dummies(train[feat_cols], drop_first=True).to_numpy()
y_train = train[target_col].to_numpy().astype(int)
metadata_tr = train[metadata_cols]

X_test = pd.get_dummies(test[feat_cols], drop_first=True).to_numpy()
y_test = test[target_col].to_numpy().astype(int)
metadata_te = test[metadata_cols]

print("Train features shape:", X_train.shape)
print("Train labels shape:", y_train.shape)
print("Test features shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Saving data files...")
dir_name = "diabetes"
os.makedirs(os.path.join(data_path, dir_name), exist_ok=True)

np.save(os.path.join(data_path, dir_name, "X_train"), X_train)
np.save(os.path.join(data_path, dir_name, "y_train"), y_train)
metadata_tr.to_csv(os.path.join(data_path, dir_name, "metadata_tr.csv"))

np.save(os.path.join(data_path, dir_name, "X_test"), X_test)
np.save(os.path.join(data_path, dir_name, "y_test"), y_test)
metadata_te.to_csv(os.path.join(data_path, dir_name, "metadata_te.csv"))
