import os
from os import path
import pandas as pd
import zipfile
import urllib.request

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset="yacht", test_size=0.2, data_path="data/"):
    if not os.path.exists(data_path):
        raise ValueError(
            f"Invalid 'data_path': '{data_path}'! Please make sure data directory exists."
        )
    elif dataset in [
        "concrete",
        "power",
        "yacht",
        "energy",
        "power",
    ]:
        X, y = UCIDataset(dataset, test_size=test_size, data_path=data_path).get_data()
    elif dataset == "kin8nm":
        # data_path = "data/"
        url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"
        file_name = "dataset_2175_kin8nm.csv"
        dir_name = os.path.join(data_path, "OpenML/")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        fpath = os.path.join(dir_name, file_name)
        if not path.exists(fpath):
            urllib.request.urlretrieve(url, fpath)
        data = pd.read_csv(fpath, header=0).to_numpy()
        X, y = (
            data[:, : data.shape[1] - 1],
            data[:, data.shape[1] - 1],
        )
    elif dataset in ["acsincome", "diabetes", "amazon", "iwildcam"]:
        try:
            X_train = np.load(os.path.join(data_path, dataset, "X_train.npy"))
            y_train = np.load(os.path.join(data_path, dataset, "y_train.npy"))
            X_test = np.load(os.path.join(data_path, dataset, "X_test.npy"))
            y_test = np.load(os.path.join(data_path, dataset, "y_test.npy"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find data files in 'data/{dataset}'. Did you run 'scripts/prepare_{dataset}'?"
            )

        X_train = torch.tensor(X_train, dtype=torch.float64)
        y_train = torch.tensor(y_train, dtype=torch.float64)
        X_test = torch.tensor(X_test, dtype=torch.float64)
        y_test = torch.tensor(y_test, dtype=torch.float64)

        return X_train, y_train, X_test, y_test
    elif dataset == "iwildcam":
        X_train = np.load(os.path.join(data_path, dataset, "X_train.npy"))
        y_train = np.load(os.path.join(data_path, dataset, "y_train.npy"))
        X_test = np.load(os.path.join(data_path, dataset, "X_test.npy"))
        y_test = np.load(os.path.join(data_path, dataset, "y_test.npy"))

        X_train = torch.tensor(X_train, dtype=torch.float64)
        y_train = torch.tensor(y_train, dtype=torch.float64)
        X_test = torch.tensor(X_test, dtype=torch.float64)
        y_test = torch.tensor(y_test, dtype=torch.float64)

        return X_train, y_train, X_test, y_test
    elif dataset == "amazon":
        raise NotImplementedError
    elif dataset == "iwildcam_std":
        raise NotImplementedError
    else:
        raise Exception(f"Unknown dataset: {dataset}!")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Standardize inputs.
    scaler = StandardScaler().fit(X_train)
    center = y_train.mean()
    spread = y_train.std(ddof=1)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = (y_train - center) / spread
    y_test = (y_test - center) / spread

    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    return X_train, y_train, X_test, y_test


class UCIDataset:
    def __init__(self, name, test_size=0.2, data_path="../data/"):
        self.datasets = {
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        }
        self.data_path = data_path
        self.name = name
        self._load_dataset()
        self.test_size = test_size

    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path + "UCI"):
            os.mkdir(self.data_path + "UCI")

        url = self.datasets[self.name]
        file_name = url.split("/")[-1]
        if not path.exists(self.data_path + "UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path + "UCI/" + file_name
            )
        data = None

        if self.name == "concrete":
            self.data = pd.read_excel(
                self.data_path + "UCI/Concrete_Data.xls", header=0
            ).to_numpy()
            # self.data = np.delete(self.data, [491, 492], axis=0)
        elif self.name == "energy":
            self.data = pd.read_excel(
                self.data_path + "UCI/ENB2012_data.xlsx", header=0
            ).to_numpy()
        elif self.name == "power":
            if not os.path.exists(self.data_path + "UCI/CCPP/CCPP/Folds5x2_pp.xlsx"):
                zipfile.ZipFile(self.data_path + "UCI/CCPP.zip").extractall(
                    self.data_path + "UCI/CCPP/"
                )
            self.data = pd.read_excel(
                self.data_path + "UCI/CCPP/CCPP/Folds5x2_pp.xlsx", header=0
            ).to_numpy()
        elif self.name == "yacht":
            self.data = pd.read_csv(
                self.data_path + "UCI/yacht_hydrodynamics.data",
                header=1,
                delimiter="\s+",
            ).to_numpy()
        elif self.name == "naval":
            if not os.path.exists(self.data_path + "UCI/UCI CBM Dataset/data.txt"):
                zipfile.ZipFile(
                    self.data_path + "UCI/UCI%20CBM%20Dataset.zip"
                ).extractall(self.data_path + "UCI/")
            self.data = np.genfromtxt(self.data_path + "UCI/UCI CBM Dataset/data.txt")
        elif self.name == "protein":
            self.data = pd.read_csv(
                self.data_path + "UCI/CASP.csv",
                header=0,
            ).to_numpy()

    def get_data(self):
        if self.name == "energy":
            return (
                # Two responses for this dataset. Pick the first.
                self.data[:, : self.data.shape[1] - 2],
                self.data[:, self.data.shape[1] - 1],
            )
        else:
            return (
                self.data[:, : self.data.shape[1] - 1],
                self.data[:, self.data.shape[1] - 1],
            )
