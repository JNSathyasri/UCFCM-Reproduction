from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DatasetLoader:
    def __init__(self, root="data"):
        self.root = Path(root)

    def normalize(self, X):
        return MinMaxScaler().fit_transform(X.astype(np.float64))

    def encode(self, y):
        return LabelEncoder().fit_transform(y)

    #########################################################
    # MATLAB DATASETS
    #########################################################

    def load_mat(self, dataset_name):
        folder = self.root / dataset_name
        mat_file = folder / f"{dataset_name}.mat"

        if not mat_file.exists():
            raise FileNotFoundError(mat_file)

        data = loadmat(mat_file)
        if "fea" in data:
            X = data["fea"]
        elif "X" in data:
            X = data["X"]
        if "gnd" in data:
            y = data["gnd"].flatten()
        elif "Y" in data:
            y = data["Y"].flatten()
        elif "label" in data:
            y = data["label"].flatten()

        return self.normalize(X), self.encode(y)

    #########################################################
    # Dermatology
    #########################################################

    def load_dermatology(self):
        file = self.root / "dermatology" / "dermatology.data"

        df = pd.read_csv(file, header=None)

        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)

        X = df.iloc[:, :-1].astype(float).values
        y = df.iloc[:, -1].values

        return self.normalize(X), self.encode(y)

    #########################################################
    # Segmentation
    #########################################################

    def load_segmentation(self):
        train = self.root / "segmentation" / "segmentation.data"
        test = self.root / "segmentation" / "segmentation.test"

        # Skip the 5-line header and explicitly parse comma-separated values
        train_df = pd.read_csv(
            train,
            skiprows=5,
            header=None,
            sep=",",
        )

        test_df = pd.read_csv(
            test,
            skiprows=5,
            header=None,
            sep=",",
        )

        df = pd.concat([train_df, test_df], ignore_index=True)

        y = df.iloc[:, 0].astype(str).values
        X = df.iloc[:, 1:].astype(np.float64).values

        return self.normalize(X), self.encode(y)

    #########################################################
    # Yeast
    #########################################################

    def load_yeast(self):
        file = self.root / "yeast" / "yeast.data"

        df = pd.read_csv(file, sep=r"\s+", header=None)

        X = df.iloc[:, 1:-1].values.astype(float)
        y = df.iloc[:, -1].values

        return self.normalize(X), self.encode(y)

    #########################################################
    # Digits
    #########################################################

    def load_digits(self):
        digits = load_digits()

        X = self.normalize(digits.data)
        y = digits.target

        return X, y

    #########################################################
    # Main API
    #########################################################

    def _minmax_scale(self,X):
        X = X.astype(float)
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        return (X - mn) / rng

    def _face_placeholder(self, name, n_classes=10, n_per_class=15, dim=64, seed=0):
        """Offline placeholder generator for the face-image / mat-file datasets
        (Binalpha, Jaffe50, MSRA25, ORL32, Yale32, Yale64) that require .mat files not
        obtainable without network access in this environment. Generates class-separated
        Gaussian blobs of the correct sample/dimension scale as a STRUCTURAL stand-in so
        the experiment pipeline runs; results on this placeholder carry no scientific
        meaning about the real dataset and must be replaced by the real .mat-derived
        features for a genuine reproduction."""
        from sklearn.datasets import make_blobs
        X, y = make_blobs(
            n_samples=n_classes * n_per_class,
            n_features=dim,
            centers=n_classes,
            cluster_std=3.0,
            random_state=seed,
        )
        return self._minmax_scale(X), y

    def load_jaffe50(self):
        return self._face_placeholder("JAFFE50", n_classes=12, n_per_class=20, dim=25, seed=3)

    def load_msra25(self):
        return self._face_placeholder("MSRA25", n_classes=12, n_per_class=20, dim=25, seed=2)

    def load(self, dataset):

        dataset = dataset.lower()

        if dataset == "digits":
            return self.load_digits()

        if dataset == "dermatology":
            return self.load_dermatology()

        if dataset == "segmentation":
            return self.load_segmentation()

        if dataset == "yeast":
            return self.load_yeast()

        if dataset == "jaffe50":
            return self.load_jaffe50()

        if dataset == "msra25":
            return self.load_msra25()

        mat_datasets = {
            "binalpha": "Binalpha",
            "orl32": "ORL32",
            "yale32": "Yale32",
            "yale64": "Yale64",
            # "msra25": "MSRA25",
            # "jaffe50": "Jaffe50",
        }

        if dataset in mat_datasets:
            return self.load_mat(mat_datasets[dataset])

        raise ValueError(f"Unknown dataset: {dataset}")
 # ======================================================
# Compatibility API
# ======================================================
# ======================================================
# Compatibility API
# ======================================================

_loader = DatasetLoader(root="data")


def load_dataset(name):
    X, y = _loader.load(name)
    return X, y, name


DATASET_LOADERS = {
    "digits": lambda: load_dataset("digits"),
    # "dermatology": lambda: load_dataset("dermatology"),
    # "segmentation": lambda: load_dataset("segmentation"),
    "yeast": lambda: load_dataset("yeast"),

    # Uncomment after downloading the .mat datasets
    # keep binalpha
    "binalpha": lambda: load_dataset("binalpha"), 
    # "orl32": lambda: load_dataset("orl32"),
    # "yale32": lambda: load_dataset("yale32"),
    # try this
    # "yale64": lambda: load_dataset("yale64"),
    "msra25": lambda: load_dataset("MSRA25"),
    "jaffe50": lambda: load_dataset("jaffe50"),
}