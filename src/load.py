import pandas as pd


DATASETS_PATH = "../datasets"
ORIGINAL = "diamonds.csv"
PREDICT = "predict.csv"


def load_diamonds():
    """Load the diamond dataset into a pandas.DataFrame."""
    path = f"{DATASETS_PATH}/{ORIGINAL}"
    return pd.read_csv(path, index_col="id", engine="python")


def load_predict():
    """Load the prediction dataset into a pandas.DataFrame."""
    path = f"{DATASETS_PATH}/{PREDICT}"
    return pd.read_csv(path, index_col="id", engine="python")