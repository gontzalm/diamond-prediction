import pandas as pd


DATASETS_PATH = "../datasets"
ORIGINAL = "diamonds.csv"


def load_diamonds():
    """Load the diamond dataset into a pandas.DataFrame."""
    path = f"{DATASETS_PATH}/{ORIGINAL}"
    return pd.read_csv(path, engine="python", index_col="id")