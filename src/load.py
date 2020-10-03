import pandas as pd


DATASETS_PATH = "../datasets"

def load_diamonds():
    """Load the diamond dataset into a pandas.DataFrame."""
    return pd.read_csv(DATASETS_PATH, engine="python")