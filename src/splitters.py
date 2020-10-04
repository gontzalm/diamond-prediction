import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_train_test(diamonds):
    """Split the data performing a stratified split (carat)."""
    # Create carat categories
    diamonds["carat_cat"] = pd.cut(
        diamonds["carat"],
        bins=[0.2, 0.7, 1.2, 1.7, np.inf],
        labels=False,
        include_lowest=True
    )
    
    # Create a stratified splitter with fixed random state
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    # Split the dataset
    for train_idx, test_idx in splitter.split(diamonds, diamonds["carat_cat"]):
        train_set = diamonds.loc[train_idx]
        test_set = diamonds.loc[test_idx]
    
    # Drop the category column
    for set_ in train_set, test_set:
        set_.drop(columns="carat_cat", inplace=True)

    return train_set, test_set