import pandas as pd
from sklearn.base import TransformerMixin
from src.conf import BINS, CATEGORIES_DEPTH_TABLE_BINS


class AttributeAdder(TransformerMixin):
    """Transformer that adds new attributes to the dataset."""
    def __init__(self, which=None):
        self.which = which

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.which:
            X["x_squared"] = X["x"] ** 2
            X["x_times_y"] = X["x"] * X["y"]
            X["x_squared_times_y"] = X["x_squared"] * X["y"]
        else:
            if "x_squared" in self.which:
                X["x_squared"] = X["x"] ** 2
            if "x_times_y" in self.which:
                X["x_times_y"] = X["x"] * X["y"]
            if "x_squared_times_y" in self.which:
                X["x_squared_times_y"] = (X["x"] ** 2) * X["y"]
        return X


class CatEncoder(TransformerMixin):
    """Categorical encoder for the depth and table attributes."""
    def __init__(self, attr):
        self.attr = attr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bins = BINS[self.attr]
        return pd.cut(
            X,
            bins=bins,
            labels=CATEGORIES_DEPTH_TABLE_BINS,
            ordered=False
        ).to_frame()