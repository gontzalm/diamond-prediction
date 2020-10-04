import pandas as pd
from sklearn.base import TransformerMixin
from src.conf import BINS, CATEGORIES_DEPTH_TABLE_BINS


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