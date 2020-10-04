import numpy as np


BINS = {
    "depth": [np.NINF, 56.49, 57.49, 57.99, 58.99, 62.3, 63.5, 64.1, 65, np.inf],
    "table": [np.NINF, 49.99, 50, 51, 53, 58, 60, 64, 69, np.inf]
}

CATEGORIES_DEPTH_TABLE_BINS = ["Poor", "Fair", "Good", "Very Good", "Excellent", "Very Good", "Good", "Fair", "Poor"]

CATEGORIES_DEPTH_TABLE = [["Poor", "Fair", "Good", "Very Good", "Excellent"]]

CATEGORIES = [
    ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    ["J", "I", "H", "G", "F", "E", "D"],
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
]