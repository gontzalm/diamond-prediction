import numpy as np
from itertools import product
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from src.conf import CATEGORIES, CATEGORIES_DEPTH_TABLE
from src.transformers import AttributeAdder, CatEncoder

# Define different pipelines for the categories
NUM_PIPE = {
    "adder_all": Pipeline([
        ("attr_adder", AttributeAdder()),
        ("std_scaler", StandardScaler())
    ]),

    "adder_best": Pipeline([
        ("attr_adder", AttributeAdder(which=["x_squared_times_y"])),
        ("std_scaler", StandardScaler())
    ]),

    "no_adder": StandardScaler()
}

DEPTH_PIPE = {
    "ord_enc": Pipeline([
        ("cat_enc", CatEncoder("depth")),
        ("ordinal_enc", OrdinalEncoder(categories=CATEGORIES_DEPTH_TABLE)),
        ("std_scaler", StandardScaler())
    ]),

    "1_hot_enc": Pipeline([
        ("cat_enc", CatEncoder("depth")),
        ("1_hot_enc", OneHotEncoder(drop="first"))
    ])
}

TABLE_PIPE = {
    "ord_enc": Pipeline([
        ("cat_enc", CatEncoder("table")),
        ("ordinal_enc", OrdinalEncoder(categories=CATEGORIES_DEPTH_TABLE)),
        ("std_scaler", StandardScaler())
    ]),

    "1_hot_enc": Pipeline([
        ("cat_enc", CatEncoder("table")),
        ("1_hot_enc", OneHotEncoder(drop="first"))
    ])
}

OTHER_CAT_PIPE = {
    "ord_enc": Pipeline([
        ("ordinal_enc", OrdinalEncoder(categories=CATEGORIES)),
        ("std_scaler", StandardScaler())
    ]),

    "1_hot_enc": OneHotEncoder(drop="first")
}

# Define different preprocessors
PREPROCESSORS = {
    "no_depth_no_table": ColumnTransformer([
        ("num", StandardScaler(), ["carat", "x", "y", "z"]),
        ("cat", OTHER_CAT_PIPE["1_hot_enc"], ["cut", "color", "clarity"]),
    ])
}

for add, enc in product(NUM_PIPE.keys(), DEPTH_PIPE.keys()):
    k = "_".join([add, enc])
    PREPROCESSORS[k] = ColumnTransformer([
        ("num", NUM_PIPE[add], ["carat", "x", "y", "z"]),
        ("depth", DEPTH_PIPE[enc], "depth"),
        ("table", TABLE_PIPE[enc], "table"),
        ("cat", OTHER_CAT_PIPE[enc], ["cut", "color", "clarity"]),
    ])