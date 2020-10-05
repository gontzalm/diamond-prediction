from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.preprocesssors import PREPROCESSORS
from src.load import load_diamonds
from src.splitters import split_train_test, split_X_y


MODELS = {}
for prep_name, preprocessor in PREPROCESSORS.items():
    MODELS[prep_name] = {
        "linear": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]),
    
        "ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge())
        ]),
    
        "elasticnet": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", ElasticNet())
        ]),
    
        "kneighbors": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", KNeighborsRegressor()) 
        ]),
    
        "tree": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor()) 
        ]),
    
        "forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor())
        ])
    }


def top_models():
    """Print a summary of different models."""
    # Load and split dataset
    diamonds = load_diamonds()
    diamonds, _ = split_train_test(diamonds)
    X_train, y_train = split_X_y(diamonds)

    # Select processors
    for prep_name, model_dict in MODELS.items():
        print(f"{prep_name}".center(80, "_"))

        # Train and evaluate model
        for model_name, model in model_dict.items():
            print(f"{model_name}".center(80, "-"))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            rmse = mean_squared_error(y_train, y_pred, squared=False)
            scores = cross_val_score(
                model, X_train, y_train, 
                scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
            )
            scores = -scores
            print(f"Training set, RMSE: {rmse:.2f}")
            print(f"Cross-val, mean RMSE: {scores.mean():.2f}")
            print(f"Cross-val, std RMSE: {scores.std():.2f}")