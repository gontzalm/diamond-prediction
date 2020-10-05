import pandas as pd
from src.conf import SUBMISSIONS_PATH
from src.load import load_diamonds, load_predict
from src.splitters import split_X_y


def predict_diamonds(model, submission_num):
    """Predict the diamond prices and export a .csv."""
    # Load diamonds to predict and diamonds dataset
    X_pred = load_predict()
    diamonds = load_diamonds()

    # Train model on the whole dataset
    X_train, y_train = split_X_y(diamonds)
    model.fit(X_train, y_train)

    # Predict the prices
    y_pred = pd.DataFrame(model.predict(X_pred), columns=["price"])
    y_pred.index.name = "id"
    
    # Export to .csv
    if submission_num < 10:
        path = f"{SUBMISSIONS_PATH}/submission-0{submission_num}.csv"
    else:
        path = f"{SUBMISSIONS_PATH}/submission-{submission_num}.csv"

    y_pred.to_csv(path)