import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple

RANDOM_STATE = 42


def train_model(
    df: pd.DataFrame,
) -> Tuple[RandomForestClassifier, dict]:
    X = df.drop("target_home_win", axis=1)
    y = df["target_home_win"]

    # Simple train: todo el conjunto (en producción separarías train/test)
    model = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_STATE
    )
    model.fit(X, y)

    # Predicción en el mismo conjunto para ver accuracy
    y_pred = model.predict(X)
    metrics = {"accuracy": accuracy_score(y, y_pred)}

    return model, metrics
