import pandas as pd
from src.features import build_features
from src.model import train_model


def test_pipeline():
    df = pd.DataFrame(
        {
            "home_xg": [2.1, 1.6, 3.2, 1.8],
            "away_xg": [1.3, 0.8, 1.1, 1.9],
            "home_shots": [15, 12, 20, 14],
            "away_shots": [9, 7, 8, 16],
            "home_goals": [2, 1, 3, 1],
            "away_goals": [1, 0, 1, 1],
            "home_win": [1, 1, 1, 0],
            "draw": [0, 0, 0, 1],
            "away_win": [0, 0, 0, 1],
        }
    )
    df["target_home_win"] = df["home_win"]

    feat_df = build_features(df)
    model, metrics = train_model(feat_df)

    # muy básicamente comprobamos que el modelo se entrena
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.0
