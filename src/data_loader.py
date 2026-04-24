import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/matches.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Añadimos variables lógicas para ejemplo
    df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
    df["draw"] = (df["home_goals"] == df["away_goals"]).astype(int)
    df["away_win"] = (df["home_goals"] < df["away_goals"]).astype(int)
    return df
