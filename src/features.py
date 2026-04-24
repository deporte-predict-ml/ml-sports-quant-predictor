import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    # Ejemplo: diferencia de xG y de remates
    feats["xg_diff"] = feats["home_xg"] - feats["away_xg"]
    feats["shots_diff"] = feats["home_shots"] - feats["away_shots"]
    feats["total_xg"] = feats["home_xg"] + feats["away_xg"]
    feats["total_shots"] = feats["home_shots"] + feats["away_shots"]
    # Variable objetivo binaria: gana el local
    feats["target_home_win"] = feats["home_win"]
    return feats[
        ["xg_diff", "shots_diff", "total_xg", "total_shots", "target_home_win"]
    ]
