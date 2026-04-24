import pandas as pd
from src.data_loader import load_data
from src.features import build_features
from src.model import train_model

if __name__ == "__main__":
    print("Cargando datos...")
    raw_df = load_data()
    print(f"Datos cargados: {raw_df.shape[0]} partidos")

    print("Construyendo características...")
    feat_df = build_features(raw_df)

    print("Entrenando modelo...")
    model, metrics = train_model(feat_df)

    print("Métricas de entrenamiento:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
