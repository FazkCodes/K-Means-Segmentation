"""Assign new customers to a segment using the saved model."""
import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import FEATURES

def predict_segment(customers: pd.DataFrame, model_path="models/kmeans_pipeline.joblib"):
    model = joblib.load(model_path)
    return model.predict(customers[FEATURES])

if __name__ == "__main__":
    new = pd.DataFrame([
        {"age": 23, "annual_income_k": 48, "spending_score": 82, "visits_per_month": 10, "avg_basket_value": 55},
        {"age": 47, "annual_income_k": 115, "spending_score": 72, "visits_per_month": 6, "avg_basket_value": 190},
    ])
    new["cluster"] = predict_segment(new)
    print(new)
