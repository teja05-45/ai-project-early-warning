import pandas as pd
import joblib
import copy

from src.feature_engineering import engineer_features

class WhatIfSimulator:
    def __init__(self):
        self.model = joblib.load("models/delivery_risk_model.joblib")
        self.features = joblib.load("models/feature_columns.joblib")

    def predict_risk(self, row: pd.Series) -> float:
        df = pd.DataFrame([row])
        df = engineer_features(df)
        X = df[self.features]
        return self.model.predict_proba(X)[0][1]

    def simulate(self, row: pd.Series, actions: dict):
        base_risk = self.predict_risk(row)

        modified = copy.deepcopy(row)
        for feature, change in actions.items():
            if feature in modified:
                modified[feature] = max(0, modified[feature] * (1 + change))

        new_risk = self.predict_risk(modified)

        return {
            "baseline_risk": round(base_risk * 100, 1),
            "new_risk": round(new_risk * 100, 1),
            "risk_change": round((new_risk - base_risk) * 100, 1)
        }
