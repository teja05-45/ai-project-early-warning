import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.feature_engineering import engineer_features

def train_model():
    df = pd.read_csv("data/raw/project_delivery_risk.csv")
    df = engineer_features(df)

    y = df["delivery_risk"]
    X = df.drop(columns=["delivery_risk", "project_id", "sprint_id"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC: {auc:.3f}")

    joblib.dump(model, "models/delivery_risk_model.joblib")
    joblib.dump(list(X.columns), "models/feature_columns.joblib")

if __name__ == "__main__":
    train_model()
