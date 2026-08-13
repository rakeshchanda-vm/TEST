"""Fraud detection model training — Isolation Forest."""
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from pathlib import Path
from backend.config.config import settings

MODEL_DIR = Path("models/fraud")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def train():
    print("🔄 Training Isolation Forest fraud detector...")
    np.random.seed(42)
    n_normal = 9000
    n_fraud = 1000

    normal = np.random.randn(n_normal, 4) * [0.3, 0.2, 0.4, 0.2] + [0.5, 0.3, 0.4, 0.3]
    fraud = np.random.randn(n_fraud, 4) * [0.5, 0.6, 0.8, 0.3] + [0.1, 0.8, 0.1, 0.7]
    X = np.vstack([normal, fraud])

    with mlflow.start_run(run_name="fraud_model_isolation_forest"):
        model = IsolationForest(
            n_estimators=200,
            contamination=0.1,   # ~10% fraud rate
            max_samples="auto",
            random_state=42,
        )
        model.fit(X)

        mlflow.sklearn.log_model(model, "fraud_model")
        mlflow.log_param("contamination", 0.1)
        mlflow.log_param("n_estimators", 200)

        joblib.dump(model, MODEL_DIR / "fraud_model.pkl")
        print(f"✅ Fraud model saved to {MODEL_DIR}")

    return model


if __name__ == "__main__":
    train()