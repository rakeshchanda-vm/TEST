"""
Credit Scoring Model Training Pipeline.
Uses XGBoost with Optuna hyperparameter optimization.
Tracks experiments in MLflow.
"""
import mlflow
import mlflow.xgboost
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from pathlib import Path
from backend.config.config import settings
from backend.ml.credit_model.features import FEATURE_NAMES

MODEL_DIR = Path("models/credit")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare training data. Replace with your actual data source."""
    # In production: load from PostgreSQL feature store
    # df = pd.read_sql("SELECT * FROM credit_features WHERE split='train'", engine)
    # For demo: generate synthetic data
    np.random.seed(42)
    n = 10000
    X = np.random.randn(n, len(FEATURE_NAMES))
    # Synthetic target: default rate ~15%
    y = (X[:, 2] > 0.5).astype(int)  # High DTI → default
    return X, y


def objective(trial, X, y) -> float:
    """Optuna objective function for hyperparameter search."""
    params = {
        "max_depth":           trial.suggest_int("max_depth", 3, 10),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":        trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight":    trial.suggest_int("min_child_weight", 1, 10),
        "subsample":           trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":           trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "scale_pos_weight":    trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "use_label_encoder":   False,
        "eval_metric":         "auc",
        "tree_method":         "hist",
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def train():
    print("Loading training data...")
    X, y = load_training_data()

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    print("🔍 Hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_scaled, y_res), n_trials=50, n_jobs=-1)

    best_params = study.best_params
    print(f"Best AUC: {study.best_value:.4f}")

    with mlflow.start_run(run_name="credit_model_v1"):
        mlflow.log_params(best_params)

        final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="auc")
        final_model.fit(X_scaled, y_res)

        preds = final_model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y_res, preds)

        mlflow.log_metric("train_auc", auc)
        mlflow.xgboost.log_model(final_model, "credit_model")

        # Save artifacts
        joblib.dump(final_model, MODEL_DIR / "credit_model.pkl")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

        print(f"Model trained. AUC: {auc:.4f}")
        print(f"Saved to {MODEL_DIR}")
        return final_model, scaler


if __name__ == "__main__":
    print("Starting Execution")
    train()