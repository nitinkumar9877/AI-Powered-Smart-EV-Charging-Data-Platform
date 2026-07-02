import json
import pickle

import numpy as np
import pandas as pd

from config import FEATURE_FILE, MODELS_DIR, TARGET_OCCUPANCY, TARGET_OVERLOAD, TARGET_WAITING_TIME


CATEGORICAL_FEATURES = [
    "Charging Station Location",
    "Vehicle Model",
    "Charger Type",
    "User Type",
    "weather_condition",
    "traffic_level",
]

NUMERIC_FEATURES = [
    "charging_hour",
    "charging_month",
    "is_weekend",
    "is_peak_hour",
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "Battery Capacity (kWh)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kph",
    "traffic_score",
    "road_congestion_index",
    "nearby_events_count",
]


def _try_sklearn_training(df: pd.DataFrame):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError:
        return None

    def preprocessing_pipeline() -> ColumnTransformer:
        numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        return ColumnTransformer(
            transformers=[("num", numeric_pipe, NUMERIC_FEATURES), ("cat", categorical_pipe, CATEGORICAL_FEATURES)]
        )

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    targets = {
        "occupancy_model": (TARGET_OCCUPANCY, RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=3)),
        "waiting_time_model": (TARGET_WAITING_TIME, RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=3)),
        "overload_risk_model": (
            TARGET_OVERLOAD,
            RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", min_samples_leaf=3),
        ),
    }

    metrics = {"training_backend": "scikit-learn_random_forest"}
    for model_name, (target, estimator) in targets.items():
        y = df[target]
        stratify = y if target == TARGET_OVERLOAD and y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
        model = Pipeline(steps=[("preprocess", preprocessing_pipeline()), ("model", estimator)])
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if target == TARGET_OVERLOAD:
            model_metrics = {
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            }
            if len(set(y_test)) > 1:
                model_metrics["roc_auc"] = round(float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])), 4)
        else:
            model_metrics = {
                "mae": round(float(mean_absolute_error(y_test, pred)), 4),
                "rmse": round(float(mean_squared_error(y_test, pred, squared=False)), 4),
                "r2": round(float(r2_score(y_test, pred)), 4),
            }
        metrics[model_name] = model_metrics
        with (MODELS_DIR / f"{model_name}.pkl").open("wb") as fh:
            pickle.dump(model, fh)
    return metrics


def _design_matrix(df: pd.DataFrame):
    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(df[NUMERIC_FEATURES].median(numeric_only=True))
    numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    categorical = pd.get_dummies(df[CATEGORICAL_FEATURES].fillna("Unknown"), drop_first=False, dtype=float)
    x = pd.concat([numeric, categorical], axis=1)
    x.insert(0, "intercept", 1.0)
    return x


def _train_test_split(df: pd.DataFrame, test_size=0.2):
    rng = np.random.default_rng(42)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    return indices[:split], indices[split:]


def _regression_metrics(y_true, pred):
    errors = y_true - pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - np.sum(errors**2) / denominator if denominator else 0
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}


def _classification_metrics(y_true, pred):
    accuracy = np.mean(y_true == pred)
    tp = np.sum((y_true == 1) & (pred == 1))
    fp = np.sum((y_true == 0) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"accuracy": round(float(accuracy), 4), "f1": round(float(f1), 4)}


def _fallback_training(df: pd.DataFrame):
    x = _design_matrix(df)
    train_idx, test_idx = _train_test_split(df)
    x_train = x.iloc[train_idx].to_numpy()
    x_test = x.iloc[test_idx].to_numpy()
    feature_columns = list(x.columns)

    metrics = {"training_backend": "numpy_least_squares_baseline"}
    for model_name, target in {
        "occupancy_model": TARGET_OCCUPANCY,
        "waiting_time_model": TARGET_WAITING_TIME,
    }.items():
        y_train = df.iloc[train_idx][target].to_numpy(dtype=float)
        y_test = df.iloc[test_idx][target].to_numpy(dtype=float)
        coefficients = np.linalg.pinv(x_train).dot(y_train)
        pred = x_test.dot(coefficients)
        metrics[model_name] = _regression_metrics(y_test, pred)
        with (MODELS_DIR / f"{model_name}.pkl").open("wb") as fh:
            pickle.dump({"backend": "numpy_least_squares", "features": feature_columns, "coefficients": coefficients}, fh)

    y_train = df.iloc[train_idx][TARGET_OVERLOAD].to_numpy(dtype=float)
    y_test = df.iloc[test_idx][TARGET_OVERLOAD].to_numpy(dtype=int)
    coefficients = np.linalg.pinv(x_train).dot(y_train)
    score = x_test.dot(coefficients)
    pred = (score >= 0.5).astype(int)
    metrics["overload_risk_model"] = _classification_metrics(y_test, pred)
    with (MODELS_DIR / "overload_risk_model.pkl").open("wb") as fh:
        pickle.dump({"backend": "numpy_linear_classifier", "features": feature_columns, "coefficients": coefficients}, fh)
    return metrics


def train_models() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_FILE)
    metrics = _try_sklearn_training(df)
    if metrics is None:
        metrics = _fallback_training(df)

    metrics_path = MODELS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Models saved to: {MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    train_models()
