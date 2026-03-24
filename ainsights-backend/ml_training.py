import logging
import os
import json as _json
from typing import Dict, Any, Tuple, List
from datetime import datetime
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# Identifier columns to exclude from features (drop if present, before X/y split)
ID_COLUMNS = frozenset({
    "id", "employee_id", "employee_code", "snapshot_id", "user_id",
    "name", "email", "created_at", "updated_at",
})


def _infer_feature_types(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """
    Infer numeric and categorical feature columns from a dataframe.

    Target column should already be removed from df.
    """
    # Categorical: object, category, bool
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Numeric: anything else that is not in categorical
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def _make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numeric features
    - One-hot encodes categorical features
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", categorical_transformer),
        ]
    )

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def _build_models(
    preprocessor: ColumnTransformer,
) -> Dict[str, Pipeline]:
    """
    Create sklearn Pipelines for the candidate models.
    Each pipeline gets a clone of the preprocessor so CV fits do not conflict.
    """
    log_reg = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return {"logistic_regression": log_reg, "random_forest": rf}


def _evaluate_model_cv(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a model using StratifiedKFold cross-validation.
    Returns mean ROC-AUC and mean F1 (and per-fold metrics if useful).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["roc_auc", "f1"]
    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )
    mean_roc_auc = float(np.mean(scores["test_roc_auc"]))
    mean_f1 = float(np.mean(scores["test_f1"]))
    return {
        "model": name,
        "mean_roc_auc": mean_roc_auc,
        "mean_f1": mean_f1,
        "cv_roc_auc_scores": scores["test_roc_auc"].tolist(),
        "cv_f1_scores": scores["test_f1"].tolist(),
    }


def _log_model_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    selected_name: str,
    best_roc_auc: float,
) -> None:
    """Log and print mean ROC-AUC and mean F1 per model, and selected model (presentation-ready)."""
    lines = [
        "",
        "========== Model comparison (StratifiedKFold CV) ==========",
    ]
    for name, m in all_metrics.items():
        roc = m.get("mean_roc_auc", None)
        f1 = m.get("mean_f1", None)
        mark = "  <-- SELECTED" if name == selected_name else ""
        lines.append(f"  {name}:  mean ROC-AUC = {roc:.4f},  mean F1 = {f1:.4f}{mark}")
    lines.append(f"Selected model: {selected_name} (best mean ROC-AUC = {best_roc_auc:.4f})")
    lines.append("=" * 52)
    block = "\n".join(lines)
    logger.info(block)
    print(block)


def _save_feature_importance(
    pipeline: Pipeline,
    model_name: str,
    output_path: str,
) -> None:
    """
    Extract top 10 feature importance (RandomForest) or absolute coefficients (LogisticRegression).
    Save to feature_importance.json as [{"feature": "...", "importance": 0.123}, ...].
    """
    preprocess = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocess is None or model is None:
        logger.warning("Pipeline missing 'preprocess' or 'model' step; skipping feature importance.")
        return
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception as e:
        logger.warning("Could not get feature names from preprocess: %s", e)
        return

    if model_name == "random_forest" and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = list(zip(feature_names, imp))
        pairs.sort(key=lambda x: x[1], reverse=True)
    elif model_name == "logistic_regression" and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).flatten()
        if coef.shape[0] != len(feature_names):
            # multiclass: coef has shape (n_classes, n_features)
            coef = np.abs(model.coef_).max(axis=0)
        else:
            coef = np.abs(coef)
        pairs = list(zip(feature_names, coef.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
    else:
        logger.warning("Unknown model or no feature importance / coef_; skipping.")
        return

    top10 = [{"feature": str(f), "importance": float(v)} for f, v in pairs[:10]]
    with open(output_path, "w", encoding="utf-8") as fh:
        _json.dump(top10, fh, indent=2)
    logger.info("Wrote top 10 feature importance to %s", output_path)


def _log_class_imbalance(y: pd.Series, target_column: str) -> None:
    """Print and log class counts and percentage split (positive vs negative)."""
    counts = y.value_counts()
    if len(counts) < 2:
        msg = (
            f"Target '{target_column}' has only one class: {counts.index.tolist()}. "
            "Stratified CV requires at least two classes."
        )
        logger.warning(msg)
        print(msg)
        return
    # Convention: assume last/sorted class is "positive" (e.g. 1 or Yes)
    classes_sorted = counts.index.tolist()
    try:
        classes_sorted.sort(key=lambda x: (isinstance(x, str), x))
    except Exception:
        pass
    neg_label, pos_label = classes_sorted[0], classes_sorted[-1]
    neg_count = int(counts.get(neg_label, 0))
    pos_count = int(counts.get(pos_label, 0))
    n = neg_count + pos_count
    neg_pct = 100.0 * neg_count / n if n else 0
    pos_pct = 100.0 * pos_count / n if n else 0
    msg = (
        f"Class imbalance — negative ({neg_label}): {neg_count} ({neg_pct:.2f}%); "
        f"positive ({pos_label}): {pos_count} ({pos_pct:.2f}%)"
    )
    logger.info(msg)
    print(msg)


def train_turnover_model(
    csv_path: str,
    target_column: str,
    model_output_path: str | None = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate attrition models on a CSV dataset.

    - Loads data from `csv_path`
    - Drops rows with missing target
    - Drops identifier columns if present: id, employee_id, employee_code, snapshot_id,
      user_id, name, email, created_at, updated_at
    - Infers categorical/numeric feature columns (no hard-coded feature names)
    - Builds full sklearn Pipelines (preprocess + model)
    - Evaluates with StratifiedKFold (5 folds): mean ROC-AUC, mean F1
    - Selects best model by mean ROC-AUC, fits on full data, saves as turnover_model.pkl only
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    y = df[target_column]

    # Drop identifier columns if present (before feature/target split)
    id_present = [c for c in ID_COLUMNS if c in df.columns]
    cols_to_drop = [target_column] + id_present
    X = df.drop(columns=cols_to_drop)

    if X.empty:
        raise ValueError(
            "No feature columns remaining after dropping target and identifier columns."
        )

    # Class imbalance: print and log
    _log_class_imbalance(y, target_column)

    # Infer feature types
    numeric_cols, categorical_cols = _infer_feature_types(X)
    if not numeric_cols and not categorical_cols:
        raise ValueError("Could not infer any feature columns from dataset.")

    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    models = _build_models(preprocessor)

    all_metrics: Dict[str, Dict[str, Any]] = {}
    best_model_name: str | None = None
    best_model: Pipeline | None = None
    best_mean_roc_auc: float = -np.inf

    for name, model in models.items():
        metrics = _evaluate_model_cv(
            name, model, X, y, n_splits=n_folds, random_state=random_state
        )
        all_metrics[name] = metrics
        mean_roc_auc = metrics["mean_roc_auc"]
        if mean_roc_auc > best_mean_roc_auc:
            best_mean_roc_auc = mean_roc_auc
            best_model_name = name
            best_model = model

    if best_model is None or best_model_name is None:
        raise RuntimeError("Model training failed; no best model determined.")

    # ----- Model comparison: log and print (presentation-ready)
    _log_model_comparison(all_metrics, best_model_name, best_mean_roc_auc)

    # ----- Confusion matrix + classification report (OOF predictions, before final fit)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_pred_oof = cross_val_predict(best_model, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred_oof)
    cm_list: List[List[int]] = cm.tolist()
    report_str = classification_report(y, y_pred_oof)
    report_dict: Dict[str, Any] = classification_report(
        y, y_pred_oof, output_dict=True, zero_division=0
    )
    # Log and print
    logger.info("Confusion matrix (OOF predictions):\n%s", cm)
    logger.info("Classification report (OOF predictions):\n%s", report_str)
    print("\n--- Confusion matrix (out-of-fold) ---")
    print(cm)
    print("\n--- Classification report (out-of-fold) ---")
    print(report_str)

    # Refit best model on full data and save single artifact (preprocess + model)
    best_model.fit(X, y)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_output_path is None:
        model_output_path = os.path.join(base_dir, "turnover_model.pkl")

    joblib.dump(best_model, model_output_path)

    # ----- Global feature importance (top 10) -> feature_importance.json
    importance_path = os.path.join(base_dir, "feature_importance.json")
    _save_feature_importance(best_model, best_model_name, importance_path)

    # ----- Lightweight training metadata for reproducibility
    metadata: Dict[str, Any] = {
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "scikit_learn": sklearn_version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__ if hasattr(joblib, "__version__") else None,
        },
        "training": {
            "csv_path": os.path.abspath(csv_path),
            "target_column": target_column,
            "n_folds": n_folds,
            "random_state": random_state,
        },
        "model": {
            "selected_model": best_model_name,
            "model_path": os.path.abspath(model_output_path),
            "feature_importance_path": os.path.abspath(importance_path),
            "feature_columns": X.columns.tolist(),
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
        },
        "metrics": {
            "best_mean_roc_auc": best_mean_roc_auc,
            "all_models": all_metrics,
            "confusion_matrix": cm_list,
            "classification_report": report_dict,
        },
    }

    metadata_path = os.path.join(base_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        _json.dump(metadata, fh, indent=2, default=str)
    logger.info("Wrote training metadata to %s", metadata_path)

    return metadata


if __name__ == "__main__":
    """
    Simple CLI entrypoint to allow running training from the command line, e.g.:

    python ml_training.py --csv-path ./data/employee_attrition.csv --target-column attrition
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train employee turnover prediction models.")
    parser.add_argument("--csv-path", required=True, help="Path to training CSV file.")
    parser.add_argument(
        "--target-column",
        required=True,
        help="Name of the target column in the CSV (e.g. 'attrition').",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Where to save the trained model pipeline (defaults to turnover_model.pkl in backend directory).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits (default: 5).",
    )

    args = parser.parse_args()
    result = train_turnover_model(
        csv_path=args.csv_path,
        target_column=args.target_column,
        model_output_path=args.model_path,
        n_folds=args.n_folds,
    )
    print(json.dumps(result, indent=2, default=str))

