import logging
import os
import json
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

import models

logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "turnover_model.pkl")

MODEL_PATH = os.getenv("TURNOVER_MODEL_PATH", DEFAULT_MODEL_PATH)

# Minimum fraction of model input features that should be populated
# before we consider predictions academically reliable.
MIN_FEATURE_COVERAGE_RATIO = float(os.getenv("AINSIGHTS_MIN_FEATURE_COVERAGE", "0.6"))


def _load_artifact(path: str, artifact_name: str):
    if not os.path.exists(path):
        logger.warning(
            "%s file not found at %s. Prediction endpoints will return errors until the "
            "model is trained. Run ml_training.py to train and save the model.",
            artifact_name,
            path,
        )
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error("Failed to load %s from %s: %s", artifact_name, path, exc)
        return None


TURNOVER_MODEL = _load_artifact(MODEL_PATH, "turnover model")

# Optional: load accompanying metadata for inspection / logging
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")
MODEL_METADATA: Dict[str, Any] | None = None
if os.path.exists(MODEL_METADATA_PATH):
    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            MODEL_METADATA = json.load(fh)
    except Exception as exc:
        logger.warning("Could not load model metadata from %s: %s", MODEL_METADATA_PATH, exc)


def _get_feature_columns_from_model() -> List[str] | None:
    """
    Extract expected feature columns from the full pipeline's preprocess step.
    """
    if TURNOVER_MODEL is None:
        return None
    preprocess = getattr(TURNOVER_MODEL, "named_steps", {}).get("preprocess")
    if preprocess is None:
        logger.error("Model has no 'preprocess' step; cannot get feature names.")
        return None
    feature_names = getattr(preprocess, "feature_names_in_", None)
    if feature_names is None:
        logger.error(
            "Preprocess step does not expose feature_names_in_. "
            "Cannot align employee features with training features."
        )
        return None
    return list(feature_names)


def get_model_feature_columns() -> list[str]:
    """
    Public helper to expose the model's expected input feature columns.

    Returns an empty list if the model or its preprocess step is unavailable.
    """
    cols = _get_feature_columns_from_model()
    if not cols:
        return []
    return list(cols)


def build_feature_row_for_employee(emp: models.Employee, feature_columns: List[str]) -> Dict[str, Any]:
    """
    Build one raw feature dict aligned to model training columns (Employee attrs + extra_features).
    Shared by batch prediction and single-employee scenario simulation.
    """
    row: Dict[str, Any] = {}
    for col in feature_columns:
        if hasattr(emp, col):
            row[col] = getattr(emp, col)
        elif emp.extra_features and isinstance(emp.extra_features, dict) and col in emp.extra_features:
            row[col] = emp.extra_features.get(col)
        else:
            row.setdefault(col, None)
    return row


def _build_employee_features_and_coverage(
    employees: List[models.Employee],
    feature_columns: List[str],
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Build a DataFrame of employee features matching the training feature columns
    and, in parallel, compute per-employee feature coverage metrics.
    """
    rows: List[Dict[str, Any]] = []
    coverage_info: List[Dict[str, Any]] = []

    total_features = len(feature_columns)

    for emp in employees:
        row = build_feature_row_for_employee(emp, feature_columns)

        # Compute coverage metrics for this employee
        populated_features = [c for c in feature_columns if row.get(c) is not None]
        missing_features = [c for c in feature_columns if row.get(c) is None]
        populated_count = len(populated_features)
        missing_count = len(missing_features)
        coverage_ratio = float(populated_count / total_features) if total_features else 0.0

        coverage_info.append(
            {
                "employee_id": emp.id,
                "employee_code": getattr(emp, "employee_code", None),
                "employee_name": getattr(emp, "name", None),
                "total_features": total_features,
                "populated_count": populated_count,
                "missing_count": missing_count,
                "populated_features": populated_features,
                "missing_features": missing_features,
                "coverage_ratio": coverage_ratio,
            }
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        logger.error(
            "Employee features mismatch training features. Missing columns in frame: %s",
            ", ".join(missing_cols),
        )
        raise ValueError(
            "Employee features do not match training features. "
            f"Missing columns: {', '.join(missing_cols)}. "
            "Please align your employee/extra_features schema with the training dataset."
        )

    # Ensure column order matches training
    df = df[feature_columns]
    return df, coverage_info


def compute_feature_coverage_for_employees(
    employees: List[models.Employee],
    feature_columns: List[str],
) -> List[Dict[str, Any]]:
    """
    Public helper to compute feature coverage metrics for a list of employees.

    Does not run the model; only inspects which expected input features are
    populated vs missing based on current Employee attributes and extra_features.
    """
    _, coverage_info = _build_employee_features_and_coverage(employees, feature_columns)
    return coverage_info


def _build_employee_feature_frame(
    employees: List[models.Employee],
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Build a DataFrame of employee features matching the training feature columns.

    Uses direct Employee attributes only; if any expected feature is missing,
    logs an error and raises a ValueError so the caller can respond appropriately.
    """
    df, _coverage_info = _build_employee_features_and_coverage(employees, feature_columns)
    return df


def _get_positive_class_index() -> int:
    """
    Determine which probability column corresponds to 'attrition risk'.
    Prefers class label '1' if present; otherwise uses the last column.
    """
    model_step = None
    if hasattr(TURNOVER_MODEL, "named_steps"):
        model_step = TURNOVER_MODEL.named_steps.get("model")

    if model_step is None or not hasattr(model_step, "classes_"):
        return -1

    classes = list(model_step.classes_)
    if 1 in classes:
        return classes.index(1)

    # Fallback: use last column as "positive" class
    return len(classes) - 1


def _risk_level_from_probability(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


def risk_level_from_probability(prob: float) -> str:
    """Public alias for API layers that need consistent risk labels."""
    return _risk_level_from_probability(prob)


def coverage_ratio_for_row(row: Dict[str, Any], feature_columns: List[str]) -> float:
    if not feature_columns:
        return 0.0
    populated = sum(1 for c in feature_columns if row.get(c) is not None)
    return float(populated / len(feature_columns))


def apply_attrition_simulation_overrides(
    base_row: Dict[str, Any],
    feature_columns: List[str],
    *,
    salary: float | None = None,
    tenure_months: int | None = None,
    job_role: str | None = None,
    department_name: str | None = None,
    performance_rating: int | None = None,
) -> Dict[str, Any]:
    """
    Copy base_row and apply only overrides for model columns that exist in the trained schema.
    Mirrors upload-time mapping: salary→MonthlyIncome, tenure_months→YearsAtCompany (years), etc.
    """
    out = dict(base_row)
    colset = set(feature_columns)
    if salary is not None and "MonthlyIncome" in colset:
        out["MonthlyIncome"] = float(salary)
    if tenure_months is not None and "YearsAtCompany" in colset:
        out["YearsAtCompany"] = float(tenure_months) / 12.0
    if job_role is not None and "JobRole" in colset:
        out["JobRole"] = job_role
    if department_name is not None and "Department" in colset:
        out["Department"] = department_name
    if performance_rating is not None and "PerformanceRating" in colset:
        out["PerformanceRating"] = int(performance_rating)
    return out


def predict_probability_for_rows(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Run predict_proba on a raw feature frame aligned to training columns.
    Returns the positive-class probability column as a 1d array.
    """
    if TURNOVER_MODEL is None:
        raise RuntimeError(
            "Attrition model is not available. Train the model with ml_training.py "
            "and ensure turnover_model.pkl is present before requesting predictions."
        )
    if not hasattr(TURNOVER_MODEL, "predict_proba"):
        raise RuntimeError(
            "Loaded attrition model does not support predict_proba. "
            "Use a classifier with predict_proba when training the model."
        )
    positive_index = _get_positive_class_index()
    if positive_index < 0:
        raise RuntimeError(
            "Could not map model probabilities to the positive attrition class. "
            "Retrain the model and ensure the classifier exposes a valid classes_ attribute."
        )
    proba = TURNOVER_MODEL.predict_proba(df_raw)
    if positive_index >= proba.shape[1]:
        raise RuntimeError(
            "Could not map model probabilities to the positive attrition class. "
            "Retrain the model and ensure the classifier exposes a valid classes_ attribute."
        )
    return proba[:, positive_index]


def predict_probability_single_row(
    row: Dict[str, Any],
    feature_columns: List[str],
) -> float:
    """Score one employee feature row; returns P(positive class)."""
    df_raw = pd.DataFrame([row])[feature_columns]
    probs = predict_probability_for_rows(df_raw)
    return float(probs[0])


def predict_snapshot(snapshot_id: int, db_session: Session) -> List[Dict[str, Any]]:
    """
    Predict attrition risk for all employees in a given snapshot.

    - Snapshot-scoped: only uses employees linked to the provided snapshot_id
    - Multi-tenant safe: snapshot is resolved from the database, and only its employees are used
    - Uses the full saved pipeline: predict_proba(raw_dataframe) applies preprocess + model internally
    - Returns list of { employee_id, probability, risk_level }
    """
    if TURNOVER_MODEL is None:
        raise RuntimeError(
            "Attrition model is not available. Train the model with ml_training.py "
            "and ensure turnover_model.pkl is present before requesting predictions."
        )

    # Confirm snapshot exists (and implicitly its tenant)
    snapshot = db_session.query(models.Snapshot).filter(
        models.Snapshot.id == snapshot_id
    ).first()
    if snapshot is None:
        raise ValueError(f"Snapshot with id {snapshot_id} not found.")

    employees = (
        db_session.query(models.Employee)
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == snapshot.user_id,
        )
        .all()
    )

    if not employees:
        return []

    feature_columns = _get_feature_columns_from_model()
    if not feature_columns:
        raise RuntimeError(
            "Attrition model preprocess step does not expose input feature names. "
            "Retrain the model with ml_training.py and ensure feature_names_in_ is preserved."
        )

    # Build raw feature frame + per-employee coverage metrics aligned with training columns
    df_raw, coverage_info = _build_employee_features_and_coverage(employees, feature_columns)

    if not hasattr(TURNOVER_MODEL, "predict_proba"):
        raise RuntimeError(
            "Loaded attrition model does not support predict_proba. "
            "Use a classifier with predict_proba when training the model."
        )

    try:
        proba = TURNOVER_MODEL.predict_proba(df_raw)
    except Exception as exc:
        logger.error(
            "Error running model.predict_proba on snapshot %s: %s",
            snapshot_id,
            exc,
        )
        raise
    positive_index = _get_positive_class_index()

    if positive_index < 0 or positive_index >= proba.shape[1]:
        logger.error(
            "Could not determine positive class index for probability output. "
            "Model classes_ may be unavailable or mismatched."
        )
        raise RuntimeError(
            "Could not map model probabilities to the positive attrition class. "
            "Retrain the model and ensure the classifier exposes a valid classes_ attribute."
        )

    positive_probs = proba[:, positive_index]

    # Safety: if all employees are below the minimum coverage threshold,
    # abort prediction with a clear, academic-friendly error.
    low_coverage_flags = [
        c["coverage_ratio"] < MIN_FEATURE_COVERAGE_RATIO for c in coverage_info
    ]
    if low_coverage_flags and all(low_coverage_flags):
        avg_coverage = float(
            sum(c["coverage_ratio"] for c in coverage_info) / len(coverage_info)
        )
        raise ValueError(
            "Employee feature coverage is too low for this snapshot. "
            f"Model expects {len(feature_columns)} input features, but the average "
            f"populated fraction is only {avg_coverage:.2f}. "
            "Please upload data that includes more of the training schema columns "
            "(for example: Age, JobRole, MonthlyIncome, Department, YearsAtCompany)."
        )

    results: List[Dict[str, Any]] = []
    for emp, p, cov in zip(employees, positive_probs, coverage_info):
        probability = float(p)
        results.append(
            {
                "employee_id": emp.id,
                "probability": probability,
                "risk_level": _risk_level_from_probability(probability),
                "feature_coverage": cov["coverage_ratio"],
                "populated_feature_count": cov["populated_count"],
                "missing_feature_count": cov["missing_count"],
                # Keep the missing_features list short so responses stay practical
                "missing_features_sample": cov["missing_features"][:10],
                "low_coverage": cov["coverage_ratio"] < MIN_FEATURE_COVERAGE_RATIO,
            }
        )

    return results

