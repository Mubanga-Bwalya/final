import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

import models
import schemas
from security import get_current_user
import ml_service

router = APIRouter(prefix="/api/v1/model", tags=["model"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.json")
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")


@router.get("/feature-importance")
def get_feature_importance(
    current_user: models.User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Return top feature importance from training (feature_importance.json).
    Requires authentication. Does not retrain.
    """
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance file not found. Train the model first.",
        )
    try:
        with open(FEATURE_IMPORTANCE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read or parse feature_importance.json: {e!s}",
        )
    return data


@router.get("/status")
def get_model_status(
    current_user: models.User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Return current model/feature-importance status for the frontend.
    Requires authentication.
    """
    model_loaded = ml_service.TURNOVER_MODEL is not None
    model_path = ml_service.MODEL_PATH
    feature_importance_available = os.path.exists(FEATURE_IMPORTANCE_PATH)

    return {
        "model_loaded": model_loaded,
        "model_path": model_path,
        "feature_importance_available": feature_importance_available,
    }


@router.get("/summary", response_model=schemas.ModelSummary)
def get_model_summary(
    current_user: models.User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Return a compact summary of the currently trained attrition model.

    Intended for academic inspection and transparency rather than live scoring.
    Includes:
    - selected model name
    - training timestamp
    - library versions
    - feature count and sample feature names
    - top feature importance (up to 10)
    - key evaluation metrics (best mean ROC-AUC)
    """
    if not os.path.exists(MODEL_METADATA_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata not found. Train the model to generate metadata.",
        )

    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read or parse model_metadata.json: {exc!s}",
        )

    # Extract core fields defensively in case metadata shape changes
    training_info: Dict[str, Any] = metadata.get("training", {})
    model_info: Dict[str, Any] = metadata.get("model", {})
    metrics_info: Dict[str, Any] = metadata.get("metrics", {})
    libraries_info: Dict[str, Any] = metadata.get("libraries", {})

    feature_columns: List[str] = model_info.get("feature_columns", []) or []
    numeric_features: List[str] = model_info.get("numeric_features", []) or []
    categorical_features: List[str] = model_info.get("categorical_features", []) or []

    # Load feature importance if available
    feature_importance: Optional[List[Dict[str, Any]]] = None
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        try:
            with open(FEATURE_IMPORTANCE_PATH, "r", encoding="utf-8") as fh:
                feature_importance = json.load(fh)
        except Exception:
            # Keep summary usable even if importance file is malformed
            feature_importance = None

    summary: Dict[str, Any] = {
        "trained_at_utc": metadata.get("trained_at_utc"),
        "training": {
            "csv_path": training_info.get("csv_path"),
            "target_column": training_info.get("target_column"),
            "n_folds": training_info.get("n_folds"),
            "random_state": training_info.get("random_state"),
        },
        "libraries": libraries_info,
        "model": {
            "selected_model": model_info.get("selected_model"),
            "feature_count": len(feature_columns),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "sample_features": feature_columns[:10],
        },
        "metrics": {
            "best_mean_roc_auc": metrics_info.get("best_mean_roc_auc"),
        },
        "limitations": metadata.get("limitations", []),
    }

    if feature_importance is not None:
        # Only include a compact top-k list in the summary
        summary["top_feature_importance"] = feature_importance[:10]

    return summary
