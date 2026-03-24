from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
import pandas as pd
import re
import os
import json
import logging
from datetime import datetime

from database import get_db
import models
import schemas
from security import get_current_user
from ml_service import get_model_feature_columns

router = APIRouter(prefix="/api/v1/uploads", tags=["uploads"])

logger = logging.getLogger("ainsights.upload")


def _debug_enabled() -> bool:
    return os.getenv("AINSIGHTS_UPLOAD_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _df_head_records(df: pd.DataFrame, n: int = 3):
    # Convert NaN to None so the logs are readable JSON
    return df.head(n).where(pd.notna(df.head(n)), None).to_dict(orient="records")


def _log_debug(payload: dict):
    # Structured log line (single JSON object)
    try:
        logger.warning("[UPLOAD_DEBUG] %s", json.dumps(payload, default=str))
    except Exception:
        logger.warning("[UPLOAD_DEBUG] %r", payload)


def safe_int(value) -> int | None:
    """
    Safely convert value to integer.
    Returns None on failure, never raises exceptions.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return int(float(stripped))  # Handle "123.0" strings
        return None
    except (ValueError, TypeError, AttributeError):
        return None


def safe_float(value) -> float | None:
    """
    Safely convert value to float.
    Returns None on failure, never raises exceptions.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return float(stripped)
        return None
    except (ValueError, TypeError, AttributeError):
        return None


def normalize_department_name(name: str) -> str:
    """
    Normalize department name:
    - Strip whitespace
    - Collapse multiple spaces
    - Convert to title case
    
    Example: " engineering " -> "Engineering"
    """
    if not name:
        return ""
    # Strip whitespace
    normalized = name.strip()
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    # Convert to title case
    normalized = normalized.title()
    return normalized


def normalize_columns(columns):
    """Normalize CSV column names to match database field names."""
    mapping = {
        # Employee code variants
        "employeenumber": "employee_code",
        "employee_id": "employee_code",
        "employeecode": "employee_code",
        "emp_id": "employee_code",
        "empcode": "employee_code",
        
        # Name variants
        "employeename": "name",
        "fullname": "name",
        "full_name": "name",
        "employee_name": "name",
        
        # Salary variants
        "monthlyincome": "salary",
        "monthly_salary": "salary",
        "payrate": "salary",
        "pay rate": "salary",
        "pay_rate": "salary",
        "monthly_income": "salary",
        
        # Tenure variants
        "yearsatcompany": "tenure_years",
        "years_at_company": "tenure_years",
        "tenure": "tenure_years",
        
        # Performance score variants
        "performancerating": "performance_score",
        "performance rating": "performance_score",
        "performance_rating": "performance_score",
        "rating": "performance_score",
        
        # Department
        "department": "department",
        "dept": "department",
        
        # Role variants
        "jobrole": "role",
        "job_title": "role",
        "jobtitle": "role",
        "position": "role",
        "job_role": "role",
        
        # Attrition
        "attrition": "attrition",
        
        # Review period variants
        "review_period": "review_period",
        "reviewperiod": "review_period",
        "period": "review_period",
        
        # Review date variants (for deriving period)
        "review_date": "review_date",
        "reviewdate": "review_date",
        "date": "review_date",
    }
    
    normalized = {}
    for col in columns:
        key = col.strip().lower().replace(" ", "").replace("_", "")
        normalized[col] = mapping.get(key, col)
    
    return normalized


def derive_review_period(row, df_columns):
    """Derive review_period from CSV row data."""
    # Check if review_period column exists and has value
    if "review_period" in df_columns:
        try:
            val = row.get("review_period")
            if pd.notna(val):
                return str(val).strip()
        except (KeyError, AttributeError):
            pass
    
    # Check if review_date exists and derive period
    if "review_date" in df_columns:
        try:
            date_val = row.get("review_date")
            if pd.notna(date_val):
                # Handle string dates
                date_obj = pd.to_datetime(date_val, errors="coerce")
                
                if pd.notna(date_obj):
                    year = date_obj.year
                    quarter = (date_obj.month - 1) // 3 + 1
                    return f"{year}-Q{quarter}"
        except (KeyError, AttributeError, Exception):
            pass
    
    # Default fallback
    return "Unknown"


@router.post("/", response_model=schemas.UploadResponse)
def upload_hr_data(
    file: UploadFile = File(...),
    snapshot_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Upload and process HR data CSV file.
    
    Required:
    - snapshot_id: ID of the snapshot to associate this upload with
    - CSV columns: employee_code, department
    
    Optional columns: name, email, role, salary, tenure_years, performance_score, review_period, review_date
    
    Note: Each upload creates new employee records for the snapshot. Employees are not updated across snapshots.
    """
    
    # 0. Verify snapshot exists and belongs to current_user
    snapshot = (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.id == snapshot_id,
            models.Snapshot.user_id == current_user.id,
        )
        .first()
    )
    
    if not snapshot:
        raise HTTPException(
            status_code=404,
            detail="Snapshot not found or does not belong to current user"
        )
    
    # 1. Safely read CSV
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if _debug_enabled():
        _log_debug(
            {
                "stage": "raw_read_csv_before_normalization",
                "filename": file.filename,
                "snapshot_id": snapshot_id,
                "user_id": current_user.id,
                "columns": list(df.columns),
                "head3": _df_head_records(df, 3),
            }
        )
    
    # 2. Normalize columns
    raw_columns = list(df.columns)
    column_mapping = normalize_columns(df.columns)
    df.rename(columns=column_mapping, inplace=True)

    if _debug_enabled():
        _log_debug(
            {
                "stage": "after_column_normalization",
                "raw_columns": raw_columns,
                "column_mapping": column_mapping,
                "normalized_columns": list(df.columns),
                "head3": _df_head_records(df, 3),
            }
        )
    
    # 3. Validate required fields
    if "employee_code" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required column: employee_code")
    if "department" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required column: department")
    
    # 4. Standardize fields
    # Convert tenure_years to tenure_months if exists (will be validated later)
    if "tenure_years" in df.columns:
        df["tenure_months"] = None
        for idx in df.index:
            years_val = safe_float(df.loc[idx, "tenure_years"])
            if years_val is not None:
                df.loc[idx, "tenure_months"] = int(years_val * 12)
    
    # Convert attrition if exists
    if "attrition" in df.columns:
        df["attrition"] = df["attrition"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "Y": 1, "N": 0})
    
    # Handle name field - use employee_code as fallback if missing
    if "name" not in df.columns or df["name"].isna().all():
        df["name"] = df["employee_code"].astype(str)
    else:
        # Fill missing names with employee_code
        df["name"] = df["name"].fillna(df["employee_code"]).astype(str)

    if _debug_enabled():
        _log_debug(
            {
                "stage": "after_field_mapping_and_standardization",
                "columns": list(df.columns),
                "head3": _df_head_records(df, 3),
            }
        )
    
    # 5. Create Upload Record
    upload = models.DataUpload(
        user_id=current_user.id,
        filename=file.filename or "unknown.csv",
        created_at=datetime.utcnow()
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    
    # 6. Process Each Row
    inserted_count = 0
    failed_count = 0
    processed_employee_codes = set()  # Track employee codes in this snapshot for duplicate detection
    debug_failed_rows = []  # first 5 failures with reasons (debug only)

    # Resolve model feature columns once per upload (empty set if model unavailable)
    try:
        model_feature_columns = set(get_model_feature_columns())
    except Exception:
        model_feature_columns = set()

    def _json_safe_value(value):
        """
        Convert a pandas / numpy / Python scalar to a JSON-serializable value.
        - Preserves ints, floats, bools, and strings as-is
        - Treats pandas NA / NaN as missing (returns None)
        - Fallback: string representation
        """
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (int, float, bool, str)):
            return value
        scalar = getattr(value, "item", None)
        if callable(scalar):
            try:
                return scalar()
            except Exception:
                pass
        return str(value)

    def _add_model_feature_if_applicable(
        extra_features: dict,
        feature_name: str,
        raw_value,
    ):
        """
        Add a value to extra_features under the exact model feature name if:
        - The loaded model expects this feature, and
        - A non-null value is provided, and
        - The key is not already present (do not overwrite).
        """
        if feature_name not in model_feature_columns:
            return
        if feature_name in extra_features:
            return
        value = _json_safe_value(raw_value)
        if value is None:
            return
        extra_features[feature_name] = value

    def _record_fail(*, row_index: int, reason: str, context: dict):
        if not _debug_enabled():
            return
        if len(debug_failed_rows) >= 5:
            return
        payload = {
            "row_index": int(row_index) if isinstance(row_index, (int, float)) else str(row_index),
            "reason": reason,
            **context,
        }
        debug_failed_rows.append(payload)
        _log_debug({"stage": "row_validation_failed", **payload})
    
    for index, row in df.iterrows():
        try:
            # VALIDATION: Required Fields
            # Extract employee_code
            employee_code_raw = row.get("employee_code")
            if pd.isna(employee_code_raw):
                _record_fail(
                    row_index=index,
                    reason="missing_employee_code",
                    context={
                        "employee_code_raw": None,
                        "department_raw": (None if "department" not in df.columns else (None if pd.isna(row.get("department")) else str(row.get("department")))),
                        "normalized_columns": list(df.columns),
                    },
                )
                failed_count += 1
                continue
            
            employee_code = str(employee_code_raw).strip()
            if not employee_code:
                _record_fail(
                    row_index=index,
                    reason="empty_employee_code",
                    context={
                        "employee_code_raw": str(employee_code_raw),
                        "department_raw": (None if "department" not in df.columns else (None if pd.isna(row.get("department")) else str(row.get("department")))),
                    },
                )
                failed_count += 1
                continue
            
            # VALIDATION: Duplicate Employee Detection (within same snapshot - check in-process first)
            if employee_code in processed_employee_codes:
                _record_fail(
                    row_index=index,
                    reason="duplicate_employee_code_in_batch",
                    context={"employee_code": employee_code},
                )
                failed_count += 1
                continue
            
            # Check database for existing employee_code in this snapshot
            existing_employee = (
                db.query(models.Employee)
                .filter(
                    models.Employee.user_id == current_user.id,
                    models.Employee.employee_code == employee_code,
                    models.Employee.snapshot_id == snapshot_id,
                )
                .first()
            )
            
            if existing_employee:
                _record_fail(
                    row_index=index,
                    reason="duplicate_employee_code_in_snapshot_db",
                    context={"employee_code": employee_code},
                )
                failed_count += 1
                continue
            
            # Extract department
            department_raw = row.get("department")
            if pd.isna(department_raw):
                _record_fail(
                    row_index=index,
                    reason="missing_department",
                    context={"employee_code": employee_code, "department_raw": None},
                )
                failed_count += 1
                continue
            
            department_name_raw = str(department_raw).strip()
            if not department_name_raw:
                _record_fail(
                    row_index=index,
                    reason="empty_department",
                    context={"employee_code": employee_code, "department_raw": str(department_raw)},
                )
                failed_count += 1
                continue
            
            # Normalize department name
            department_name = normalize_department_name(department_name_raw)
            if not department_name:
                _record_fail(
                    row_index=index,
                    reason="normalized_department_empty",
                    context={"employee_code": employee_code, "department_raw": department_name_raw},
                )
                failed_count += 1
                continue
            
            # VALIDATION: Salary (must be >= 0 if provided)
            salary_value = None
            if "salary" in df.columns and pd.notna(row.get("salary")):
                salary_raw = safe_float(row.get("salary"))
                if salary_raw is None:
                    _record_fail(
                        row_index=index,
                        reason="salary_parse_failed",
                        context={"employee_code": employee_code, "salary_raw": str(row.get("salary"))},
                    )
                    failed_count += 1
                    continue
                if salary_raw < 0:
                    _record_fail(
                        row_index=index,
                        reason="salary_negative",
                        context={"employee_code": employee_code, "salary_raw": salary_raw},
                    )
                    failed_count += 1
                    continue
                salary_value = salary_raw
            
            # VALIDATION: Tenure (must be >= 0 if provided)
            tenure_months_value = None
            if "tenure_months" in df.columns and pd.notna(row.get("tenure_months")):
                tenure_raw = safe_int(row.get("tenure_months"))
                if tenure_raw is None:
                    # Try to convert from float if needed
                    tenure_raw = safe_float(row.get("tenure_months"))
                    if tenure_raw is not None:
                        tenure_raw = int(tenure_raw)
                    else:
                        _record_fail(
                            row_index=index,
                            reason="tenure_parse_failed",
                            context={"employee_code": employee_code, "tenure_raw": str(row.get("tenure_months"))},
                        )
                        failed_count += 1
                        continue
                if tenure_raw < 0:
                    _record_fail(
                        row_index=index,
                        reason="tenure_negative",
                        context={"employee_code": employee_code, "tenure_raw": tenure_raw},
                    )
                    failed_count += 1
                    continue
                tenure_months_value = tenure_raw
            
            # VALIDATION: Performance Score (must be integer between 1-5 if provided)
            performance_score_value = None
            if "performance_score" in df.columns and pd.notna(row.get("performance_score")):
                score_raw = safe_int(row.get("performance_score"))
                if score_raw is None:
                    _record_fail(
                        row_index=index,
                        reason="performance_score_parse_failed",
                        context={"employee_code": employee_code, "performance_score_raw": str(row.get("performance_score"))},
                    )
                    failed_count += 1
                    continue
                if score_raw < 1 or score_raw > 5:
                    _record_fail(
                        row_index=index,
                        reason="performance_score_out_of_range",
                        context={"employee_code": employee_code, "performance_score_raw": score_raw},
                    )
                    failed_count += 1
                    continue
                performance_score_value = score_raw
            
            # A. Department - Check existing or create (using normalized name)
            department = (
                db.query(models.Department)
                .filter(
                    models.Department.user_id == current_user.id,
                    models.Department.name == department_name,
                )
                .first()
            )
            
            if not department:
                department = models.Department(
                    name=department_name,
                    user_id=current_user.id
                )
                db.add(department)
                # Flush so that department.id is available without committing
                db.flush()
            
            # B. Employee - Create new employee for this snapshot
            # (Duplicate check already done above)
            # Prepare employee data
            # Handle name - use employee_code as fallback
            name_value = employee_code
            if "name" in df.columns:
                name_val = row.get("name")
                if pd.notna(name_val):
                    name_value = str(name_val).strip()
                    if not name_value:
                        name_value = employee_code
            
            # Handle role - default to "Unknown" if missing
            role_value = "Unknown"
            if "role" in df.columns:
                role_val = row.get("role")
                if pd.notna(role_val):
                    role_str = str(role_val).strip()
                    if role_str:
                        role_value = role_str
            
            employee_data = {
                "employee_code": employee_code,
                "name": name_value,
                "role": role_value,
                "department_id": department.id,
                "user_id": current_user.id,
                "snapshot_id": snapshot_id,  # Required for new inserts
            }
            
            # Optional fields (already validated)
            if "email" in df.columns and pd.notna(row.get("email")):
                email_val = str(row["email"]).strip()
                if email_val:
                    employee_data["email"] = email_val
            
            if tenure_months_value is not None:
                employee_data["tenure_months"] = tenure_months_value
            
            if salary_value is not None:
                employee_data["salary"] = salary_value
            
            # Store extra fields in extra_features JSON
            extra_features = {}

            # Preserve numeric attrition label (not a model feature; target label)
            if "attrition" in df.columns and pd.notna(row.get("attrition")):
                attrition_val = safe_int(row.get("attrition"))
                if attrition_val is not None:
                    extra_features["attrition"] = attrition_val

            # Copy through any model feature columns that survived normalization
            # using their exact training names (e.g. Age, DistanceFromHome, etc.).
            # This only applies to columns whose names were not altered by normalize_columns.
            for feature_name in model_feature_columns:
                if feature_name in df.columns:
                    _add_model_feature_if_applicable(
                        extra_features,
                        feature_name,
                        row.get(feature_name),
                    )

            # Derived / mapped IBM-style features from app-native fields
            # - role            -> JobRole
            # - department_name -> Department
            # - salary_value    -> MonthlyIncome
            # - tenure_months   -> YearsAtCompany (in years, float)
            # - performance_score_value -> PerformanceRating
            _add_model_feature_if_applicable(extra_features, "JobRole", role_value)
            _add_model_feature_if_applicable(extra_features, "Department", department_name)
            if salary_value is not None:
                _add_model_feature_if_applicable(extra_features, "MonthlyIncome", salary_value)
            if tenure_months_value is not None:
                years_at_company = tenure_months_value / 12.0
                _add_model_feature_if_applicable(extra_features, "YearsAtCompany", years_at_company)
            if performance_score_value is not None:
                _add_model_feature_if_applicable(
                    extra_features,
                    "PerformanceRating",
                    performance_score_value,
                )

            if extra_features:
                employee_data["extra_features"] = extra_features
            
            # Create new employee for this snapshot
            employee = models.Employee(**employee_data)
            db.add(employee)
            db.commit()
            db.refresh(employee)
            
            # Track this employee_code to prevent duplicates in same batch
            processed_employee_codes.add(employee_code)
            
            # C. Performance Record - Insert if performance_score exists (already validated)
            if performance_score_value is not None:
                try:
                    review_period = derive_review_period(row, df.columns)
                    
                    # Check if performance record already exists for this employee, period, and snapshot
                    existing_record = (
                        db.query(models.PerformanceRecord)
                        .filter(
                            models.PerformanceRecord.user_id == current_user.id,
                            models.PerformanceRecord.employee_id == employee.id,
                            models.PerformanceRecord.review_period == review_period,
                            models.PerformanceRecord.snapshot_id == snapshot_id,
                        )
                        .first()
                    )
                    
                    if not existing_record:
                        performance_record = models.PerformanceRecord(
                            user_id=current_user.id,
                            employee_id=employee.id,
                            snapshot_id=snapshot_id,  # Set snapshot_id
                            score=performance_score_value,
                            review_period=review_period,
                        )
                        db.add(performance_record)
                        db.commit()
                except Exception as e:
                    # Log error but don't fail the entire row
                    pass
            
            inserted_count += 1
            
        except Exception as e:
            # Row processing failed - increment failed count and continue
            failed_count += 1
            if _debug_enabled() and len(debug_failed_rows) < 5:
                _log_debug(
                    {
                        "stage": "row_processing_exception",
                        "row_index": int(index) if isinstance(index, (int, float)) else str(index),
                        "error": str(e),
                    }
                )
            db.rollback()
            continue

    if _debug_enabled():
        _log_debug(
            {
                "stage": "debug_summary_first5_failures",
                "failures_first5": debug_failed_rows,
                "rows_processed": int(len(df)),
                "rows_inserted": int(inserted_count),
                "rows_failed": int(failed_count),
            }
        )
    
    # 7. Update Upload Metadata
    upload.rows_processed = len(df)
    upload.rows_inserted = inserted_count
    upload.rows_failed = failed_count
    db.commit()
    
    # 8. Return UploadResponse
    return schemas.UploadResponse(
        upload_id=upload.id,
        rows_processed=upload.rows_processed,
        rows_inserted=upload.rows_inserted,
        rows_failed=upload.rows_failed,
    )
