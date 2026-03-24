from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, ConfigDict, Field, field_validator


# ======================
# AUTH / USER SCHEMAS
# ======================

class UserCreate(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str):
        # bcrypt hard limit (72 bytes)
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer")
        # basic security requirement
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None


class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str):
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer")
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_reset_password(cls, v: str):
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer")
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserOut(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str


# ======================
# DEPARTMENT SCHEMAS
# ======================

class DepartmentBase(BaseModel):
    name: str


class DepartmentCreate(DepartmentBase):
    pass


class DepartmentOut(DepartmentBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


# ======================
# EMPLOYEE SCHEMAS
# ======================

class EmployeeBase(BaseModel):
    employee_code: str
    name: str
    email: Optional[EmailStr]
    role: str
    tenure_months: Optional[int]
    salary: Optional[float]
    department_id: int
    extra_features: Optional[Dict[str, Any]]


class EmployeeCreate(EmployeeBase):
    pass


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    tenure_months: Optional[int] = None
    salary: Optional[float] = None
    department_id: Optional[int] = None
    extra_features: Optional[Dict[str, Any]] = None


class EmployeeOut(EmployeeBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ======================
# PERFORMANCE SCHEMAS
# ======================

class PerformanceRecordBase(BaseModel):
    employee_id: int
    score: int
    review_period: str


class PerformanceRecordCreate(PerformanceRecordBase):
    pass


class PerformanceRecordOut(PerformanceRecordBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ======================
# DASHBOARD (NO AI)
# ======================

class DashboardKPIs(BaseModel):
    total_employees: int
    average_performance: float
    average_salary: float


# ======================
# UPLOAD PIPELINE (FUTURE)
# ======================

class CleanedRow(BaseModel):
    employee_code: str
    name: str
    email: Optional[str]
    role: str
    tenure_months: Optional[int]
    salary: Optional[float]
    department: str
    extra_features: Optional[Dict[str, Any]]


class UploadPreview(BaseModel):
    upload_id: str
    preview_rows: List[CleanedRow]


class UploadResponse(BaseModel):
    upload_id: int
    rows_processed: int
    rows_inserted: int
    rows_failed: int

    model_config = ConfigDict(from_attributes=True)


# ======================
# SNAPSHOT SCHEMAS
# ======================

class SnapshotCreate(BaseModel):
    name: str
    month: int
    year: int


class SnapshotUpdate(BaseModel):
    name: str
    month: int
    year: int


class SnapshotOut(BaseModel):
    id: int
    name: str
    month: int
    year: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SnapshotAnalytics(BaseModel):
    headcount: int
    department_distribution: List[Dict[str, Any]]
    average_salary: float
    salary_std_dev: float
    average_tenure_months: float
    average_performance_score: float
    salary_by_department: List[Dict[str, Any]]
    performance_distribution: Dict[str, int]

    model_config = ConfigDict(from_attributes=True)


# ======================
# SNAPSHOT EMPLOYEE / PREDICTION / COVERAGE / MODEL INFO
# ======================


class SnapshotEmployeeSummary(BaseModel):
    id: int
    employee_code: str
    name: str
    department_id: int
    department: str
    role: str
    salary: Optional[float]
    tenure_months: Optional[int]
    latest_performance_score: Optional[int]


class PredictionResult(BaseModel):
    employee_id: int
    probability: float
    risk_level: str
    feature_coverage: float
    populated_feature_count: int
    missing_feature_count: int
    missing_features_sample: List[str]
    low_coverage: bool


class FeatureCoverageEmployeeSummary(BaseModel):
    employee_id: int
    employee_code: Optional[str]
    employee_name: Optional[str]
    coverage_ratio: float
    populated_count: int
    missing_count: int
    missing_features_sample: List[str]


class FeatureCoverageSnapshotSummary(BaseModel):
    snapshot_id: int
    total_employees: int
    model_feature_count: int
    average_coverage_ratio: float
    employees: List[FeatureCoverageEmployeeSummary]


class ModelSummary(BaseModel):
    trained_at_utc: Optional[str]
    training: Dict[str, Any]
    libraries: Dict[str, Any]
    model: Dict[str, Any]
    metrics: Dict[str, Any]
    limitations: List[str]


class DepartmentRiskSummary(BaseModel):
    department: str
    avg_risk: float
    high_risk_count: int
    total_employees: int


class SnapshotRiskSummary(BaseModel):
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk_probability: float
    department_risk_distribution: List[DepartmentRiskSummary]


class AttritionSimulationRequest(BaseModel):
    """
    Optional overrides for scenario simulation. Omitted fields keep the employee's stored values.
    """

    salary: Optional[float] = Field(None, ge=0)
    tenure_months: Optional[int] = Field(None, ge=0)
    job_role: Optional[str] = None
    department_id: Optional[int] = None
    performance_rating: Optional[int] = Field(None, ge=1, le=5)


class AttritionSimulationFeatureView(BaseModel):
    """Subset of model inputs exposed for transparency (IBM-style names where applicable)."""

    monthly_income: Optional[float] = None
    years_at_company: Optional[float] = None
    job_role: Optional[str] = None
    department: Optional[str] = None
    performance_rating: Optional[int] = None


class AttritionSimulationOutcome(BaseModel):
    probability: float
    risk_level: str
    feature_coverage_ratio: float
    low_coverage: bool


class AttritionSimulationResponse(BaseModel):
    employee_id: int
    snapshot_id: int
    baseline: AttritionSimulationOutcome
    simulated: AttritionSimulationOutcome
    probability_delta: float
    absolute_probability_delta: float
    baseline_features: AttritionSimulationFeatureView
    simulated_features: AttritionSimulationFeatureView
    overrides_applied: Dict[str, Any]
    requested_overrides: Dict[str, Any]
    applied_model_overrides: Dict[str, Any]
    changed_model_fields: Dict[str, Dict[str, Any]]
    model_feature_presence: Dict[str, bool]
    debug_probability: Dict[str, float]
    disclaimer: str
