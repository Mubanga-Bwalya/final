from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List, Dict, Any
import csv
import io
import statistics
from database import get_db
import models
import schemas
from security import get_current_user
from ml_service import (
    predict_snapshot,
    get_model_feature_columns,
    compute_feature_coverage_for_employees,
    build_feature_row_for_employee,
    predict_probability_single_row,
    apply_attrition_simulation_overrides,
    coverage_ratio_for_row,
    risk_level_from_probability,
    MIN_FEATURE_COVERAGE_RATIO,
)

router = APIRouter(prefix="/api/v1/snapshots", tags=["snapshots"])


@router.post("/", response_model=schemas.SnapshotOut, status_code=status.HTTP_201_CREATED)
def create_snapshot(
    snapshot: schemas.SnapshotCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Create a new snapshot.
    
    Requirements:
    - month must be 1-12
    - year must be >= 2000
    - Name must be unique per user per month/year
    """
    # Validate month
    if snapshot.month < 1 or snapshot.month > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Month must be between 1 and 12"
        )
    
    # Validate year
    if snapshot.year < 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Year must be >= 2000"
        )
    
    # Check if snapshot with same name, month, year exists for this user
    existing = (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.user_id == current_user.id,
            models.Snapshot.name == snapshot.name,
            models.Snapshot.month == snapshot.month,
            models.Snapshot.year == snapshot.year,
        )
        .first()
    )
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Snapshot with this name, month, and year already exists for this user"
        )
    
    db_snapshot = models.Snapshot(
        user_id=current_user.id,
        name=snapshot.name,
        month=snapshot.month,
        year=snapshot.year,
    )
    db.add(db_snapshot)
    db.commit()
    db.refresh(db_snapshot)
    return db_snapshot


@router.get("/", response_model=List[schemas.SnapshotOut])
def list_snapshots(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    List all snapshots for the current user.
    Ordered by year, then month (ascending).
    """
    snapshots = (
        db.query(models.Snapshot)
        .filter(models.Snapshot.user_id == current_user.id)
        .order_by(models.Snapshot.year, models.Snapshot.month)
        .all()
    )
    return snapshots


@router.put("/{snapshot_id}", response_model=schemas.SnapshotOut)
def update_snapshot(
    snapshot_id: int,
    payload: schemas.SnapshotUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Rename/update a snapshot.

    - Must belong to current_user
    - month 1-12, year >= 2000
    - name/month/year must be unique per user
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )

    if payload.month < 1 or payload.month > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Month must be between 1 and 12",
        )

    if payload.year < 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Year must be >= 2000",
        )

    existing = (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.user_id == current_user.id,
            models.Snapshot.name == payload.name,
            models.Snapshot.month == payload.month,
            models.Snapshot.year == payload.year,
            models.Snapshot.id != snapshot.id,
        )
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Snapshot with this name, month, and year already exists for this user",
        )

    snapshot.name = payload.name
    snapshot.month = payload.month
    snapshot.year = payload.year

    db.commit()
    db.refresh(snapshot)
    return snapshot


@router.delete("/{snapshot_id}", status_code=status.HTTP_200_OK)
def delete_snapshot(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Delete a snapshot.
    Must belong to current_user.
    Cascade delete employees and performance records for that snapshot.
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found"
        )
    
    db.delete(snapshot)
    db.commit()
    return {"detail": "Snapshot deleted successfully"}


@router.post("/{snapshot_id}/duplicate", response_model=schemas.SnapshotOut)
def duplicate_snapshot(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Duplicate a snapshot including employees and performance records.

    - Snapshot must belong to current_user
    - New snapshot keeps same month/year, name appended with " (Copy)" (and
      a numeric suffix if needed to keep name/month/year unique per user).
    """
    original = (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.id == snapshot_id,
            models.Snapshot.user_id == current_user.id,
        )
        .first()
    )

    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )

    # Find a unique name for the copy for this user/month/year
    base_name = f"{original.name} (Copy)"
    new_name = base_name
    suffix = 2
    while (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.user_id == current_user.id,
            models.Snapshot.name == new_name,
            models.Snapshot.month == original.month,
            models.Snapshot.year == original.year,
        )
        .first()
        is not None
    ):
        new_name = f"{base_name} {suffix}"
        suffix += 1

    new_snapshot = models.Snapshot(
        user_id=current_user.id,
        name=new_name,
        month=original.month,
        year=original.year,
    )
    db.add(new_snapshot)
    db.flush()  # assign id without full commit

    # Copy employees scoped to this user + snapshot
    original_employees = (
        db.query(models.Employee)
        .filter(
            models.Employee.snapshot_id == original.id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )

    employee_id_map: Dict[int, int] = {}
    for emp in original_employees:
        # Ensure employee_code remains unique per user, even if the underlying
        # database still has a UNIQUE(user_id, employee_code) constraint.
        base_code = emp.employee_code
        new_code = base_code
        suffix = 2

        # If a row with the same user_id + employee_code already exists,
        # generate a suffixed variant (e.g. "EMP-001 2", "EMP-001 3", ...).
        while (
            db.query(models.Employee)
            .filter(
                models.Employee.user_id == current_user.id,
                models.Employee.employee_code == new_code,
            )
            .first()
            is not None
        ):
            new_code = f"{base_code} {suffix}"
            suffix += 1

        new_emp = models.Employee(
            user_id=current_user.id,
            employee_code=new_code,
            name=emp.name,
            email=emp.email,
            role=emp.role,
            tenure_months=emp.tenure_months,
            salary=emp.salary,
            department_id=emp.department_id,
            snapshot_id=new_snapshot.id,
            extra_features=emp.extra_features,
        )
        db.add(new_emp)
        db.flush()
        employee_id_map[emp.id] = new_emp.id

    # Copy performance records for this snapshot/user, remapping employee_id
    original_performance = (
        db.query(models.PerformanceRecord)
        .filter(
            models.PerformanceRecord.snapshot_id == original.id,
            models.PerformanceRecord.user_id == current_user.id,
        )
        .all()
    )

    for pr in original_performance:
        new_emp_id = employee_id_map.get(pr.employee_id)
        if not new_emp_id:
            continue
        new_pr = models.PerformanceRecord(
            user_id=current_user.id,
            employee_id=new_emp_id,
            snapshot_id=new_snapshot.id,
            score=pr.score,
            review_period=pr.review_period,
        )
        db.add(new_pr)

    db.commit()
    db.refresh(new_snapshot)
    return new_snapshot


@router.get("/{snapshot_id}/export")
def export_snapshot_csv(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Export a snapshot's employees as CSV.

    Columns:
      - employee_code
      - name
      - department
      - salary
      - tenure_months
      - performance (latest score in this snapshot if multiple)
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )

    employees = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )

    performance_records = (
        db.query(models.PerformanceRecord)
        .filter(
            models.PerformanceRecord.snapshot_id == snapshot_id,
            models.PerformanceRecord.user_id == current_user.id,
        )
        .all()
    )

    # Map employee_id -> latest performance score in this snapshot (by id as proxy for recency)
    perf_by_employee: Dict[int, int] = {}
    for pr in performance_records:
        existing = perf_by_employee.get(pr.employee_id)
        if existing is None or pr.id > existing:
            perf_by_employee[pr.employee_id] = pr.score

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Employee Code", "Name", "Department", "Salary", "Tenure (months)", "Performance Score"])

    for e in employees:
        dept_name = e.department.name if e.department else ""
        perf_score = perf_by_employee.get(e.id, "")
        writer.writerow(
            [
                e.employee_code,
                e.name,
                dept_name,
                e.salary if e.salary is not None else "",
                e.tenure_months if e.tenure_months is not None else "",
                perf_score,
            ]
        )

    output.seek(0)

    filename = f"snapshot_{snapshot_id}_export.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{snapshot_id}/compare/{other_snapshot_id}")
def compare_snapshots(
    snapshot_id: int,
    other_snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Compare two snapshots.
    
    Returns:
    - headcount_change: difference in total employees
    - new_hires: employees in new snapshot but not in old
    - terminations: employees in old snapshot but not in new
    - avg_salary_change: difference in average salary
    - avg_performance_change: difference in average performance score
    - department_changes: list of department headcount changes
    """
    # Verify both snapshots belong to current user
    snapshot_old = (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.id == snapshot_id,
            models.Snapshot.user_id == current_user.id,
        )
        .first()
    )
    
    snapshot_new = (
        db.query(models.Snapshot)
        .filter(
            models.Snapshot.id == other_snapshot_id,
            models.Snapshot.user_id == current_user.id,
        )
        .first()
    )
    
    if not snapshot_old:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot {snapshot_id} not found"
        )
    
    if not snapshot_new:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot {other_snapshot_id} not found"
        )
    
    # Get employees for both snapshots (with department relationship loaded)
    employees_old = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )
    
    employees_new = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.snapshot_id == other_snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )
    
    # Create sets of employee codes for comparison
    codes_old = {emp.employee_code for emp in employees_old}
    codes_new = {emp.employee_code for emp in employees_new}
    
    # Calculate metrics
    headcount_change = len(employees_new) - len(employees_old)
    new_hires = len(codes_new - codes_old)
    terminations = len(codes_old - codes_new)
    
    # Calculate average salary changes
    salaries_old = [emp.salary for emp in employees_old if emp.salary is not None]
    salaries_new = [emp.salary for emp in employees_new if emp.salary is not None]
    
    avg_salary_old = sum(salaries_old) / len(salaries_old) if salaries_old else 0.0
    avg_salary_new = sum(salaries_new) / len(salaries_new) if salaries_new else 0.0
    avg_salary_change = avg_salary_new - avg_salary_old
    
    # Calculate average performance changes
    # Get performance records for both snapshots
    perf_old = (
        db.query(models.PerformanceRecord)
        .filter(
            models.PerformanceRecord.snapshot_id == snapshot_id,
            models.PerformanceRecord.user_id == current_user.id,
        )
        .all()
    )
    
    perf_new = (
        db.query(models.PerformanceRecord)
        .filter(
            models.PerformanceRecord.snapshot_id == other_snapshot_id,
            models.PerformanceRecord.user_id == current_user.id,
        )
        .all()
    )
    
    scores_old = [p.score for p in perf_old]
    scores_new = [p.score for p in perf_new]
    
    avg_perf_old = sum(scores_old) / len(scores_old) if scores_old else 0.0
    avg_perf_new = sum(scores_new) / len(scores_new) if scores_new else 0.0
    avg_performance_change = avg_perf_new - avg_perf_old
    
    # Calculate department changes
    # Group by department for old snapshot
    dept_counts_old = {}
    for emp in employees_old:
        dept_name = emp.department.name
        dept_counts_old[dept_name] = dept_counts_old.get(dept_name, 0) + 1
    
    # Group by department for new snapshot
    dept_counts_new = {}
    for emp in employees_new:
        dept_name = emp.department.name
        dept_counts_new[dept_name] = dept_counts_new.get(dept_name, 0) + 1
    
    # Calculate changes per department
    all_departments = set(dept_counts_old.keys()) | set(dept_counts_new.keys())
    department_changes = []
    for dept in all_departments:
        count_old = dept_counts_old.get(dept, 0)
        count_new = dept_counts_new.get(dept, 0)
        department_changes.append({
            "department": dept,
            "headcount_change": count_new - count_old
        })
    
    return {
        "headcount_change": headcount_change,
        "new_hires": new_hires,
        "terminations": terminations,
        "avg_salary_change": round(avg_salary_change, 2),
        "avg_performance_change": round(avg_performance_change, 2),
        "department_changes": department_changes
    }


@router.get("/{snapshot_id}/analytics", response_model=schemas.SnapshotAnalytics)
def get_snapshot_analytics(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Get analytics for a specific snapshot.
    
    Returns:
    - headcount: Total number of employees in the snapshot
    - department_distribution: List of departments with counts
    - average_salary: Average salary (ignoring nulls, 0 if none)
    - salary_std_dev: Standard deviation of salaries (0 if < 2 valid salaries)
    - average_tenure_months: Average tenure in months (ignoring nulls, 0 if none)
    - average_performance_score: Average performance score (ignoring nulls, 0 if none)
    - salary_by_department: Average salary per department
    - performance_distribution: Count of performance scores (1-5)
    """
    # Verify snapshot exists and belongs to current_user
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found"
        )
    
    # Load all employees for this snapshot in one query (with department relationship)
    employees = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )
    
    # Load all performance records for this snapshot in one query
    performance_records = (
        db.query(models.PerformanceRecord)
        .filter(
            models.PerformanceRecord.snapshot_id == snapshot_id,
            models.PerformanceRecord.user_id == current_user.id,
        )
        .all()
    )
    
    # 1. Headcount
    headcount = len(employees)
    
    # 2. Department Distribution
    dept_counts = {}
    for emp in employees:
        dept_name = emp.department.name
        dept_counts[dept_name] = dept_counts.get(dept_name, 0) + 1
    
    department_distribution = [
        {"department": dept, "count": count}
        for dept, count in dept_counts.items()
    ]
    
    # 3. Average Salary (ignore nulls)
    salaries = [emp.salary for emp in employees if emp.salary is not None]
    average_salary = sum(salaries) / len(salaries) if salaries else 0.0
    
    # 4. Salary Standard Deviation
    salary_std_dev = 0.0
    if len(salaries) >= 2:
        salary_std_dev = statistics.stdev(salaries)
    
    # 5. Average Tenure (in months, ignore nulls)
    tenures = [emp.tenure_months for emp in employees if emp.tenure_months is not None]
    average_tenure_months = sum(tenures) / len(tenures) if tenures else 0.0
    
    # 6. Average Performance Score (only for employees in this snapshot, ignore nulls)
    # Create set of employee IDs in this snapshot for efficient lookup
    employee_ids_in_snapshot = {emp.id for emp in employees}
    performance_scores = [
        pr.score for pr in performance_records
        if pr.employee_id in employee_ids_in_snapshot and pr.score is not None
    ]
    average_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
    
    # 7. Salary by Department
    dept_salaries = {}
    dept_salary_counts = {}
    for emp in employees:
        if emp.salary is not None:
            dept_name = emp.department.name
            if dept_name not in dept_salaries:
                dept_salaries[dept_name] = 0.0
                dept_salary_counts[dept_name] = 0
            dept_salaries[dept_name] += emp.salary
            dept_salary_counts[dept_name] += 1
    
    salary_by_department = [
        {
            "department": dept,
            "avg_salary": dept_salaries[dept] / dept_salary_counts[dept]
        }
        for dept in dept_salaries.keys()
    ]
    
    # 8. Performance Distribution (only valid integer scores 1-5)
    performance_distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for pr in performance_records:
        if pr.employee_id in employee_ids_in_snapshot and pr.score is not None:
            score_str = str(pr.score)
            if score_str in performance_distribution:
                performance_distribution[score_str] += 1
    
    return schemas.SnapshotAnalytics(
        headcount=headcount,
        department_distribution=department_distribution,
        average_salary=round(average_salary, 2),
        salary_std_dev=round(salary_std_dev, 2),
        average_tenure_months=round(average_tenure_months, 2),
        average_performance_score=round(average_performance_score, 2),
        salary_by_department=salary_by_department,
        performance_distribution=performance_distribution,
    )


@router.get("/{snapshot_id}/employees", response_model=List[schemas.SnapshotEmployeeSummary])
def list_snapshot_employees(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    List employees for a snapshot (for dashboard table merge with predictions).

    Returns, per employee:
    - id
    - employee_code
    - name
    - department (name)
    - role
    - salary
    - tenure_months
    - latest_performance_score (if any performance records exist for this snapshot)

    Snapshot must belong to current_user.
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )
    employees = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )
    # Map employee_id -> latest performance score in this snapshot (by id as proxy for recency)
    performance_records = (
        db.query(models.PerformanceRecord)
        .filter(
            models.PerformanceRecord.snapshot_id == snapshot_id,
            models.PerformanceRecord.user_id == current_user.id,
        )
        .all()
    )
    latest_perf_by_employee: Dict[int, int] = {}
    for pr in performance_records:
        existing_id = latest_perf_by_employee.get(pr.employee_id)
        if existing_id is None or pr.id > existing_id:
            latest_perf_by_employee[pr.employee_id] = pr.score

    return [
        schemas.SnapshotEmployeeSummary(
            id=e.id,
            employee_code=e.employee_code,
            name=e.name,
            department_id=e.department_id,
            department=e.department.name if e.department else "",
            role=e.role,
            salary=e.salary,
            tenure_months=e.tenure_months,
            latest_performance_score=latest_perf_by_employee.get(e.id),
        )
        for e in employees
    ]


@router.get("/{snapshot_id}/risk-summary", response_model=schemas.SnapshotRiskSummary)
def get_snapshot_risk_summary(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Get aggregated attrition risk summary for a snapshot.
    Calls predict_snapshot once and aggregates in memory.
    One employee query (with department) for department_risk_distribution.
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )
    try:
        predictions = predict_snapshot(snapshot_id, db)
    except ValueError as e:
        # Includes low-coverage safety errors from ml_service
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        # Model is unavailable or improperly configured for predictions
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    if not predictions:
        return schemas.SnapshotRiskSummary(
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            avg_risk_probability=0.0,
            department_risk_distribution=[],
        )

    pred_by_id = {p["employee_id"]: p for p in predictions}
    high_risk_count = sum(1 for p in predictions if p["risk_level"] == "High")
    medium_risk_count = sum(1 for p in predictions if p["risk_level"] == "Medium")
    low_risk_count = sum(1 for p in predictions if p["risk_level"] == "Low")
    avg_risk_probability = sum(p["probability"] for p in predictions) / len(predictions)

    employees = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )
    emp_id_to_dept = {e.id: e.department.name for e in employees if e.department}

    dept_agg: Dict[str, Dict[str, Any]] = {}
    for p in predictions:
        dept = emp_id_to_dept.get(p["employee_id"], "Unknown")
        if dept not in dept_agg:
            dept_agg[dept] = {"sum_prob": 0.0, "count": 0, "high": 0}
        dept_agg[dept]["sum_prob"] += p["probability"]
        dept_agg[dept]["count"] += 1
        if p["risk_level"] == "High":
            dept_agg[dept]["high"] += 1

    department_risk_distribution: List[schemas.DepartmentRiskSummary] = [
        schemas.DepartmentRiskSummary(
            department=dept,
            avg_risk=round(data["sum_prob"] / data["count"], 4) if data["count"] else 0.0,
            high_risk_count=data["high"],
            total_employees=data["count"],
        )
        for dept, data in dept_agg.items()
    ]

    return schemas.SnapshotRiskSummary(
        high_risk_count=high_risk_count,
        medium_risk_count=medium_risk_count,
        low_risk_count=low_risk_count,
        avg_risk_probability=round(avg_risk_probability, 4),
        department_risk_distribution=department_risk_distribution,
    )


@router.get("/{snapshot_id}/predictions", response_model=List[schemas.PredictionResult])
def get_snapshot_predictions(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Get attrition-risk predictions for all employees in a snapshot.

    Returns a list of { employee_id, probability, risk_level } per employee.
    Requires authentication; snapshot must belong to the current user.
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )
    try:
        predictions = predict_snapshot(snapshot_id, db)
        return predictions
    except ValueError as e:
        # Includes low-coverage safety errors from ml_service
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        # Model is unavailable or improperly configured for predictions
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


def _attrition_feature_view_from_row(row: Dict[str, Any]) -> schemas.AttritionSimulationFeatureView:
    return schemas.AttritionSimulationFeatureView(
        monthly_income=row.get("MonthlyIncome"),
        years_at_company=row.get("YearsAtCompany"),
        job_role=row.get("JobRole"),
        department=row.get("Department"),
        performance_rating=row.get("PerformanceRating"),
    )


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.post(
    "/{snapshot_id}/employees/{employee_id}/simulate-attrition",
    response_model=schemas.AttritionSimulationResponse,
)
def simulate_attrition_scenario(
    snapshot_id: int,
    employee_id: int,
    body: schemas.AttritionSimulationRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    What-if / scenario analysis: re-score one employee after applying optional overrides
    to model-aligned fields (same pipeline as batch prediction). Does not persist changes.
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )

    employee = (
        db.query(models.Employee)
        .options(joinedload(models.Employee.department))
        .filter(
            models.Employee.id == employee_id,
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .first()
    )
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found in this snapshot",
        )

    feature_columns = get_model_feature_columns()
    if not feature_columns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Attrition model not available or missing input feature schema. "
                "Train the model with ml_training.py and ensure feature names are saved."
            ),
        )

    base_row = build_feature_row_for_employee(employee, feature_columns)

    dept_name_override: str | None = None
    fields_set = body.model_fields_set
    if "department_id" in fields_set:
        if body.department_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="department_id was set but is null",
            )
        dept = (
            db.query(models.Department)
            .filter(
                models.Department.id == body.department_id,
                models.Department.user_id == current_user.id,
            )
            .first()
        )
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Department not found",
            )
        dept_name_override = dept.name

    try:
        baseline_prob = predict_probability_single_row(base_row, feature_columns)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    baseline_cov = coverage_ratio_for_row(base_row, feature_columns)
    baseline_low = baseline_cov < MIN_FEATURE_COVERAGE_RATIO

    sim_row = apply_attrition_simulation_overrides(
        base_row,
        feature_columns,
        salary=body.salary if "salary" in fields_set else None,
        tenure_months=body.tenure_months if "tenure_months" in fields_set else None,
        job_role=body.job_role if "job_role" in fields_set else None,
        department_name=dept_name_override,
        performance_rating=body.performance_rating if "performance_rating" in fields_set else None,
    )

    try:
        sim_prob = predict_probability_single_row(sim_row, feature_columns)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    sim_cov = coverage_ratio_for_row(sim_row, feature_columns)
    sim_low = sim_cov < MIN_FEATURE_COVERAGE_RATIO

    delta = sim_prob - baseline_prob
    requested_overrides: Dict[str, Any] = {}
    if "salary" in fields_set:
        requested_overrides["salary"] = body.salary
    if "tenure_months" in fields_set:
        requested_overrides["tenure_months"] = body.tenure_months
    if "job_role" in fields_set:
        requested_overrides["job_role"] = body.job_role
    if "department_id" in fields_set:
        requested_overrides["department_id"] = body.department_id
    if "performance_rating" in fields_set:
        requested_overrides["performance_rating"] = body.performance_rating

    overrides_applied: Dict[str, Any] = {}
    if "salary" in fields_set and body.salary is not None and "MonthlyIncome" in feature_columns:
        overrides_applied["MonthlyIncome"] = body.salary
    if "tenure_months" in fields_set and body.tenure_months is not None and "YearsAtCompany" in feature_columns:
        overrides_applied["YearsAtCompany"] = float(body.tenure_months) / 12.0
    if "job_role" in fields_set and body.job_role is not None and "JobRole" in feature_columns:
        overrides_applied["JobRole"] = body.job_role
    if dept_name_override is not None and "Department" in feature_columns:
        overrides_applied["Department"] = dept_name_override
    if "performance_rating" in fields_set and body.performance_rating is not None and "PerformanceRating" in feature_columns:
        overrides_applied["PerformanceRating"] = body.performance_rating

    changed_model_fields: Dict[str, Dict[str, Any]] = {}
    for model_field in [
        "MonthlyIncome",
        "YearsAtCompany",
        "JobRole",
        "Department",
        "PerformanceRating",
    ]:
        before = base_row.get(model_field)
        after = sim_row.get(model_field)
        if before != after:
            changed_model_fields[model_field] = {"baseline": before, "simulated": after}

    model_feature_presence = {
        "MonthlyIncome": "MonthlyIncome" in feature_columns,
        "YearsAtCompany": "YearsAtCompany" in feature_columns,
        "JobRole": "JobRole" in feature_columns,
        "Department": "Department" in feature_columns,
        "PerformanceRating": "PerformanceRating" in feature_columns,
    }

    disclaimer = (
        "This is a scenario-based estimate from the trained statistical model. "
        "It is an indicator of relative risk, not a certain outcome, and must not replace "
        "professional HR judgment or policy."
    )

    return schemas.AttritionSimulationResponse(
        employee_id=employee_id,
        snapshot_id=snapshot_id,
        baseline=schemas.AttritionSimulationOutcome(
            probability=round(baseline_prob, 6),
            risk_level=risk_level_from_probability(baseline_prob),
            feature_coverage_ratio=round(baseline_cov, 6),
            low_coverage=baseline_low,
        ),
        simulated=schemas.AttritionSimulationOutcome(
            probability=round(sim_prob, 6),
            risk_level=risk_level_from_probability(sim_prob),
            feature_coverage_ratio=round(sim_cov, 6),
            low_coverage=sim_low,
        ),
        probability_delta=round(delta, 6),
        absolute_probability_delta=round(abs(delta), 6),
        baseline_features=_attrition_feature_view_from_row(base_row),
        simulated_features=_attrition_feature_view_from_row(sim_row),
        overrides_applied=overrides_applied,
        requested_overrides=requested_overrides,
        applied_model_overrides=overrides_applied,
        changed_model_fields=changed_model_fields,
        model_feature_presence=model_feature_presence,
        debug_probability={
            "baseline_raw": baseline_prob,
            "simulated_raw": sim_prob,
            "delta_raw": delta,
            "delta_percentage_points_raw": delta * 100.0,
            "baseline_percent_raw": baseline_prob * 100.0,
            "simulated_percent_raw": sim_prob * 100.0,
            "baseline_rounded_6dp": round(baseline_prob, 6),
            "simulated_rounded_6dp": round(sim_prob, 6),
            "delta_rounded_6dp": round(delta, 6),
            "baseline_rounded_1dp_percent": round((baseline_prob * 100.0), 1),
            "simulated_rounded_1dp_percent": round((sim_prob * 100.0), 1),
            "delta_rounded_1dp_points": round((delta * 100.0), 1),
            "baseline_monthly_income_raw": _safe_float_or_none(base_row.get("MonthlyIncome")),
            "simulated_monthly_income_raw": _safe_float_or_none(sim_row.get("MonthlyIncome")),
            "baseline_years_at_company_raw": _safe_float_or_none(base_row.get("YearsAtCompany")),
            "simulated_years_at_company_raw": _safe_float_or_none(sim_row.get("YearsAtCompany")),
            "baseline_performance_rating_raw": _safe_float_or_none(base_row.get("PerformanceRating")),
            "simulated_performance_rating_raw": _safe_float_or_none(sim_row.get("PerformanceRating")),
        },
        disclaimer=disclaimer,
    )


@router.get("/{snapshot_id}/feature-coverage", response_model=schemas.FeatureCoverageSnapshotSummary)
def get_snapshot_feature_coverage(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Diagnose how well the current snapshot data covers the model's expected
    input feature schema.

    Returns:
    - snapshot_id
    - total_employees
    - model_feature_count
    - average_coverage_ratio
    - employees: list of per-employee coverage summaries
      (employee_id, employee_code, employee_name, coverage_ratio,
       populated_count, missing_count, missing_features_sample)
    """
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot not found",
        )

    employees = (
        db.query(models.Employee)
        .filter(
            models.Employee.snapshot_id == snapshot_id,
            models.Employee.user_id == current_user.id,
        )
        .all()
    )

    feature_columns = get_model_feature_columns()
    if not feature_columns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Attrition model not available or missing input feature schema. "
                "Train the model with ml_training.py and ensure feature names are saved."
            ),
        )

    if not employees:
        return schemas.FeatureCoverageSnapshotSummary(
            snapshot_id=snapshot_id,
            total_employees=0,
            model_feature_count=len(feature_columns),
            average_coverage_ratio=0.0,
            employees=[],
        )

    coverage_info = compute_feature_coverage_for_employees(employees, feature_columns)

    if not coverage_info:
        return schemas.FeatureCoverageSnapshotSummary(
            snapshot_id=snapshot_id,
            total_employees=0,
            model_feature_count=len(feature_columns),
            average_coverage_ratio=0.0,
            employees=[],
        )

    avg_coverage = float(
        sum(c["coverage_ratio"] for c in coverage_info) / len(coverage_info)
    )

    employees_summary: List[schemas.FeatureCoverageEmployeeSummary] = []
    for cov in coverage_info:
        employees_summary.append(
            schemas.FeatureCoverageEmployeeSummary(
                employee_id=cov["employee_id"],
                employee_code=cov["employee_code"],
                employee_name=cov["employee_name"],
                coverage_ratio=cov["coverage_ratio"],
                populated_count=cov["populated_count"],
                missing_count=cov["missing_count"],
                # Keep the list short enough for practical inspection
                missing_features_sample=cov["missing_features"][:10],
            )
        )

    return schemas.FeatureCoverageSnapshotSummary(
        snapshot_id=snapshot_id,
        total_employees=len(employees),
        model_feature_count=len(feature_columns),
        average_coverage_ratio=avg_coverage,
        employees=employees_summary,
    )
