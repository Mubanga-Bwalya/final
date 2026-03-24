from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta, timezone
import secrets

from database import engine, get_db, migrate_add_missing_columns
import models
import schemas
from security import create_access_token, get_current_user, get_password_hash, verify_password
from users_routes import router as users_router
from upload_routes import router as upload_router
from snapshot_routes import router as snapshot_router
from model_routes import router as model_router

# ======================
# APP SETUP
# ======================

app = FastAPI(title="A-Insights API", version="1.0")
app.include_router(users_router)
app.include_router(upload_router)
app.include_router(snapshot_router)
app.include_router(model_router)

# ======================
# CORS CONFIGURATION
# ======================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# DATABASE SETUP
# ======================

models.Base.metadata.create_all(bind=engine)
migrate_add_missing_columns()

# ======================
# AUTH ROUTES
# ======================


@app.post("/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(email=user_in.email, hashed_password=get_password_hash(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm provides username + password; we use username as email
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user_id=user.id)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/logout")
def logout():
    # Stateless JWT logout: client discards token
    return {"detail": "Logged out"}


@app.get("/auth/me", response_model=schemas.UserOut)
def me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/auth/forgot-password")
def forgot_password(
    body: schemas.ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Generate a password reset token if the email exists.
    Always return a generic message to avoid leaking account existence.
    """
    user = db.query(models.User).filter(models.User.email == body.email).first()

    if user:
        # Create a secure, unique token with 30 minute expiry
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)

        reset_token = models.PasswordResetToken(
            user_id=user.id,
            token=token,
            expires_at=expires_at,
        )
        db.add(reset_token)
        db.commit()

        # In a real system we would email the link containing this token.

    return {
        "detail": "If an account with that email exists, a reset link has been generated."
    }


@app.post("/auth/reset-password")
def reset_password(
    body: schemas.ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Reset a user's password using a one-time token.
    """
    token_row = (
        db.query(models.PasswordResetToken)
        .filter(models.PasswordResetToken.token == body.token)
        .first()
    )

    if not token_row:
        raise HTTPException(status_code=400, detail="Invalid token")

    now = datetime.now(timezone.utc)

    if token_row.used:
        raise HTTPException(status_code=400, detail="Token has already been used")

    if token_row.expires_at < now:
        raise HTTPException(status_code=400, detail="Token has expired")

    user = db.query(models.User).filter(models.User.id == token_row.user_id).first()
    if not user:
        # Defensive: mark token used and treat as invalid
        token_row.used = True
        db.commit()
        raise HTTPException(status_code=400, detail="Invalid token")

    user.hashed_password = get_password_hash(body.new_password)
    token_row.used = True
    db.commit()

    return {"detail": "Password reset successful"}


# ======================
# DEPARTMENT CRUD
# ======================

@app.post("/api/v1/departments", response_model=schemas.DepartmentOut)
def create_department(
    department: schemas.DepartmentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    existing = (
        db.query(models.Department)
        .filter(
            models.Department.user_id == current_user.id,
            models.Department.name == department.name,
        )
        .first()
    )

    if existing:
        raise HTTPException(status_code=400, detail="Department already exists")

    db_department = models.Department(name=department.name, user_id=current_user.id)
    db.add(db_department)
    db.commit()
    db.refresh(db_department)
    return db_department


@app.get("/api/v1/departments", response_model=List[schemas.DepartmentOut])
def get_departments(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.Department).filter(models.Department.user_id == current_user.id).all()


# ======================
# EMPLOYEE CRUD
# ======================

@app.post("/api/v1/employees", response_model=schemas.EmployeeOut)
def create_employee(
    employee: schemas.EmployeeCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    department = (
        db.query(models.Department)
        .filter(
            models.Department.id == employee.department_id,
            models.Department.user_id == current_user.id,
        )
        .first()
    )
    if not department:
        raise HTTPException(status_code=404, detail="Department not found")

    db_employee = models.Employee(**employee.dict(), user_id=current_user.id)
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee


@app.get("/api/v1/employees", response_model=List[schemas.EmployeeOut])
def get_employees(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.Employee).filter(models.Employee.user_id == current_user.id).all()


@app.get("/api/v1/employees/{employee_id}", response_model=schemas.EmployeeOut)
def get_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    employee = (
        db.query(models.Employee)
        .filter(models.Employee.id == employee_id, models.Employee.user_id == current_user.id)
        .first()
    )

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    return employee


@app.put("/api/v1/employees/{employee_id}", response_model=schemas.EmployeeOut)
def update_employee(
    employee_id: int,
    employee_update: schemas.EmployeeUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    employee = (
        db.query(models.Employee)
        .filter(models.Employee.id == employee_id, models.Employee.user_id == current_user.id)
        .first()
    )

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    for field, value in employee_update.dict(exclude_unset=True).items():
        setattr(employee, field, value)

    if employee_update.department_id is not None:
        department = (
            db.query(models.Department)
            .filter(
                models.Department.id == employee_update.department_id,
                models.Department.user_id == current_user.id,
            )
            .first()
        )
        if not department:
            raise HTTPException(status_code=404, detail="Department not found")

    db.commit()
    db.refresh(employee)
    return employee


@app.delete("/api/v1/employees/{employee_id}")
def delete_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    employee = (
        db.query(models.Employee)
        .filter(models.Employee.id == employee_id, models.Employee.user_id == current_user.id)
        .first()
    )

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    db.delete(employee)
    db.commit()
    return {"detail": "Employee deleted successfully"}


# ======================
# PERFORMANCE RECORD CRUD
# ======================

@app.post("/api/v1/performance", response_model=schemas.PerformanceRecordOut)
def create_performance_record(
    record: schemas.PerformanceRecordCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    employee = (
        db.query(models.Employee)
        .filter(models.Employee.id == record.employee_id, models.Employee.user_id == current_user.id)
        .first()
    )
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    db_record = models.PerformanceRecord(**record.dict(), user_id=current_user.id)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


@app.get("/api/v1/performance", response_model=List[schemas.PerformanceRecordOut])
def get_performance_records(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return (
        db.query(models.PerformanceRecord)
        .filter(models.PerformanceRecord.user_id == current_user.id)
        .all()
    )


@app.delete("/api/v1/users/me")
def delete_my_account(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Delete the currently authenticated user's account.

    All related objects (departments, snapshots, employees, performance records,
    uploads, password reset tokens) are removed via SQLAlchemy cascades.
    """
    db.delete(current_user)
    db.commit()
    return {"detail": "Account deleted successfully"}
