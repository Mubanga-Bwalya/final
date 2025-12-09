from typing import List

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Employee
from schemas import EmployeeCreate, EmployeeOut, DashboardKPIs

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="A-Insights FastAPI Backend")

# Frontend dev origins (Vite)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "A-Insights FastAPI backend"}


@app.get("/api/dashboard/kpis", response_model=DashboardKPIs)
def get_dashboard_kpis(db: Session = Depends(get_db)):
    total_employees = db.query(Employee).count()

    # For now, only totalEmployees is real; others are placeholders
    return DashboardKPIs(
        totalEmployees=total_employees,
        predictedTurnoverRate=0.0,
        highAbsenteeismRate=0.0,
        underpaidHighPerformers=0,
    )


@app.post("/api/employees", response_model=EmployeeOut)
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)):
    # Generate employee_id like "emp-001"
    last = db.query(Employee).order_by(Employee.id.desc()).first()
    next_number = 1 if not last else last.id + 1
    employee_id = f"emp-{next_number:03d}"

    employee = Employee(
        employee_id=employee_id,
        name=payload.name,
        email=payload.email,
        department=payload.department,
        role=payload.role,
        tenure_months=payload.tenure_months,
        salary=payload.salary,
        performance_score=payload.performance_score,
    )

    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee


@app.get("/api/employees", response_model=List[EmployeeOut])
def list_employees(db: Session = Depends(get_db)):
    employees = db.query(Employee).all()
    return employees


@app.get("/api/employees/{employee_id}", response_model=EmployeeOut)
def get_employee(employee_id: str, db: Session = Depends(get_db)):
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee
