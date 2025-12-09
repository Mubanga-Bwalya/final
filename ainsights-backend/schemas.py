from pydantic import BaseModel, EmailStr


class EmployeeBase(BaseModel):
    name: str
    email: EmailStr
    department: str
    role: str
    tenure_months: int
    salary: float
    performance_score: int


class EmployeeCreate(EmployeeBase):
    pass


class EmployeeOut(EmployeeBase):
    employee_id: str

    class Config:
        orm_mode = True


class DashboardKPIs(BaseModel):
    totalEmployees: int
    predictedTurnoverRate: float
    highAbsenteeismRate: float
    underpaidHighPerformers: int
