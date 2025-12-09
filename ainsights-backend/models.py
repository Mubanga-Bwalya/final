from sqlalchemy import Column, Integer, String, Float
from database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True)  # e.g. "emp-001"
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    department = Column(String, nullable=False)
    role = Column(String, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    salary = Column(Float, nullable=False)
    performance_score = Column(Integer, nullable=False)
