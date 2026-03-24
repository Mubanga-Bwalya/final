from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, UniqueConstraint, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    departments = relationship(
        "Department",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    employees = relationship(
        "Employee",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    performance_records = relationship(
        "PerformanceRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    uploads = relationship(
        "DataUpload",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    snapshots = relationship(
        "Snapshot",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    password_reset_tokens = relationship(
        "PasswordResetToken",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Department(Base):
    __tablename__ = "departments"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_departments_user_name"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)

    user = relationship("User", back_populates="departments")
    employees = relationship("Employee", back_populates="department", cascade="all, delete-orphan")


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)  # e.g. "Jan 2024"
    month = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="snapshots")
    employees = relationship("Employee", back_populates="snapshot", cascade="all, delete-orphan")
    performance_records = relationship("PerformanceRecord", back_populates="snapshot", cascade="all, delete-orphan")


class Employee(Base):
    __tablename__ = "employees"
    # Note: Unique constraint removed to allow same employee_code across snapshots
    # Uniqueness enforced per snapshot in application logic

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    employee_code = Column(String, index=True, nullable=False)  # EMP-001
    name = Column(String, nullable=False)
    email = Column(String, nullable=True)
    role = Column(String, nullable=False)

    tenure_months = Column(Integer, nullable=True)
    salary = Column(Float, nullable=True)

    department_id = Column(Integer, ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=True)

    extra_features = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="employees")
    department = relationship("Department", back_populates="employees")
    snapshot = relationship("Snapshot", back_populates="employees")
    performance_records = relationship(
        "PerformanceRecord",
        back_populates="employee",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PerformanceRecord(Base):
    __tablename__ = "performance_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=True)

    score = Column(Integer, nullable=False)  # e.g. 1–5
    review_period = Column(String, nullable=False)  # "2024-Q4"

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="performance_records")
    employee = relationship("Employee", back_populates="performance_records")
    snapshot = relationship("Snapshot", back_populates="performance_records")


class DataUpload(Base):
    __tablename__ = "data_uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    rows_processed = Column(Integer, default=0)
    rows_inserted = Column(Integer, default=0)
    rows_failed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="uploads")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    used = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="password_reset_tokens")
