from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = "sqlite:///./ainsights.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def _column_exists(conn, table: str, column: str) -> bool:
    """Return True if column exists in table (SQLite)."""
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result.fetchall())


def migrate_add_missing_columns():
    """
    Add columns that were added to models after the DB was first created.
    create_all() only creates new tables; it does not add columns to existing tables.
    """
    with engine.connect() as conn:
        # employees.snapshot_id
        if not _column_exists(conn, "employees", "snapshot_id"):
            conn.execute(text("ALTER TABLE employees ADD COLUMN snapshot_id INTEGER REFERENCES snapshots(id) ON DELETE CASCADE"))
            conn.commit()
        # performance_records.snapshot_id
        if not _column_exists(conn, "performance_records", "snapshot_id"):
            conn.execute(text("ALTER TABLE performance_records ADD COLUMN snapshot_id INTEGER REFERENCES snapshots(id) ON DELETE CASCADE"))
            conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
