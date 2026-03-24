import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

import models
from database import get_db


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Fail fast at startup with a clear RuntimeError if required config is missing.
_require_env("AINSIGHTS_SECRET_KEY")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(*, user_id: int, expires_minutes: Optional[int] = None) -> str:
    secret_key = _require_env("AINSIGHTS_SECRET_KEY")
    algorithm = os.getenv("AINSIGHTS_JWT_ALGORITHM", "HS256")
    expiry_minutes = int(os.getenv("AINSIGHTS_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    if expires_minutes is not None:
        expiry_minutes = expires_minutes

    expire = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
    to_encode = {"sub": str(user_id), "exp": expire}
    return jwt.encode(to_encode, secret_key, algorithm=algorithm)


def decode_token(token: str) -> int:
    secret_key = _require_env("AINSIGHTS_SECRET_KEY")
    algorithm = os.getenv("AINSIGHTS_JWT_ALGORITHM", "HS256")
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        sub = payload.get("sub")
        if sub is None:
            raise ValueError("missing sub")
        return int(sub)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    user_id = decode_token(token)
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

