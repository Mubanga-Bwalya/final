from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from security import get_current_user, get_password_hash, verify_password


router = APIRouter(prefix="/users", tags=["users"])


@router.put("/me", response_model=schemas.UserOut)
def update_me(
    user_update: schemas.UserUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if user_update.email and user_update.email != current_user.email:
        existing = db.query(models.User).filter(models.User.email == user_update.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = user_update.email

    db.commit()
    db.refresh(current_user)
    return current_user


@router.put("/me/password")
def change_password(
    body: schemas.PasswordUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    return {"detail": "Password updated"}


@router.delete("/me")
def delete_me(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    db.delete(current_user)
    db.commit()
    return {"detail": "Account deleted"}

