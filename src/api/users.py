from fastapi import Depends, HTTPException, APIRouter
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.models.user import User
from src.schemas.user import UserCreateSchema, UserLoginSchema
from src.services.auth import create_access_token, JWTBearer
from src.services.user_service import get_password_hash, verify_password

router = APIRouter(prefix="/user", tags=["user"])

@router.post('/register')
async def register(user: UserCreateSchema, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post('/login')
async def login(user: UserLoginSchema, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if not existing_user:
        raise HTTPException(status_code=400, detail="Email not registered")
    if not verify_password(user.password, existing_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    return create_access_token(user.model_dump())


@router.get('/me', dependencies=[Depends(JWTBearer())])
async def get_me(db: Session = Depends(get_db)):
    return {"username": db.query(User).first().email}



