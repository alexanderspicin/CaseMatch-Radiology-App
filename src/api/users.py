from fastapi import Depends, HTTPException, APIRouter
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.models import Balance, Transaction
from src.models.enums import TransactionType, Status
from src.models.user import User
from src.schemas.user_schemas import UserCreateSchema, UserLoginSchema, UserSchema
from src.services.auth_service import create_access_token, JWTBearer, get_current_user
from src.services.user_service import get_password_hash, verify_password

router = APIRouter(prefix="/user", tags=["user"])

@router.post('/register')
async def register(user: UserCreateSchema, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.flush()

    db_balance = Balance(
        user_id=new_user.id,
    )
    db.add(db_balance)
    initial_transaction = Transaction(
        user_id=new_user.id,
        amount=100.0,
        transaction_status=Status.DONE,
        transaction_type=TransactionType.CREDIT
    )
    db.add(initial_transaction)

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


@router.get('/me', response_model=UserSchema)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user



