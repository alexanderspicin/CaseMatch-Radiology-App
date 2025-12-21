import uuid
from typing import List

from pydantic import BaseModel, EmailStr

from src.schemas.balance import BalanceSchema, TransactionSchema


class UserSchema(BaseModel):
    id: uuid.UUID
    email: EmailStr
    balance: BalanceSchema
    transactions: List[TransactionSchema]

    class Config:
        from_attributes = True


class UserInDB(UserSchema):
    hashed_password: str


class UserCreateSchema(BaseModel):
    email: EmailStr
    password: str


class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str