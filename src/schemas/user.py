import uuid

from pydantic import BaseModel, EmailStr

class UserSchema(BaseModel):
    id: uuid.UUID
    email: EmailStr


class UserInDB(UserSchema):
    hashed_password: str


class UserCreateSchema(BaseModel):
    email: EmailStr
    password: str


class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str