import uuid

from pydantic import BaseModel

class User(BaseModel):
    id: uuid.UUID
    email: str


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    email: str
    password: str