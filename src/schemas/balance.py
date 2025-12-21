from datetime import datetime

from pydantic import BaseModel, field_validator
import uuid


class BalanceSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    amount: float

    class Config:
        from_attributes = True


class TransactionSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    amount: float
    timestamp: datetime
    transaction_type: str
    transaction_status: str

    class Config:
        from_attributes = True


class CreateTransactionSchema(BaseModel):
    user_id: uuid.UUID
    amount: float
    transaction_type: str


class ExchangeServiceSchema(BaseModel):
    id: uuid.UUID
    type: str
    rate: float
    last_update: datetime

    class Config:
        from_attributes = True


class UpdateExchangeRateSchema(BaseModel):
    rate: float

    @field_validator('rate')
    def validate_rate(cls, rate):
        if rate <= 0:
            raise ValueError("Exchange rate must be positive")
        return rate