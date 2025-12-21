import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, UUID, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import relationship

from src.db.database import Base
from src.models.enums import Status, TransactionType


class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.now(tz=timezone.utc), nullable=False)
    user_id = Column(UUID, ForeignKey('users.id'), nullable=False)
    amount = Column(Float, nullable=False)
    transaction_status = Column(ENUM(Status), nullable=False)
    transaction_type = Column(ENUM(TransactionType), nullable=False)
    user = relationship("User", back_populates="transactions")


class Balance(Base):
    __tablename__ = 'balance'
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID, ForeignKey('users.id'), nullable=False)
    amount = Column(Float, default=100, nullable=False)
    user = relationship("User", back_populates="balance")


class ExchangeService(Base):
    __tablename__ = 'exchange_service'
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False, unique=True)
    rate = Column(Float, nullable=False, default=1.2)
    last_update = Column(DateTime, default=datetime.now(tz=timezone.utc), nullable=False)