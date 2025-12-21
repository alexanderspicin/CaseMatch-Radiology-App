import uuid
from typing import Optional

from fastapi import Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.session import Session

from src.db.database import get_db
from src.models.balance import Transaction, Balance
from src.models.enums import Status, TransactionType
from src.schemas.balance import CreateTransactionSchema, TransactionSchema
from src.models.user import User
from src.services.exchange_service import convert_rub_to_tokens


def create_transaction(transaction_data: CreateTransactionSchema, db: Session = Depends(get_db)) -> TransactionSchema:

    try:
        user = db.query(User).filter(User.id == transaction_data.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        db_transaction = Transaction(
            user_id=transaction_data.user_id,
            amount=transaction_data.amount,
            transaction_type=transaction_data.transaction_type,
            transaction_status=Status.PROCESSING
        )

        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)

        result = TransactionSchema.model_validate(db_transaction)
        return result

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


def process_transaction(transaction_id: uuid.UUID, db: Session = Depends(get_db)) -> TransactionSchema:
    """
    Обрабатывает транзакцию: обновляет баланс пользователя и статус транзакции
    """

    try:
        db_transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not db_transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")

        if db_transaction.transaction_status != Status.PROCESSING:
            return TransactionSchema.model_validate(db_transaction)

        user_balance = db.query(Balance).filter(Balance.user_id == db_transaction.user_id).first()
        if not user_balance:
            db_transaction.transaction_status = Status.FAILED
            db.commit()
            raise HTTPException(status_code=404, detail="User balance not found")

        if db_transaction.transaction_type == TransactionType.CREDIT:
            amount_in_rub = float(db_transaction.amount)
            tokens_to_add = convert_rub_to_tokens(amount_in_rub, db)
            user_balance.amount += tokens_to_add

        elif db_transaction.transaction_type == TransactionType.DEBIT:
            if user_balance.amount < db_transaction.amount:
                db_transaction.transaction_status = Status.FAILED
                db.commit()
                db.refresh(db_transaction)
                return TransactionSchema.model_validate(db_transaction)

            user_balance.amount -= db_transaction.amount

        else:
            db_transaction.transaction_status = Status.FAILED
            db.commit()
            raise HTTPException(status_code=400, detail="Invalid transaction type")

        db_transaction.transaction_status = Status.DONE

        db.commit()
        db.refresh(db_transaction)
        db.refresh(user_balance)

        return TransactionSchema.model_validate(db_transaction)

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


def get_transaction_by_id(transaction_id: uuid.UUID, db: Session = Depends(get_db)) -> Optional[TransactionSchema]:
    """
    Получает транзакцию по ID
    """

    try:
        db_transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not db_transaction:
            return None

        return TransactionSchema.model_validate(db_transaction)

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Database error occurred")


def get_user_transactions(user_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TransactionSchema]:
    """
    Получает все транзакции пользователя
    """

    try:
        db_transactions = db.query(Transaction).filter(Transaction.user_id == user_id).all()
        return [TransactionSchema.model_validate(transaction) for transaction in db_transactions]

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Database error occurred")