from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm.session import Session

from src.db.database import get_db
from src.models import User
from src.schemas.balance import TransactionSchema, CreateTransactionSchema
from src.services.auth_service import JWTBearer, get_current_user
from src.services.transaction_service import create_transaction, process_transaction

router = APIRouter(prefix="/balance", tags=["balance"])


@router.get("/credit", response_model=TransactionSchema)
def create_credit(
        amount: float,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero.")

    transaction_schema = CreateTransactionSchema(
        user_id=current_user.id,
        transaction_type='CREDIT',
        amount=amount
    )
    db_transaction = create_transaction(transaction_schema, db=db)
    process_transaction(db_transaction.id, db=db)
    return db_transaction