from datetime import datetime, timezone

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.session import Session

from src.api import users, predict, balance
from src.db.database import engine, SessionLocal
from src.models import ExchangeService


def init_exchange_service():
    """
    Инициализирует единственную запись ExchangeService в базе данных
    если она еще не существует
    """
    db: Session = SessionLocal()
    try:
        # Проверяем, существует ли уже запись
        existing_service = db.query(ExchangeService).first()

        if existing_service is None:
            # Создаем единственную запись с курсом по умолчанию 1.2
            exchange_service = ExchangeService(
                type="default",
                rate=1.2,
                last_update=datetime.now(tz=timezone.utc)
            )
            db.add(exchange_service)
            db.commit()
        else:
            pass
    except IntegrityError as e:
        db.rollback()
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
app = FastAPI(title="CaseMatch Radiology API")
init_exchange_service()
app.include_router(users.router)
app.include_router(predict.router)
app.include_router(balance.router)
@app.get("/")
def root():
    return {"message": "CaseMatch API is running"}


@app.get("/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}