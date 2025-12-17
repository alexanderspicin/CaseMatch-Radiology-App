from fastapi import FastAPI
from sqlalchemy import text

from src.api import users, predict
from src.db.database import engine

app = FastAPI(title="CaseMatch Radiology API")

app.include_router(users.router)
app.include_router(predict.router)
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