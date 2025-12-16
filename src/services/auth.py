from datetime import datetime, timezone, timedelta

from src.core.config import settings
from jose import jwt


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        settings.secret_key.get_secret_value(),  # Извлекаем секрет
        algorithm=settings.algorithm,
    )