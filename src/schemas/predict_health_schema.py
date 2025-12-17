from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Модель ответа healthcheck"""
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str