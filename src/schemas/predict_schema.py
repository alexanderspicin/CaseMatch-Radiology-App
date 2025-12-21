from pydantic import BaseModel


class PredictObjectSchema(BaseModel):
    label: str
    probability: float
    detected: bool


class PredictResponseSchema(BaseModel):
    detected: list[str]
    predictions: list[PredictObjectSchema]
    threshold: float


class HealthResponse(BaseModel):
    """Модель ответа healthcheck"""
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str