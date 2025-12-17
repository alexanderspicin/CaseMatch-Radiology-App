from datetime import datetime

from fastapi import APIRouter

from src.schemas.predict_health_schema import HealthResponse
from src.services.predict_service import model_manager

model_manager.load_model()

router = APIRouter(prefix="/predict", tags=["predicts"])


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        model_path=str(model_manager.model_path),
        timestamp=datetime.now().isoformat()
    )