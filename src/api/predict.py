from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException

from src.schemas.predict_schema import HealthResponse, PredictResponseSchema
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


@router.post("/predict", response_model=PredictResponseSchema, tags=["Predict"])
async def predict(image: UploadFile = File(...), threshold: float = 0.5, save_to_examples: bool = False):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Use JPEG or PNG"
        )
    try:
        contents = await image.read()
        predictions = model_manager.predict_from_bytes(contents, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed with error: {e}")
    return PredictResponseSchema(**predictions)
