from datetime import datetime
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from src.schemas.predict_schema import HealthResponse, PredictResponseSchema
from src.services.predict_service import model_manager
from src.services.auth_service import get_current_user
from src.models.user import User

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
async def predict(
        image: UploadFile = File(...),
        threshold: float = 0.5,
        save_to_db: bool = False,
        current_user: User = Depends(get_current_user)
):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Use JPEG or PNG"
        )
    try:
        contents = await image.read()
        predictions = model_manager.predict_from_bytes(
            contents,
            threshold=threshold,
            save_to_db=save_to_db,
            user_id=str(current_user.id),
            metadata={
                "user_email": current_user.email,
                "filename": image.filename
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed with error: {e}"
        )
    return PredictResponseSchema(**predictions)


@router.post("/search-similar", tags=["Predict"])
async def search_similar(
        image: UploadFile = File(...),
        limit: int = 5,
        score_threshold: float = 0.7,
        current_user: User = Depends(get_current_user)
):
    """
    Ищет похожие случаи в базе данных по загруженному изображению
    """
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Use JPEG or PNG"
        )

    try:
        from PIL import Image
        import io
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        results = model_manager.search_similar(
            img,
            limit=limit,
            score_threshold=score_threshold
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed with error: {e}"
        )