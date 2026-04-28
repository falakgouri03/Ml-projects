from fastapi import APIRouter

from app.services.model_service import model_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": model_service.is_model_loaded,
        "model_version": model_service.model_version,
    }
