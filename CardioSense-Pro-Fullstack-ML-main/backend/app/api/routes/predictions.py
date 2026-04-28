from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.prediction import (
    HeartFeatures,
    PredictionHistoryItem,
    PredictionInferenceResponse,
)
from app.services.model_service import model_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionInferenceResponse)
def create_prediction(
    payload: HeartFeatures,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PredictionInferenceResponse:
    result = model_service.predict(payload.model_dump())

    record = Prediction(
        user_id=current_user.id,
        input_payload=payload.model_dump(),
        probability=result["probability"],
        prediction_label=result["prediction"],
        risk_level=result["risk_level"],
        risk_label=result["risk_label"],
        model_version=result["model_version"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionInferenceResponse(
        prediction_id=record.id,
        probability=record.probability,
        prediction=record.prediction_label,
        risk_level=record.risk_level,
        risk_label=record.risk_label,
        recommendation=result["recommendation"],
        risk_factors=result["risk_factors"],
        model_version=record.model_version,
        created_at=record.created_at,
    )


@router.get("/history", response_model=list[PredictionHistoryItem])
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PredictionHistoryItem]:
    stmt = (
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    records = db.scalars(stmt).all()
    return list(records)
