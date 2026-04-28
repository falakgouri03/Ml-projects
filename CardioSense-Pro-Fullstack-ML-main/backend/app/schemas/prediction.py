from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=0, le=300)
    chol: int = Field(ge=0, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=0, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=0, le=3)


class PredictionInferenceResponse(BaseModel):
    prediction_id: int
    probability: float
    prediction: int
    risk_level: str
    risk_label: str
    recommendation: str
    risk_factors: list[str]
    model_version: str
    created_at: datetime


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    probability: float
    prediction_label: int
    risk_level: str
    risk_label: str
    model_version: str
    input_payload: dict[str, Any]
    created_at: datetime
