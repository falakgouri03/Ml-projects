import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401
from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.predictions import router as predictions_router
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import engine
from app.services.model_service import model_service

logging.basicConfig(level=logging.INFO)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="CardioSense Pro backend with JWT auth, ML inference, and per-user prediction history.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    model_service.reload()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "CardioSense Pro API is running"}


app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(predictions_router, prefix=settings.api_v1_prefix)
