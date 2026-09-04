"""FastAPI application serving the warehouse dashboard API.

Run it with::

    uvicorn backend.main:app --reload

The API is deliberately read-mostly: it exposes the scenarios, the recorded
episodes and the evaluation results, plus a single endpoint that runs one
episode on demand. Training happens offline through ``rl_agent/train.py``,
never through an HTTP request.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

#: Comma-separated list of allowed browser origins. The default covers local
#: development; set ALLOWED_ORIGINS to the deployed dashboard URL in production.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Warehouse RL API",
        version="0.1.0",
        description=(
            "Backend for the AI-Based Warehouse Automation System "
            "(Reinforcement Learning for intelligent robot navigation)."
        ),
    )
    origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/")
    def index() -> dict[str, str]:
        return {
            "service": "warehouse-rl",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
