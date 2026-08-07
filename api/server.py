"""
EXAI-ResumeIntel: FastAPI application factory
==============================================

Boots the FastAPI application, wires CORS, registers routes, and exposes
the entry point for uvicorn.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8765 --reload

Module: api.server
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import logging
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router as api_router
from core.constants import APP_NAME, APP_VERSION, AUTHOR, AUTHOR_GITHUB

# ─── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("API_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("exai.server")


def _get_cors_origins() -> List[str]:
    """Read CORS_ORIGINS from the environment or return sensible defaults."""
    raw = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:8080,http://localhost:8501,http://localhost:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=APP_NAME,
        description=(
            "Explainable AI Framework for Automated Resume Analysis Using "
            "Shapley Values, LIME, and Domain Skill Ontology.\n\n"
            f"**Author:** [{AUTHOR}]({AUTHOR_GITHUB})"
        ),
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Meta", "description": "Health and metadata endpoints"},
            {"name": "Analysis", "description": "Resume analysis endpoints"},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/", tags=["Meta"])
    def root() -> JSONResponse:
        """Root endpoint. Points clients to the interactive documentation."""
        return JSONResponse({
            "name": APP_NAME,
            "version": APP_VERSION,
            "author": AUTHOR,
            "github": AUTHOR_GITHUB,
            "docs": "/docs",
            "redoc": "/redoc",
            "endpoints": ["/health", "/roles", "/analyze"],
        })

    log.info("%s v%s ready", APP_NAME, APP_VERSION)
    return app


app = create_app()


def main() -> None:
    """Entry point for the ``exai-serve`` console script."""
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8765"))
    reload = os.environ.get("API_RELOAD", "false").lower() == "true"

    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
