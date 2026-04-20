from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.middleware.sessions import SessionMiddleware

from auth.api import router as auth_api_router
from auth.views import router as auth_views_router
from core.config import APP_NAME, get_https_only, get_session_secret
from summaries.api import router as summaries_api_router
from summaries.views import router as summaries_views_router


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME)
    app.add_middleware(
        SessionMiddleware,
        secret_key=get_session_secret(),
        https_only=get_https_only(),
        same_site="lax",
    )

    app.include_router(auth_api_router)
    app.include_router(auth_views_router)
    app.include_router(summaries_api_router)
    app.include_router(summaries_views_router)

    @app.get("/health", response_class=PlainTextResponse)
    def health() -> str:
        return "ok"

    return app


app = create_app()
