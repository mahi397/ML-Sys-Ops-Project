from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.config import APP_DIR, APP_PREFIX, get_public_url

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def render_template(
    request: Request,
    template_name: str,
    user: dict[str, Any] | None,
    **context: Any,
) -> HTMLResponse:
    base_context = {
        "request": request,
        "app_prefix": APP_PREFIX,
        "public_url": get_public_url(),
        "user": user,
    }
    base_context.update(context)
    return templates.TemplateResponse(template_name, base_context)
