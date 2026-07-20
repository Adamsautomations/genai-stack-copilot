"""FastAPI service.

Two answer paths behind one endpoint — `rag` runs the LangGraph pipeline,
`cag` answers from a cached context block — so the comparison is reproducible
by anyone using the deployed app, not just from the writeup.

A per-session spend cap is enforced before the model is ever called. A public
demo that can call a frontier model without a ceiling is a demo that hands a
stranger your billing account.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config import Settings
from src.graph.build import answer as run_rag

log = logging.getLogger("copilot")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
SESSION_COOKIE = "copilot_session"

# Process-local. Fine for a single-instance demo; a multi-instance deployment
# would move this to Redis or a table. Called out rather than pretended away.
_spend_cents: dict[str, float] = {}
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["settings"] = Settings.load()
    log.info("settings loaded; index=%s", _state["settings"].search_index)
    yield
    _state.clear()


app = FastAPI(title="GenAI Stack Copilot", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["rag", "cag"] = "rag"
    source: str | None = None


def _settings() -> Settings:
    settings = _state.get("settings")
    if settings is None:  # pragma: no cover - lifespan always populates this
        raise HTTPException(503, "service not ready")
    return settings


def _session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie(
            SESSION_COOKIE, sid, httponly=True, samesite="lax", max_age=86400
        )
    return sid


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "index": _settings().search_index}


@app.post("/api/ask")
def ask(payload: AskRequest, request: Request, response: Response) -> JSONResponse:
    settings = _settings()
    sid = _session_id(request, response)
    spent = _spend_cents.get(sid, 0.0)

    # Check the ceiling *before* spending, not after.
    if spent >= settings.session_cost_cap_cents:
        return JSONResponse(
            status_code=429,
            content={
                "error": "session_budget_exhausted",
                "message": (
                    "This session has reached its demo spending limit. "
                    "Everything else still works — start a new session to continue."
                ),
                "spent_cents": round(spent, 4),
                "cap_cents": settings.session_cost_cap_cents,
            },
        )

    started = time.perf_counter()
    try:
        if payload.mode == "rag":
            result = run_rag(settings, payload.question, source_filter=payload.source)
        else:
            from src.cag.cag import answer_cag, build_context

            context = _state.get("cag_context")
            if context is None:
                context = build_context(settings)
                _state["cag_context"] = context  # built once, then cached upstream
            result = answer_cag(settings, payload.question, context)
    except Exception:
        log.exception("ask failed")
        raise HTTPException(500, "answer pipeline failed") from None

    result["mode"] = payload.mode
    result.setdefault("latency_s", round(time.perf_counter() - started, 3))

    cost_cents = float(result.get("usage", {}).get("cost_usd", 0.0)) * 100
    _spend_cents[sid] = spent + cost_cents
    result["session"] = {
        "spent_cents": round(_spend_cents[sid], 4),
        "cap_cents": settings.session_cost_cap_cents,
    }

    return JSONResponse(content=result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
