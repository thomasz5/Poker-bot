"""FastAPI app exposing simulation endpoints.

You can mount this in a server process. Designed to be scalable:
- stateless handlers
- clean DTOs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional

from .simulator import (
    create_match,
    start_new_hand,
    get_state,
    get_legal_actions,
    apply_action,
    PlayerSpec,
    configure_store_from_env,
)
from ai_core.neural_networks.action_predictor import PokerAI


configure_store_from_env()
app = FastAPI(title="Poker Simulation API", version="0.1.0")

# Serve static dashboard under /static and redirect root
app.mount(
    "/static",
    StaticFiles(directory="interface/web_dashboard/static"),
    name="static",
)


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/static/index.html")


class PlayerIn(BaseModel):
    id: str
    name: str
    stack: float = Field(gt=0)


class CreateMatchIn(BaseModel):
    players: List[PlayerIn]
    small_blind: float = 1.0
    big_blind: float = 2.0


class StepIn(BaseModel):
    action: str
    amount: Optional[float] = None


@app.post("/match", summary="Create match")
def api_create_match(payload: CreateMatchIn):
    try:
        return create_match(
            players=[PlayerSpec(**p.model_dump()) for p in payload.players],
            small_blind=payload.small_blind,
            big_blind=payload.big_blind,
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/match/{match_id}/hand", summary="Start new hand")
def api_new_hand(match_id: str):
    try:
        return start_new_hand(match_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/match/{match_id}", summary="Get state")
def api_get_state(match_id: str):
    try:
        return get_state(match_id)
    except Exception as e:
        raise HTTPException(404, str(e))


@app.get("/match/{match_id}/legal_actions", summary="Get legal actions")
def api_legal_actions(match_id: str):
    try:
        return {"legal_actions": get_legal_actions(match_id)}
    except Exception as e:
        raise HTTPException(404, str(e))


@app.post("/match/{match_id}/step", summary="Apply action")
def api_step(match_id: str, payload: StepIn):
    try:
        return apply_action(match_id, payload.action, payload.amount)
    except Exception as e:
        raise HTTPException(400, str(e))


# ---- AI Endpoints ----

class DecisionIn(BaseModel):
    features: list[float]
    legal_actions: Optional[list[str]] = None


ai_singleton: Optional[PokerAI] = None


def get_ai() -> PokerAI:
    global ai_singleton
    if ai_singleton is None:
        ai_singleton = PokerAI()
    return ai_singleton


@app.post("/ai/decision", summary="Get AI decision")
def ai_decision(payload: DecisionIn):
    try:
        ai = get_ai()
        return ai.make_decision(payload.features, payload.legal_actions)
    except Exception as e:
        raise HTTPException(400, str(e))


class AutoStepOut(BaseModel):
    state: dict
    decision: dict


@app.post("/match/{match_id}/auto_step", response_model=AutoStepOut, summary="AI chooses and applies an action")
def auto_step(match_id: str):
    try:
        state = get_state(match_id)
        legal = get_legal_actions(match_id)
        # Placeholder features; later wire from real extractor
        features = [0.0] * 110
        ai = get_ai()
        decision = ai.make_decision(features, legal)
        # Apply
        amount = decision.get("amount", 0.0)
        new_state = apply_action(match_id, decision["action"], amount)
        return {"state": new_state, "decision": decision}
    except Exception as e:
        raise HTTPException(400, str(e))


