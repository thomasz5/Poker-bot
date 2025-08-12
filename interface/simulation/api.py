"""FastAPI app exposing simulation endpoints.

You can mount this in a server process. Designed to be scalable:
- stateless handlers
- clean DTOs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from .simulator import create_match, start_new_hand, get_state, get_legal_actions, apply_action, PlayerSpec


app = FastAPI(title="Poker Simulation API", version="0.1.0")


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


