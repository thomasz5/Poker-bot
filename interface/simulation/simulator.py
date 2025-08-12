"""Simulation engine wrapper for running poker matches in-memory.

Scalable design notes:
- Stateless API surface; state kept in a registry (pluggable backend later)
- Clean separation between transport (API) and domain (GameState/Action)
- All IO-free so it can be reused by CLI, API, or tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from poker_engine.game_state import GameState, GamePhase, Player as EnginePlayer
from poker_engine.actions import Action, ActionType


class SimulationError(Exception):
    pass


class ActionString(str, Enum):
    fold = "fold"
    check = "check"
    call = "call"
    bet = "bet"
    raise_ = "raise"  # name can't be 'raise' in python identifier context
    all_in = "all_in"

    @staticmethod
    def to_action_type(name: str) -> ActionType:
        if name == "raise":
            return ActionType.RAISE
        return ActionType(name)


@dataclass
class PlayerSpec:
    id: str
    name: str
    stack: float


@dataclass
class SimulationState:
    match_id: str
    game: GameState
    players: Dict[str, EnginePlayer] = field(default_factory=dict)

    def summarize(self) -> Dict:
        current = self.game.get_current_player()
        return {
            "match_id": self.match_id,
            "phase": self.game.phase.value,
            "pot": self.game.main_pot.amount,
            "current_bet": self.game.current_bet,
            "min_raise": self.game.min_raise,
            "dealer_position": self.game.dealer_position,
            "board": [str(c) for c in self.game.board],
            "current_player": current.id if current else None,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "stack": p.stack,
                    "current_bet": p.current_bet,
                    "is_all_in": p.is_all_in,
                    "has_folded": p.has_folded,
                }
                for p in self.game.active_players
            ],
            "legal_actions": [a.value for a in self.game.get_legal_actions()],
        }


class SimulationRegistry:
    """Simple in-memory registry. Replace with DB/Redis later."""

    def __init__(self) -> None:
        self._by_id: Dict[str, SimulationState] = {}

    def create(self, small_blind: float = 1.0, big_blind: float = 2.0) -> SimulationState:
        match_id = str(uuid4())
        state = SimulationState(match_id=match_id, game=GameState(small_blind, big_blind))
        self._by_id[match_id] = state
        return state

    def get(self, match_id: str) -> SimulationState:
        if match_id not in self._by_id:
            raise SimulationError(f"Match not found: {match_id}")
        return self._by_id[match_id]


registry = SimulationRegistry()


def create_match(players: List[PlayerSpec], small_blind: float = 1.0, big_blind: float = 2.0) -> Dict:
    state = registry.create(small_blind, big_blind)
    # Add players
    for spec in players:
        p = state.game.add_player(spec.id, spec.name, float(spec.stack))
        state.players[spec.id] = p
    # Initialize first hand
    state.game.start_new_hand()
    return state.summarize()


def start_new_hand(match_id: str) -> Dict:
    state = registry.get(match_id)
    state.game.start_new_hand()
    return state.summarize()


def get_state(match_id: str) -> Dict:
    return registry.get(match_id).summarize()


def get_legal_actions(match_id: str) -> List[str]:
    state = registry.get(match_id)
    return [a.value for a in state.game.get_legal_actions()]


def apply_action(match_id: str, action_name: str, amount: Optional[float] = None) -> Dict:
    state = registry.get(match_id)
    atype = ActionString.to_action_type(action_name)
    action = Action(atype, amount=float(amount) if amount is not None else 0.0)
    ok = state.game.process_action(action)
    if not ok:
        raise SimulationError("Illegal or failed action")
    return state.summarize()


