# Simulation Service

Minimal simulation engine and API for running poker matches in-memory.

## Python usage

```python
from interface.simulation.simulator import create_match, PlayerSpec, apply_action, get_state

state = create_match([
    PlayerSpec(id="p1", name="Alice", stack=100.0),
    PlayerSpec(id="p2", name="Bob", stack=100.0),
])
match_id = state["match_id"]
print(get_state(match_id))

# First action (depends on position); attempt a call or check
state = apply_action(match_id, "check")  # or "call", "raise", etc.
print(state["legal_actions"])
```

## FastAPI

Run API (example):

```bash
uvicorn interface.simulation.api:app --reload
```

Endpoints:
- POST /match { players: [{id,name,stack}], small_blind, big_blind }
- POST /match/{id}/hand
- GET /match/{id}
- GET /match/{id}/legal_actions
- POST /match/{id}/step { action, amount? }

Later: move state to Redis/DB, add AI decision endpoints.

