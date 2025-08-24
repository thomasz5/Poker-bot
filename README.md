# My Poker Bot Project

I'm building a poker bot that learns how to play like the pros (and maybe win me some money 🤑). It uses machine learning and game theory to make smart decisions at the poker table.


## Getting Started

Want to try it out? Here's how to get everything running on your computer.

### Step 1: 

First, let's get your computer ready:

```bash
# Go to the project folder
cd "Poker Bot"

# Create a clean Python environment (so nothing conflicts)
python3.9 -m venv poker_bot_env
source poker_bot_env/bin/activate  # If you're on Windows: poker_bot_env\Scripts\activate

# Install all the stuff we need
pip install -r requirements.txt
```

### Step 2: 

Let's make sure the poker engine works:

```bash
# Run a quick test
python poker_engine/hand_evaluator.py
```

You should see something like this:
```
Royal flush: HandRank.ROYAL_FLUSH, kickers: [12]
Pair of Aces: HandRank.PAIR, kickers: [12, 12, 11, 9, 8]
AA vs KK comparison: 1
AA hand strength on 234 board: 0.300
```

### Step 3:

Now let's check if the betting logic works:

```bash
# Test how the bot handles bets and raises
python poker_engine/actions.py
```

### Step 4: 

You'll want to customize things for your setup:

```bash
# Open config.py and change settings to match your computer
# Things like database connections and training settings
```

This is a research and educational project. Key areas for contribution:
- Advanced neural network architectures
- GTO solver integration
- Opponent modeling techniques
- Performance optimization
- Testing and validation

---

## Docker

### Build images
```bash
docker compose build
```

### Run API
```bash
docker compose up api
# API at http://localhost:8000 (dashboard at /static)
```

Environment options:
- `SIM_STORE=redis` to back matches by Redis (compose starts `redis` service)

### Run trainer
```bash
docker compose run --rm trainer
```

### Develop interactively
```bash
docker compose up --build
```

### Create a starter dataset
```bash
# Uses sample hand histories under data/sample_hands/
docker compose run --rm dataset
# Artifacts written to training_data/dataset.npz and training_data/manifest.json
```

### MLflow tracking
```bash
docker compose up -d mlflow
export USE_MLFLOW=1
docker compose run --rm -e USE_MLFLOW=1 trainer
# Open http://localhost:5000 to view runs
```
