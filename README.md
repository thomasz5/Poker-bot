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