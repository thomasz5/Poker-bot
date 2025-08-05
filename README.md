# Professional Poker Bot Project

A sophisticated poker bot that learns from professional players using machine learning and game theory optimal (GTO) strategies.

## 🎯 Project Goals

- Create a trainable poker bot that emulates professional player strategies
- Implement state-of-the-art machine learning techniques for decision making
- Integrate GTO solver foundations with adaptive opponent modeling
- Achieve competitive performance against skilled human players

## 📁 Project Structure

```
poker_bot/
├── PROJECT_DESIGN.md           # Comprehensive design document
├── IMPLEMENTATION_GUIDE.md     # Technical implementation details
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── poker_engine/              # Core poker game logic
│   ├── hand_evaluator.py      # Fast hand strength calculation
│   ├── game_state.py          # Game state management
│   ├── actions.py             # Action types and validation
│   └── rules.py               # Poker rules enforcement
├── ai_core/                   # AI decision making
│   ├── neural_networks/       # Deep learning models
│   ├── gto_solver/           # Game theory optimal strategies
│   ├── opponent_modeling/    # Player profiling and adaptation
│   └── decision_engine.py    # Main decision logic
├── training/                  # ML training pipeline
│   ├── data_processor.py     # Data preprocessing
│   ├── feature_extractor.py  # Feature engineering
│   ├── trainers/             # Training algorithms
│   └── rl_environment.py     # Reinforcement learning env
├── data/                      # Data management
│   ├── parsers/              # Hand history parsers
│   ├── storage/              # Database and file storage
│   └── models/               # Trained model storage
├── interface/                 # User interfaces and APIs
│   ├── web_dashboard/        # Web-based monitoring
│   ├── simulation/           # Testing environment
│   └── config/               # Interface configuration
├── tests/                     # Test suite
└── scripts/                   # Utility scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd "Poker Bot"

# Create virtual environment
python3.9 -m venv poker_bot_env
source poker_bot_env/bin/activate  # On Windows: poker_bot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Hand Evaluator

```bash
# Test the basic poker engine
python poker_engine/hand_evaluator.py
```

Expected output:
```
Royal flush: HandRank.ROYAL_FLUSH, kickers: [12]
Pair of Aces: HandRank.PAIR, kickers: [12, 12, 11, 9, 8]
AA vs KK comparison: 1
AA hand strength on 234 board: 0.300
```

### 3. Test Actions System

```bash
# Test the action validation system
python poker_engine/actions.py
```

### 4. Configuration

Copy and modify the configuration:
```bash
# Edit config.py to match your environment
# Set database URLs, API keys, and training parameters
```

## 📋 Development Roadmap

### Phase 1: Foundation (Weeks 1-4) ✅
- [x] Project structure setup
- [x] Basic hand evaluator implementation
- [x] Action system and validation
- [ ] Game state management
- [ ] Database schema and data pipeline
- [ ] Unit tests for core components

### Phase 2: Basic AI (Weeks 5-8)
- [ ] GTO solver integration
- [ ] Neural network architecture
- [ ] Feature extraction pipeline
- [ ] Initial training on professional data
- [ ] Basic decision engine

### Phase 3: Advanced Learning (Weeks 9-12)
- [ ] Opponent modeling system
- [ ] Reinforcement learning environment
- [ ] Self-play training
- [ ] Professional data integration
- [ ] Performance optimization

### Phase 4: Deployment (Weeks 13-16)
- [ ] Web dashboard interface
- [ ] Real-time performance monitoring
- [ ] Model compression and optimization
- [ ] Statistical validation
- [ ] Documentation and deployment

## 🛠 Core Technologies

- **Language**: Python 3.9+
- **ML Framework**: PyTorch with Lightning
- **Database**: PostgreSQL + Redis
- **Web Framework**: FastAPI
- **Visualization**: Plotly + Weights & Biases
- **Testing**: Pytest

## 📊 Key Features

### Hand Evaluation Engine
- Fast 7-card hand evaluation
- Support for all poker variants
- Hand strength and equity calculations
- Optimized for millions of evaluations per second

### AI Decision Making
- Transformer-based neural networks
- GTO solver integration
- Opponent modeling and adaptation
- Real-time decision optimization

### Training Pipeline
- Imitation learning from professional players
- Reinforcement learning through self-play
- Multi-stage curriculum learning
- Distributed training support

### Data Management
- Hand history parsing (PokerStars, GGPoker formats)
- Professional player database
- Feature extraction and storage
- Model versioning and checkpoints

## 🎮 Current Capabilities

The current implementation provides:

1. **Hand Evaluator**: Complete poker hand ranking system
   - Supports 5-7 card evaluation
   - All standard poker hands (high card to royal flush)
   - Hand comparison and equity estimation

2. **Action System**: Comprehensive action validation
   - All poker actions (fold, check, call, bet, raise, all-in)
   - Legal action determination
   - Bet sizing recommendations

3. **Configuration Management**: Flexible settings system
   - Database connections
   - Training parameters
   - Model architecture settings
   - Feature engineering configuration

## 🧪 Testing

Run the test suite:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=poker_engine --cov-report=html
```

## 📈 Performance Targets

- **Hand Evaluation**: 10M+ hands/second
- **Neural Network Inference**: <50ms per decision
- **Memory Usage**: <4GB for complete system
- **Win Rate**: 2+ BB/100 against intermediate players

## 🗂 Data Requirements

### Professional Hand Histories
- **Volume**: 10M+ hands from high-stakes games
- **Sources**: PokerTracker exports, tournament databases
- **Formats**: PokerStars, GGPoker, Hold'em Manager

### Training Data
- **Game States**: 100M+ decision points
- **Features**: 500+ per decision
- **Storage**: ~500GB for complete dataset

## 🔧 Next Steps

1. **Implement Game State Manager** (`poker_engine/game_state.py`)
   - Track players, positions, and betting rounds
   - Handle pot calculations and side pots
   - Manage game flow and street transitions

2. **Create Hand History Parser** (`data/parsers/hand_history_parser.py`)
   - Parse major poker site formats
   - Extract decision points and outcomes
   - Generate training data from professional games

3. **Build Feature Extractor** (`training/feature_extractor.py`)
   - Convert game states to neural network inputs
   - Implement position, stack, and board features
   - Add opponent modeling features

4. **Design Neural Network** (`ai_core/neural_networks/decision_network.py`)
   - Implement transformer architecture
   - Add multi-head decision outputs
   - Include value estimation

5. **Set Up Training Pipeline** (`training/trainers/`)
   - Imitation learning trainer
   - Reinforcement learning environment
   - Model evaluation and validation

## 📚 Resources

- **Design Document**: `PROJECT_DESIGN.md` - Complete architecture overview
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md` - Technical details and code examples
- **Configuration**: `config.py` - All configurable parameters

## 🤝 Contributing

This is a research and educational project. Key areas for contribution:
- Advanced neural network architectures
- GTO solver integration
- Opponent modeling techniques
- Performance optimization
- Testing and validation

## ⚖️ Legal & Ethical Considerations

This bot is designed for:
- Educational and research purposes
- Strategy analysis and improvement
- Simulation and testing environments

**Important**: Ensure compliance with poker platform terms of service and local regulations before any practical application.

## 🎯 Success Metrics

- **Technical**: Sub-second decisions, 95%+ uptime
- **Performance**: Positive win rate against skilled players
- **Adaptation**: Quick adjustment to new opponents (<1000 hands)
- **Scalability**: Support for multiple game variants

---

Ready to build a world-class poker bot? Start with the foundation and work through each phase systematically. The modular architecture ensures you can test and validate each component independently while building toward the complete system. 