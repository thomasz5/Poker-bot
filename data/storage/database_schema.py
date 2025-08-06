"""
Database Schema for Poker Bot

This module defines the database schema for storing:
- Game sessions and hands
- Player actions and decisions  
- Training data for the AI
- Performance metrics

Uses SQLAlchemy for ORM and supports both SQLite (development) and PostgreSQL (production)
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()


class GameSession(Base):
    """Represents a complete game session with multiple hands"""
    __tablename__ = 'game_sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    session_type = Column(String, nullable=False)  # 'cash', 'tournament', 'simulation'
    game_variant = Column(String, default='NLHE')  # No Limit Hold'em
    table_size = Column(Integer, default=6)  # 6-max, 9-max, heads-up
    
    # Blinds and stakes
    small_blind = Column(Float, nullable=False)
    big_blind = Column(Float, nullable=False)
    ante = Column(Float, default=0.0)
    
    # Session stats
    total_hands = Column(Integer, default=0)
    duration_minutes = Column(Integer)
    ended_at = Column(DateTime)
    
    # Relationships
    hands = relationship("Hand", back_populates="session")
    players = relationship("SessionPlayer", back_populates="session")


class SessionPlayer(Base):
    """Player information for a specific session"""
    __tablename__ = 'session_players'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey('game_sessions.id'), nullable=False)
    player_name = Column(String, nullable=False)
    is_bot = Column(Boolean, default=False)
    bot_version = Column(String)  # If this is a bot player
    
    # Position and starting info
    seat_number = Column(Integer)
    starting_stack = Column(Float)
    final_stack = Column(Float)
    
    # Session performance
    hands_played = Column(Integer, default=0)
    vpip_percentage = Column(Float)  # Voluntarily Put in Pot
    pfr_percentage = Column(Float)   # Pre-Flop Raise
    aggression_factor = Column(Float)
    
    # Relationships
    session = relationship("GameSession", back_populates="players")
    hand_players = relationship("HandPlayer", back_populates="session_player")


class Hand(Base):
    """Individual poker hand data"""
    __tablename__ = 'hands'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey('game_sessions.id'), nullable=False)
    hand_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Hand setup
    dealer_position = Column(Integer)  # Seat number of dealer
    small_blind_seat = Column(Integer)
    big_blind_seat = Column(Integer)
    
    # Board cards (stored as JSON array of card strings)
    flop_cards = Column(JSON)  # ["As", "Kh", "Qd"]
    turn_card = Column(String)
    river_card = Column(String)
    
    # Hand outcome
    winner_seat = Column(Integer)
    winning_hand_type = Column(String)  # "pair", "two_pair", etc.
    total_pot = Column(Float)
    rake = Column(Float, default=0.0)
    
    # Hand duration and stats
    duration_seconds = Column(Integer)
    total_actions = Column(Integer, default=0)
    went_to_showdown = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("GameSession", back_populates="hands")
    players = relationship("HandPlayer", back_populates="hand")
    actions = relationship("PlayerAction", back_populates="hand")


class HandPlayer(Base):
    """Player state and cards for a specific hand"""
    __tablename__ = 'hand_players'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    hand_id = Column(String, ForeignKey('hands.id'), nullable=False)
    session_player_id = Column(String, ForeignKey('session_players.id'), nullable=False)
    
    # Position and stack info
    seat_number = Column(Integer, nullable=False)
    position_name = Column(String)  # "BTN", "SB", "BB", "UTG", etc.
    starting_stack = Column(Float, nullable=False)
    ending_stack = Column(Float, nullable=False)
    
    # Hole cards (encrypted for security)
    hole_card_1 = Column(String)  # "As"
    hole_card_2 = Column(String)  # "Kh"
    
    # Hand outcome for this player
    final_hand_rank = Column(String)  # Best 5-card hand rank
    final_hand_cards = Column(JSON)   # The actual 5 cards used
    won_amount = Column(Float, default=0.0)
    
    # Hand statistics
    vpip = Column(Boolean, default=False)  # Voluntarily put money in pot
    pfr = Column(Boolean, default=False)   # Pre-flop raise
    went_to_showdown = Column(Boolean, default=False)
    folded_to_steal = Column(Boolean, default=False)
    
    # Relationships
    hand = relationship("Hand", back_populates="players")
    session_player = relationship("SessionPlayer", back_populates="hand_players")


class PlayerAction(Base):
    """Individual actions taken by players"""
    __tablename__ = 'player_actions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    hand_id = Column(String, ForeignKey('hands.id'), nullable=False)
    session_player_id = Column(String, ForeignKey('session_players.id'), nullable=False)
    
    # Action details
    action_sequence = Column(Integer, nullable=False)  # Order of action in hand
    street = Column(String, nullable=False)  # "preflop", "flop", "turn", "river"
    action_type = Column(String, nullable=False)  # "fold", "call", "bet", "raise", "check"
    amount = Column(Float, default=0.0)
    
    # Game state when action was taken
    pot_size_before = Column(Float)
    current_bet = Column(Float)
    position_from_dealer = Column(Integer)
    players_in_hand = Column(Integer)
    
    # AI decision data (for training)
    decision_time_ms = Column(Integer)  # How long the decision took
    confidence_score = Column(Float)    # AI confidence in decision (if applicable)
    alternative_actions = Column(JSON)  # Other actions considered
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    hand = relationship("Hand", back_populates="actions")


class TrainingFeature(Base):
    """Extracted features for ML training"""
    __tablename__ = 'training_features'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    action_id = Column(String, ForeignKey('player_actions.id'), nullable=False)
    
    # Feature vectors (stored as JSON for flexibility)
    position_features = Column(JSON)     # Position-based features
    hand_strength_features = Column(JSON) # Hand equity, strength, etc.
    opponent_features = Column(JSON)     # Opponent modeling features
    betting_features = Column(JSON)      # Pot odds, betting patterns
    board_texture_features = Column(JSON) # Board analysis features
    
    # Target values for supervised learning
    target_action = Column(String)       # The actual action taken
    target_amount = Column(Float)        # The actual amount
    hand_outcome_value = Column(Float)   # How much was won/lost
    
    # Meta information
    feature_version = Column(String)     # Version of feature extractor used
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPerformance(Base):
    """Track AI model performance over time"""
    __tablename__ = 'model_performance'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    session_id = Column(String, ForeignKey('game_sessions.id'))
    
    # Performance metrics
    hands_played = Column(Integer)
    bb_per_100_hands = Column(Float)     # Big blinds won per 100 hands
    vpip_percentage = Column(Float)
    pfr_percentage = Column(Float)
    aggression_factor = Column(Float)
    
    # Advanced stats
    showdown_win_rate = Column(Float)
    steal_success_rate = Column(Float)
    fold_to_steal_rate = Column(Float)
    
    # Technical performance
    avg_decision_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    errors_count = Column(Integer, default=0)
    
    # Evaluation period
    evaluation_start = Column(DateTime)
    evaluation_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database management functions
class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, database_url: str = "sqlite:///poker_bot.db"):
        """Initialize database connection"""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
        
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)


# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager("sqlite:///test_poker_bot.db")
    db.create_tables()
    
    print("Database tables created successfully!")
    
    # Test creating a game session
    session = db.get_session()
    try:
        # Create a test game session
        game_session = GameSession(
            session_type="simulation",
            small_blind=1.0,
            big_blind=2.0,
            table_size=6
        )
        session.add(game_session)
        
        session.commit()  # Commit game session first to get ID
        
        # Add test players
        player1 = SessionPlayer(
            session_id=game_session.id,
            player_name="TestBot_v1.0",
            is_bot=True,
            bot_version="1.0",
            seat_number=1,
            starting_stack=100.0
        )
        
        player2 = SessionPlayer(
            session_id=game_session.id,
            player_name="Human_Player",
            is_bot=False,
            seat_number=2,
            starting_stack=100.0
        )
        
        session.add(player1)
        session.add(player2)
        session.commit()
        
        print(f"Created game session: {game_session.id}")
        print(f"Added {len([player1, player2])} players")
        
        # Query test
        sessions = session.query(GameSession).all()
        print(f"Total sessions in database: {len(sessions)}")
        
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()
    
    print("Database test completed!")