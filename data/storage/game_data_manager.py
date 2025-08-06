"""
Game Data Manager

High-level interface for storing and retrieving poker game data.
Handles the complexity of the database operations so other parts 
of the bot can easily save and load game information.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
try:
    from .database_schema import (
        DatabaseManager, GameSession, SessionPlayer, Hand, 
        HandPlayer, PlayerAction, TrainingFeature, ModelPerformance
    )
except ImportError:
    # For standalone testing
    from database_schema import (
        DatabaseManager, GameSession, SessionPlayer, Hand, 
        HandPlayer, PlayerAction, TrainingFeature, ModelPerformance
    )


class GameDataManager:
    """Manages saving and loading poker game data"""
    
    def __init__(self, database_url: str = "sqlite:///poker_bot.db"):
        self.db = DatabaseManager(database_url)
        self.db.create_tables()
    
    def create_game_session(
        self, 
        session_type: str = "simulation",
        small_blind: float = 1.0,
        big_blind: float = 2.0,
        table_size: int = 6,
        game_variant: str = "NLHE"
    ) -> str:
        """Create a new game session and return its ID"""
        session = self.db.get_session()
        try:
            game_session = GameSession(
                session_type=session_type,
                small_blind=small_blind,
                big_blind=big_blind,
                table_size=table_size,
                game_variant=game_variant
            )
            session.add(game_session)
            session.commit()
            return game_session.id
        finally:
            session.close()
    
    def add_player_to_session(
        self,
        session_id: str,
        player_name: str,
        seat_number: int,
        starting_stack: float,
        is_bot: bool = False,
        bot_version: str = None
    ) -> str:
        """Add a player to a game session"""
        session = self.db.get_session()
        try:
            player = SessionPlayer(
                session_id=session_id,
                player_name=player_name,
                seat_number=seat_number,
                starting_stack=starting_stack,
                is_bot=is_bot,
                bot_version=bot_version
            )
            session.add(player)
            session.commit()
            return player.id
        finally:
            session.close()
    
    def start_new_hand(
        self,
        session_id: str,
        hand_number: int,
        dealer_position: int,
        small_blind_seat: int,
        big_blind_seat: int
    ) -> str:
        """Start a new hand and return its ID"""
        session = self.db.get_session()
        try:
            hand = Hand(
                session_id=session_id,
                hand_number=hand_number,
                dealer_position=dealer_position,
                small_blind_seat=small_blind_seat,
                big_blind_seat=big_blind_seat
            )
            session.add(hand)
            session.commit()
            return hand.id
        finally:
            session.close()
    
    def add_hand_player(
        self,
        hand_id: str,
        session_player_id: str,
        seat_number: int,
        position_name: str,
        starting_stack: float,
        hole_cards: List[str] = None
    ) -> str:
        """Add a player's information for a specific hand"""
        session = self.db.get_session()
        try:
            hand_player = HandPlayer(
                hand_id=hand_id,
                session_player_id=session_player_id,
                seat_number=seat_number,
                position_name=position_name,
                starting_stack=starting_stack,
                ending_stack=starting_stack,  # Will be updated later
                hole_card_1=hole_cards[0] if hole_cards and len(hole_cards) > 0 else None,
                hole_card_2=hole_cards[1] if hole_cards and len(hole_cards) > 1 else None
            )
            session.add(hand_player)
            session.commit()
            return hand_player.id
        finally:
            session.close()
    
    def record_action(
        self,
        hand_id: str,
        session_player_id: str,
        action_sequence: int,
        street: str,
        action_type: str,
        amount: float = 0.0,
        pot_size_before: float = 0.0,
        current_bet: float = 0.0,
        decision_time_ms: int = None
    ) -> str:
        """Record a player action"""
        session = self.db.get_session()
        try:
            action = PlayerAction(
                hand_id=hand_id,
                session_player_id=session_player_id,
                action_sequence=action_sequence,
                street=street,
                action_type=action_type,
                amount=amount,
                pot_size_before=pot_size_before,
                current_bet=current_bet,
                decision_time_ms=decision_time_ms
            )
            session.add(action)
            session.commit()
            return action.id
        finally:
            session.close()
    
    def update_hand_outcome(
        self,
        hand_id: str,
        winner_seat: int,
        winning_hand_type: str,
        total_pot: float,
        board_cards: Dict[str, Any],
        went_to_showdown: bool = False
    ):
        """Update hand with final outcome"""
        session = self.db.get_session()
        try:
            hand = session.query(Hand).filter(Hand.id == hand_id).first()
            if hand:
                hand.winner_seat = winner_seat
                hand.winning_hand_type = winning_hand_type
                hand.total_pot = total_pot
                hand.went_to_showdown = went_to_showdown
                
                # Update board cards
                if 'flop' in board_cards:
                    hand.flop_cards = board_cards['flop']
                if 'turn' in board_cards:
                    hand.turn_card = board_cards['turn']
                if 'river' in board_cards:
                    hand.river_card = board_cards['river']
                
                session.commit()
        finally:
            session.close()
    
    def update_player_stack(self, hand_player_id: str, ending_stack: float):
        """Update a player's ending stack for a hand"""
        session = self.db.get_session()
        try:
            hand_player = session.query(HandPlayer).filter(HandPlayer.id == hand_player_id).first()
            if hand_player:
                hand_player.ending_stack = ending_stack
                session.commit()
        finally:
            session.close()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a game session"""
        session = self.db.get_session()
        try:
            game_session = session.query(GameSession).filter(GameSession.id == session_id).first()
            if not game_session:
                return {}
            
            # Get hand count
            hand_count = session.query(Hand).filter(Hand.session_id == session_id).count()
            
            # Get player stats
            players = session.query(SessionPlayer).filter(SessionPlayer.session_id == session_id).all()
            
            return {
                "session_id": session_id,
                "session_type": game_session.session_type,
                "created_at": game_session.created_at,
                "total_hands": hand_count,
                "small_blind": game_session.small_blind,
                "big_blind": game_session.big_blind,
                "players": [
                    {
                        "name": p.player_name,
                        "is_bot": p.is_bot,
                        "starting_stack": p.starting_stack,
                        "final_stack": p.final_stack
                    }
                    for p in players
                ]
            }
        finally:
            session.close()
    
    def get_player_hands(self, session_player_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent hands for a specific player"""
        session = self.db.get_session()
        try:
            hand_players = (session.query(HandPlayer)
                          .filter(HandPlayer.session_player_id == session_player_id)
                          .limit(limit)
                          .all())
            
            results = []
            for hp in hand_players:
                hand = session.query(Hand).filter(Hand.id == hp.hand_id).first()
                if hand:
                    results.append({
                        "hand_id": hand.id,
                        "hand_number": hand.hand_number,
                        "position": hp.position_name,
                        "hole_cards": [hp.hole_card_1, hp.hole_card_2],
                        "starting_stack": hp.starting_stack,
                        "ending_stack": hp.ending_stack,
                        "won_amount": hp.won_amount,
                        "went_to_showdown": hp.went_to_showdown
                    })
            
            return results
        finally:
            session.close()
    
    def save_training_features(
        self,
        action_id: str,
        features: Dict[str, Any],
        target_action: str,
        target_amount: float = 0.0,
        feature_version: str = "1.0"
    ):
        """Save extracted features for machine learning training"""
        session = self.db.get_session()
        try:
            training_feature = TrainingFeature(
                action_id=action_id,
                position_features=features.get('position', {}),
                hand_strength_features=features.get('hand_strength', {}),
                opponent_features=features.get('opponent', {}),
                betting_features=features.get('betting', {}),
                board_texture_features=features.get('board_texture', {}),
                target_action=target_action,
                target_amount=target_amount,
                feature_version=feature_version
            )
            session.add(training_feature)
            session.commit()
        finally:
            session.close()
    
    def get_training_data(
        self, 
        limit: int = 10000, 
        feature_version: str = None
    ) -> List[Dict[str, Any]]:
        """Get training data for machine learning"""
        session = self.db.get_session()
        try:
            query = session.query(TrainingFeature)
            if feature_version:
                query = query.filter(TrainingFeature.feature_version == feature_version)
            
            features = query.limit(limit).all()
            
            results = []
            for f in features:
                results.append({
                    "features": {
                        "position": f.position_features,
                        "hand_strength": f.hand_strength_features,
                        "opponent": f.opponent_features,
                        "betting": f.betting_features,
                        "board_texture": f.board_texture_features
                    },
                    "target_action": f.target_action,
                    "target_amount": f.target_amount
                })
            
            return results
        finally:
            session.close()
    
    def record_model_performance(
        self,
        model_name: str,
        model_version: str,
        session_id: str,
        performance_metrics: Dict[str, float]
    ):
        """Record AI model performance metrics"""
        session = self.db.get_session()
        try:
            performance = ModelPerformance(
                model_name=model_name,
                model_version=model_version,
                session_id=session_id,
                **performance_metrics
            )
            session.add(performance)
            session.commit()
        finally:
            session.close()


# Integration with GameState
def integrate_with_game_state(game_state, data_manager: GameDataManager, session_id: str):
    """Helper function to integrate GameDataManager with GameState"""
    
    # Map player IDs to database player IDs
    player_mapping = {}
    
    # Add all players to the session
    for i, player in enumerate(game_state.players):
        db_player_id = data_manager.add_player_to_session(
            session_id=session_id,
            player_name=player.name,
            seat_number=i,
            starting_stack=player.stack,
            is_bot=player.name.startswith("Bot"),  # Simple bot detection
        )
        player_mapping[player.id] = db_player_id
    
    return player_mapping


# Example usage
if __name__ == "__main__":
    # Test the data manager
    data_manager = GameDataManager("sqlite:///test_game_data.db")
    
    # Create a test session
    session_id = data_manager.create_game_session(
        session_type="test",
        small_blind=1.0,
        big_blind=2.0
    )
    print(f"Created session: {session_id}")
    
    # Add players
    player1_id = data_manager.add_player_to_session(
        session_id, "TestBot", 1, 100.0, is_bot=True, bot_version="1.0"
    )
    player2_id = data_manager.add_player_to_session(
        session_id, "Human", 2, 100.0, is_bot=False
    )
    
    print(f"Added players: {player1_id}, {player2_id}")
    
    # Start a hand
    hand_id = data_manager.start_new_hand(
        session_id, hand_number=1, dealer_position=1, 
        small_blind_seat=1, big_blind_seat=2
    )
    print(f"Started hand: {hand_id}")
    
    # Record some actions
    action1_id = data_manager.record_action(
        hand_id, player1_id, 1, "preflop", "call", 2.0, 0.0, 2.0
    )
    action2_id = data_manager.record_action(
        hand_id, player2_id, 2, "preflop", "check", 0.0, 4.0, 2.0
    )
    
    print(f"Recorded actions: {action1_id}, {action2_id}")
    
    # Get session stats
    stats = data_manager.get_session_stats(session_id)
    print(f"Session stats: {stats}")
    
    print("Data manager test completed!")