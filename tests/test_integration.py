"""
Comprehensive Integration Tests for Poker Bot

Tests the integration between all major components:
- Poker Engine (GameState, HandEvaluator, Actions)
- AI Core (Neural Networks, Action Prediction)
- Training Pipeline (Data Processing, Feature Extraction)
- Data Management (Parsing, Storage)

These tests ensure all components work together correctly
and simulate realistic poker bot scenarios.
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from poker_engine.game_state import GameState, Player, GamePhase, Position
from poker_engine.hand_evaluator import HandEvaluator, Card, HandRank
from poker_engine.actions import Action, ActionType, ActionValidator

# AI imports
try:
    from ai_core.neural_networks.action_predictor import ActionPredictorNetwork, PokerAI
    from ai_core.neural_networks.simple_action_predictor import SimpleActionPredictor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Training imports
try:
    from training.data_processor import DataProcessor, TrainingExample
    from training.feature_extractor import FeatureExtractor, GameContext
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

# Data imports
try:
    from data.parsers.hand_history_parser import HandHistoryParser, ParsedHand, PlayerAction, Street
    from data.storage.game_data_manager import GameDataManager
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False


class TestPokerEngineIntegration:
    """Test integration between poker engine components"""
    
    def test_game_state_with_hand_evaluator(self):
        """Test game state working with hand evaluator"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        hand_evaluator = HandEvaluator()
        
        # Add players
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        # Start hand
        game.start_new_hand()
        
        # Deal some cards for testing
        alice_cards = [Card("As"), Card("Kh")]
        bob_cards = [Card("Qd"), Card("Jc")]
        board_cards = [Card("Ah"), Card("Ks"), Card("Qh"), Card("Jh"), Card("Th")]
        
        # Evaluate hands
        alice_strength = hand_evaluator.evaluate_hand(alice_cards + board_cards)
        bob_strength = hand_evaluator.evaluate_hand(bob_cards + board_cards)
        
        # Alice should have a straight (A-K-Q-J-T)
        # Bob should have a straight (A-K-Q-J-T) 
        assert alice_strength.rank >= HandRank.STRAIGHT
        assert bob_strength.rank >= HandRank.STRAIGHT
        
        # Game state should be properly initialized
        assert game.phase == GamePhase.PREFLOP
        assert len(game.players) == 2
        assert game.total_pot > 0  # Blinds posted
    
    def test_action_validation_with_game_state(self):
        """Test action validation integrated with game state"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        validator = ActionValidator()
        
        # Add players
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        game.start_new_hand()
        
        # Get current player (should be Alice - small blind)
        current_player = game.get_current_player()
        assert current_player is not None
        
        # Test legal actions for current game state
        legal_actions = validator.get_legal_actions(
            player_stack=current_player.stack,
            current_bet=game.current_bet,
            player_bet=current_player.current_bet,
            min_bet=game.big_blind
        )
        
        # Should include fold, call, raise options
        assert ActionType.FOLD in legal_actions
        assert ActionType.CALL in legal_actions or ActionType.CHECK in legal_actions
        
        # Process a valid action
        call_amount = game.current_bet - current_player.current_bet
        call_action = Action(ActionType.CALL, amount=call_amount)
        
        success = game.process_action(call_action)
        assert success
        assert current_player.current_bet == game.current_bet
    
    def test_complete_hand_with_evaluator(self):
        """Test complete hand with hand evaluation"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        hand_evaluator = HandEvaluator()
        
        # Setup players
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        game.start_new_hand()
        
        # Simulate preflop action (call, check)
        call_amount = game.current_bet - game.get_current_player().current_bet
        game.process_action(Action(ActionType.CALL, amount=call_amount))
        game.process_action(Action(ActionType.CHECK))
        
        assert game.phase == GamePhase.FLOP
        
        # Simulate flop action (check, check)
        game.process_action(Action(ActionType.CHECK))
        game.process_action(Action(ActionType.CHECK))
        
        assert game.phase == GamePhase.TURN
        
        # Simulate turn action (bet, call)
        game.process_action(Action(ActionType.BET, amount=5.0))
        call_amount = game.current_bet - game.get_current_player().current_bet
        game.process_action(Action(ActionType.CALL, amount=call_amount))
        
        assert game.phase == GamePhase.RIVER
        
        # Simulate river action (check, check)
        game.process_action(Action(ActionType.CHECK))
        game.process_action(Action(ActionType.CHECK))
        
        assert game.phase == GamePhase.SHOWDOWN
        
        # Test hand evaluation at showdown
        test_cards = [Card("As"), Card("Kh"), Card("Qd"), Card("Jc"), Card("Th")]
        strength = hand_evaluator.evaluate_hand(test_cards)
        assert strength.rank == HandRank.STRAIGHT


@pytest.mark.skipif(not AI_AVAILABLE, reason="AI modules not available")
class TestAIIntegration:
    """Test integration between AI components and game engine"""
    
    def test_action_predictor_with_game_state(self):
        """Test neural network action predictor with real game state"""
        # Create simple model for testing
        model = ActionPredictorNetwork(input_dim=50, hidden_dims=[32, 16], num_actions=6)
        
        # Create game state
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "AI", 100.0)
        game.add_player("p2", "Human", 100.0)
        game.start_new_hand()
        
        # Create dummy features (would normally come from feature extractor)
        features = np.random.rand(50).astype(np.float32)
        features_tensor = np.expand_dims(features, axis=0)
        
        # Get legal actions from game state
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            player_stack=game.get_current_player().stack,
            current_bet=game.current_bet,
            player_bet=game.get_current_player().current_bet,
            min_bet=game.big_blind
        )
        
        # Convert to action indices
        action_mapping = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
        legal_action_indices = [action_mapping.get(action.value, 0) for action in legal_actions]
        
        # Get prediction
        action_id, bet_size, confidence = model.predict_action(features_tensor, legal_action_indices)
        
        # Verify prediction is legal
        assert action_id in legal_action_indices
        assert 0.0 <= confidence <= 1.0
        assert bet_size >= 0.0
    
    def test_poker_ai_integration(self):
        """Test high-level PokerAI with game state"""
        # Create AI with default config
        ai = PokerAI()
        
        # Create game state
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "AI", 100.0)
        game.add_player("p2", "Human", 100.0)
        game.start_new_hand()
        
        # Create dummy features
        features = np.random.rand(110).astype(np.float32)
        
        # Get legal actions
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            player_stack=game.get_current_player().stack,
            current_bet=game.current_bet,
            player_bet=game.get_current_player().current_bet,
            min_bet=game.big_blind
        )
        
        legal_action_names = [action.value for action in legal_actions]
        
        # Make decision
        decision = ai.make_decision(features, legal_action_names)
        
        # Verify decision format
        assert 'action' in decision
        assert 'amount' in decision
        assert 'confidence' in decision
        assert decision['action'] in legal_action_names
        
        # Create corresponding Action object
        action_type = ActionType(decision['action'])
        if action_type in [ActionType.BET, ActionType.RAISE, ActionType.CALL]:
            action = Action(action_type, amount=decision['amount'])
        else:
            action = Action(action_type)
        
        # Process action in game
        success = game.process_action(action)
        assert success
    
    def test_simple_action_predictor_integration(self):
        """Test simple action predictor with game state"""
        predictor = SimpleActionPredictor()
        
        # Create dummy training data
        n_samples = 100
        features = np.random.rand(n_samples, 50)
        actions = np.random.choice(['fold', 'check', 'call', 'bet', 'raise'], n_samples)
        amounts = np.random.uniform(0, 50, n_samples)
        
        dataset = {
            'features': features,
            'actions': actions,
            'amounts': amounts
        }
        
        # Train predictor
        predictor.train(dataset)
        
        # Create game state
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "AI", 100.0)
        game.add_player("p2", "Human", 100.0)
        game.start_new_hand()
        
        # Get legal actions
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            player_stack=game.get_current_player().stack,
            current_bet=game.current_bet,
            player_bet=game.get_current_player().current_bet,
            min_bet=game.big_blind
        )
        
        legal_action_names = [action.value for action in legal_actions]
        
        # Make prediction
        test_features = np.random.rand(50)
        prediction = predictor.predict_action(test_features, legal_action_names)
        
        # Verify prediction
        assert prediction['action'] in legal_action_names
        assert 'confidence' in prediction
        assert 'amount' in prediction


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training modules not available")
class TestTrainingPipelineIntegration:
    """Test integration of training pipeline components"""
    
    def test_feature_extractor_with_game_state(self):
        """Test feature extractor with real game state"""
        extractor = FeatureExtractor()
        
        # Create game context from game state
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Hero", 100.0)
        game.add_player("p2", "Villain", 150.0)
        game.start_new_hand()
        
        # Create game context
        context = GameContext(
            hero_cards=["As", "Kh"],
            board_cards=["Qd", "Jc", "Th"],
            position="BTN",
            num_players=2,
            stack_size=100.0,
            pot_size=3.0,
            current_bet=2.0,
            hero_bet=1.0,
            street="flop",
            num_opponents=1,
            is_tournament=False,
            opponent_aggression=0.3,
            opponent_vpip=0.25
        )
        
        # Extract features
        features_dict = extractor.extract_features(context)
        features_vector = extractor.extract_features_vector(context)
        
        # Verify feature extraction
        assert 'position' in features_dict
        assert 'hand_strength' in features_dict
        assert 'pot_odds' in features_dict
        assert len(features_vector) == extractor.total_features
        assert np.all(np.isfinite(features_vector))  # No NaN or inf values
    
    @patch('training.data_processor.GameDataManager')
    def test_data_processor_integration(self, mock_data_manager):
        """Test data processor with mocked dependencies"""
        # Mock the data manager
        mock_data_manager.return_value = MagicMock()
        
        processor = DataProcessor()
        
        # Create sample parsed hand
        sample_hand = ParsedHand(
            hand_id="test_hand_1",
            site="pokerstars",
            timestamp=None,
            game_type="Hold'em No Limit",
            stakes=(1.0, 2.0),
            table_name="Test Table",
            max_players=6,
            button_seat=1
        )
        
        # Add players
        sample_hand.players = [
            type('Player', (), {
                'name': 'Hero',
                'seat': 1,
                'starting_stack': 100.0,
                'position': 'BTN',
                'hole_cards': ['As', 'Kh'],
                'actions': []
            })(),
            type('Player', (), {
                'name': 'Villain',
                'seat': 2,
                'starting_stack': 100.0,
                'position': 'BB',
                'hole_cards': [],
                'actions': []
            })()
        ]
        
        # Add some actions
        sample_hand.all_actions = [
            PlayerAction("Hero", "raise", 6.0, "preflop"),
            PlayerAction("Villain", "call", 6.0, "preflop")
        ]
        
        # Process hand
        training_examples = processor.process_single_hand(sample_hand, save_to_db=False)
        
        # Verify training examples generated
        assert len(training_examples) > 0
        for example in training_examples:
            assert isinstance(example, TrainingExample)
            assert len(example.features) == processor.feature_extractor.total_features
            assert example.target_action in processor.action_to_id
            assert example.target_amount >= 0.0


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Data modules not available")
class TestDataIntegration:
    """Test integration with data processing components"""
    
    def test_hand_history_parser_integration(self):
        """Test hand history parser with sample data"""
        parser = HandHistoryParser()
        
        # Create sample PokerStars hand history
        sample_pokerstars = """PokerStars Hand #123456789: Hold'em No Limit ($0.50/$1.00 USD) - 2024/01/01 12:00:00 ET
Table 'Test Table' 6-max Seat #1 is the button
Seat 1: Hero ($100.00 in chips)
Seat 2: Villain ($100.00 in chips)
Hero: posts small blind $0.50
Villain: posts big blind $1.00
*** HOLE CARDS ***
Dealt to Hero [As Kh]
Hero: raises $2.50 to $3.50
Villain: calls $2.50
*** FLOP *** [Qd Jc Th]
Villain: checks
Hero: bets $5.00
Villain: calls $5.00
*** TURN *** [Qd Jc Th] [9h]
Villain: checks
Hero: bets $12.00
Villain: folds
Uncalled bet ($12.00) returned to Hero
Hero collected $16.50 from pot
*** SUMMARY ***
Total pot $17.00 | Rake $0.50
Board [Qd Jc Th 9h]
Seat 1: Hero (button) (small blind) collected ($16.50)
Seat 2: Villain (big blind) folded on the Turn"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_pokerstars)
            temp_filename = f.name
        
        try:
            # Parse the file
            parsed_hands = parser.parse_file(temp_filename)
            
            # Verify parsing
            assert len(parsed_hands) == 1
            hand = parsed_hands[0]
            
            assert hand.hand_id == "123456789"
            assert len(hand.players) == 2
            assert hand.stakes == (0.5, 1.0)
            assert len(hand.all_actions) > 0
            
            # Verify players
            hero = next(p for p in hand.players if p.name == "Hero")
            assert hero.hole_cards == ["As", "Kh"]
            assert hero.starting_stack == 100.0
            
        finally:
            # Clean up
            os.unlink(temp_filename)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""
    
    def test_complete_poker_game_simulation(self):
        """Test a complete poker game from start to finish"""
        # Initialize components
        game = GameState(small_blind=1.0, big_blind=2.0)
        hand_evaluator = HandEvaluator()
        validator = ActionValidator()
        
        # Add players
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        # Track game progression
        hands_played = 0
        max_hands = 5
        
        while hands_played < max_hands and len([p for p in game.players if p.stack > 0]) >= 2:
            # Start new hand
            game.start_new_hand()
            hands_played += 1
            
            # Play through streets
            while game.phase not in [GamePhase.SHOWDOWN, GamePhase.HAND_COMPLETE]:
                current_player = game.get_current_player()
                if not current_player or not current_player.can_act():
                    break
                
                # Get legal actions
                legal_actions = validator.get_legal_actions(
                    player_stack=current_player.stack,
                    current_bet=game.current_bet,
                    player_bet=current_player.current_bet,
                    min_bet=game.big_blind
                )
                
                # Simple strategy: call/check if possible, otherwise fold
                if ActionType.CHECK in legal_actions:
                    action = Action(ActionType.CHECK)
                elif ActionType.CALL in legal_actions:
                    call_amount = game.current_bet - current_player.current_bet
                    action = Action(ActionType.CALL, amount=call_amount)
                else:
                    action = Action(ActionType.FOLD)
                
                # Process action
                success = game.process_action(action)
                if not success:
                    break
            
            # Verify game state consistency
            assert game.total_pot >= 0
            assert all(p.stack >= 0 for p in game.players)
            assert sum(p.total_bet_this_hand for p in game.players) <= game.total_pot
        
        # Verify multiple hands were played
        assert hands_played > 0
    
    @pytest.mark.skipif(not AI_AVAILABLE, reason="AI modules not available")
    def test_ai_vs_ai_simulation(self):
        """Test AI vs AI game simulation"""
        # Create AIs
        ai1 = PokerAI()
        ai2 = PokerAI()
        
        # Create game
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("ai1", "AI_1", 100.0)
        game.add_player("ai2", "AI_2", 100.0)
        
        # Play one hand
        game.start_new_hand()
        
        # Track decisions
        decisions_made = 0
        
        while game.phase not in [GamePhase.SHOWDOWN, GamePhase.HAND_COMPLETE] and decisions_made < 20:
            current_player = game.get_current_player()
            if not current_player or not current_player.can_act():
                break
            
            # Get legal actions
            validator = ActionValidator()
            legal_actions = validator.get_legal_actions(
                player_stack=current_player.stack,
                current_bet=game.current_bet,
                player_bet=current_player.current_bet,
                min_bet=game.big_blind
            )
            
            legal_action_names = [action.value for action in legal_actions]
            
            # Create dummy features for AI decision
            features = np.random.rand(110).astype(np.float32)
            
            # Select AI based on current player
            current_ai = ai1 if current_player.id == "ai1" else ai2
            
            # Make decision
            decision = current_ai.make_decision(features, legal_action_names)
            
            # Convert to action
            action_type = ActionType(decision['action'])
            if action_type in [ActionType.BET, ActionType.RAISE, ActionType.CALL]:
                action = Action(action_type, amount=decision['amount'])
            else:
                action = Action(action_type)
            
            # Process action
            success = game.process_action(action)
            assert success
            
            decisions_made += 1
        
        # Verify game progressed
        assert decisions_made > 0
        assert game.phase in [GamePhase.FLOP, GamePhase.TURN, GamePhase.RIVER, 
                             GamePhase.SHOWDOWN, GamePhase.HAND_COMPLETE]


class TestErrorHandlingAndPerformance:
    """Test error handling and performance characteristics"""
    
    def test_invalid_action_handling(self):
        """Test system handles invalid actions gracefully"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        # Try invalid actions
        invalid_actions = [
            Action(ActionType.BET, amount=-5.0),  # Negative bet
            Action(ActionType.CALL, amount=1000.0),  # More than stack
            Action(ActionType.RAISE, amount=0.5),  # Below minimum raise
        ]
        
        for action in invalid_actions:
            success = game.process_action(action)
            assert not success  # Should fail gracefully
            
            # Game state should remain consistent
            assert all(p.stack >= 0 for p in game.players)
            assert game.total_pot >= 0
    
    def test_hand_evaluator_performance(self):
        """Test hand evaluator performance with many evaluations"""
        evaluator = HandEvaluator()
        
        # Test cards
        test_hands = [
            [Card("As"), Card("Kh"), Card("Qd"), Card("Jc"), Card("Th")],  # Straight
            [Card("Ac"), Card("Ad"), Card("Ah"), Card("As"), Card("Kh")],  # Four of a kind
            [Card("2c"), Card("7d"), Card("9h"), Card("Js"), Card("Ah")],  # High card
        ]
        
        # Time many evaluations
        import time
        start_time = time.time()
        
        for _ in range(1000):
            for hand in test_hands:
                result = evaluator.evaluate_hand(hand)
                assert result.rank >= HandRank.HIGH_CARD
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # Should be reasonably fast (less than 1 second for 3000 evaluations)
        assert evaluation_time < 1.0
        
        # Verify evaluations per second
        evals_per_second = 3000 / evaluation_time
        assert evals_per_second > 1000  # Should handle at least 1000 evaluations/second
    
    def test_memory_usage_stability(self):
        """Test that repeated operations don't cause memory leaks"""
        import gc
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(100):
            game = GameState(small_blind=1.0, big_blind=2.0)
            game.add_player(f"p1_{i}", f"Player1_{i}", 100.0)
            game.add_player(f"p2_{i}", f"Player2_{i}", 100.0)
            game.start_new_hand()
            
            # Play some actions
            game.process_action(Action(ActionType.CALL, amount=1.0))
            game.process_action(Action(ActionType.CHECK))
            
            # Clean up explicitly
            del game
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow too much (allowing for some growth)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold


if __name__ == "__main__":
    # Run specific test groups
    pytest.main([
        __file__ + "::TestPokerEngineIntegration",
        "-v", "--tb=short"
    ])
