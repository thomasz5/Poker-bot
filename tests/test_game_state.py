"""
Test suite for the Game State Manager

Tests all game state functionality including:
- Player management and positions
- Betting rounds and pot calculations
- Game flow and street transitions
- Hand management
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import poker_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_engine.game_state import (
    GameState, Player, GamePhase, Position, Pot
)
from poker_engine.actions import Action, ActionType


class TestPlayer:
    """Test the Player class"""
    
    def test_player_creation(self):
        """Test creating a player"""
        player = Player(id="player1", name="Alice", stack=100.0)
        assert player.id == "player1"
        assert player.name == "Alice"
        assert player.stack == 100.0
        assert player.can_act()
        assert not player.has_folded
        assert not player.is_all_in
    
    def test_player_betting(self):
        """Test player betting functionality"""
        player = Player(id="player1", name="Alice", stack=100.0)
        
        # Normal bet
        amount_bet = player.bet(20.0)
        assert amount_bet == 20.0
        assert player.stack == 80.0
        assert player.current_bet == 20.0
        assert player.total_bet_this_hand == 20.0
        
        # All-in bet
        amount_bet = player.bet(90.0)  # More than remaining stack
        assert amount_bet == 80.0  # Can only bet what's left
        assert player.stack == 0.0
        assert player.is_all_in
        assert player.current_bet == 100.0
    
    def test_player_folding(self):
        """Test player folding"""
        player = Player(id="player1", name="Alice", stack=100.0)
        player.fold()
        
        assert player.has_folded
        assert not player.is_active
        assert not player.can_act()
    
    def test_player_reset(self):
        """Test resetting player for new hand"""
        player = Player(id="player1", name="Alice", stack=100.0)
        player.bet(20.0)
        player.hole_cards = ["As", "Ks"]
        
        player.reset_for_new_hand()
        
        assert player.current_bet == 0.0
        assert player.total_bet_this_hand == 0.0
        assert not player.has_folded
        assert not player.is_all_in
        assert player.hole_cards == []
        assert player.is_active  # Still active since has chips


class TestPot:
    """Test the Pot class"""
    
    def test_pot_creation(self):
        """Test creating a pot"""
        pot = Pot()
        assert pot.amount == 0.0
        assert pot.eligible_players == []
    
    def test_adding_chips(self):
        """Test adding chips to pot"""
        pot = Pot()
        pot.add_chips(10.0)
        assert pot.amount == 10.0
        
        pot.add_chips(5.0)
        assert pot.amount == 15.0


class TestGameState:
    """Test the GameState class"""
    
    def test_game_state_creation(self):
        """Test creating a game state"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        assert game.small_blind == 1.0
        assert game.big_blind == 2.0
        assert game.phase == GamePhase.PREFLOP
        assert game.hand_number == 0
    
    def test_adding_players(self):
        """Test adding players to the game"""
        game = GameState()
        
        player1 = game.add_player("p1", "Alice", 100.0, Position.BUTTON)
        player2 = game.add_player("p2", "Bob", 100.0, Position.SMALL_BLIND)
        
        assert len(game.players) == 2
        assert player1.name == "Alice"
        assert player1.position == Position.BUTTON
        assert player2.name == "Bob"
    
    def test_heads_up_hand_setup(self):
        """Test setting up a heads-up hand"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        game.start_new_hand()
        
        # Check basic setup
        assert game.hand_number == 1
        assert game.phase == GamePhase.PREFLOP
        assert len(game.active_players) == 2
        assert game.current_bet == 2.0  # Big blind
        assert game.main_pot.amount == 3.0  # SB + BB
        
        # Check positions (heads-up: dealer is SB)
        dealer_player = game.active_players[game.dealer_position]
        other_player = game.active_players[(game.dealer_position + 1) % 2]
        assert dealer_player.position == Position.SMALL_BLIND
        assert other_player.position == Position.BIG_BLIND
    
    def test_full_table_hand_setup(self):
        """Test setting up a full table hand"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        
        # Add 6 players
        for i in range(6):
            game.add_player(f"p{i}", f"Player{i}", 100.0)
        
        game.start_new_hand()
        
        assert len(game.active_players) == 6
        assert game.main_pot.amount == 3.0  # SB + BB
        
        # Check that positions are assigned
        button_player = None
        sb_player = None
        bb_player = None
        
        for player in game.active_players:
            if player.position == Position.BUTTON:
                button_player = player
            elif player.position == Position.SMALL_BLIND:
                sb_player = player
            elif player.position == Position.BIG_BLIND:
                bb_player = player
        
        assert button_player is not None
        assert sb_player is not None
        assert bb_player is not None
    
    def test_legal_actions_preflop(self):
        """Test legal actions preflop"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        legal_actions = game.get_legal_actions()
        expected = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
    
    def test_processing_call_action(self):
        """Test processing a call action"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        current_player = game.get_current_player()
        call_amount = game.current_bet - current_player.current_bet
        action = Action(ActionType.CALL, amount=call_amount)
        
        success = game.process_action(action)
        assert success
        assert game.main_pot.amount == 4.0  # SB + BB + Call
        
        # Should move to next player
        next_player = game.get_current_player()
        assert next_player != current_player
    
    def test_processing_raise_action(self):
        """Test processing a raise action"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        # First player raises to 6
        action = Action(ActionType.RAISE, amount=6.0)
        success = game.process_action(action)
        
        assert success
        assert game.current_bet == 6.0
        assert game.main_pot.amount > 3.0  # More than just blinds
    
    def test_processing_fold_action(self):
        """Test processing a fold action"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        current_player = game.get_current_player()
        action = Action(ActionType.FOLD)
        success = game.process_action(action)
        
        assert success
        assert current_player.has_folded
        assert not current_player.can_act()
    
    def test_betting_round_completion(self):
        """Test that betting rounds complete properly"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        assert game.phase == GamePhase.PREFLOP
        
        # Both players call/check to complete preflop
        # First player calls
        current_player = game.get_current_player()
        call_amount = game.current_bet - current_player.current_bet
        action = Action(ActionType.CALL, amount=call_amount)
        game.process_action(action)
        
        # Second player checks (already posted BB)
        action = Action(ActionType.CHECK)
        game.process_action(action)
        
        # Should advance to flop
        assert game.phase == GamePhase.FLOP
        assert game.current_bet == 0.0  # Reset for new street
    
    def test_all_in_scenario(self):
        """Test all-in scenarios"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 50.0)  # Short stack
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        # Alice goes all-in
        action = Action(ActionType.ALL_IN)
        success = game.process_action(action)
        
        assert success
        current_player = game.get_current_player()
        assert current_player.name == "Alice"
        assert current_player.is_all_in
        assert current_player.stack == 0.0
    
    def test_multiple_hands(self):
        """Test playing multiple hands"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        # Play first hand
        game.start_new_hand()
        assert game.hand_number == 1
        old_dealer = game.dealer_position
        
        # Play second hand
        game.start_new_hand()
        assert game.hand_number == 2
        # Dealer button should move
        assert game.dealer_position == (old_dealer + 1) % 2
    
    def test_insufficient_players(self):
        """Test error when not enough players"""
        game = GameState()
        game.add_player("p1", "Alice", 100.0)
        
        # Should raise error with only 1 player
        with pytest.raises(ValueError):
            game.start_new_hand()
    
    def test_game_info(self):
        """Test getting game information"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        info = game.get_game_info()
        
        assert info["hand_number"] == 1
        assert info["phase"] == "preflop"
        assert info["pot_size"] == 3.0
        assert info["current_bet"] == 2.0
        assert info["active_players"] == 2
        assert info["current_player"] is not None
    
    def test_three_player_game(self):
        """Test three-player game functionality"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.add_player("p3", "Charlie", 100.0)
        
        game.start_new_hand()
        
        assert len(game.active_players) == 3
        assert game.main_pot.amount == 3.0
        
        # Check that UTG is first to act (position after BB)
        current_player = game.get_current_player()
        assert current_player.position in [Position.UNDER_THE_GUN, Position.BUTTON]  # Depends on dealer position
    
    def test_invalid_action_rejection(self):
        """Test that invalid actions are rejected"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.start_new_hand()
        
        # Try to check when there's a bet to call
        action = Action(ActionType.CHECK)
        success = game.process_action(action)
        assert not success
        
        # Try to bet an invalid amount
        action = Action(ActionType.RAISE, amount=0.5)  # Below min raise
        success = game.process_action(action)
        assert not success


class TestIntegrationScenarios:
    """Test realistic poker scenarios end-to-end"""
    
    def test_complete_hand_scenario(self):
        """Test a complete hand from start to finish"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        game.start_new_hand()
        assert game.phase == GamePhase.PREFLOP
        
        # Preflop action: call and check
        call_amount = game.current_bet - game.get_current_player().current_bet
        game.process_action(Action(ActionType.CALL, amount=call_amount))
        game.process_action(Action(ActionType.CHECK))
        
        assert game.phase == GamePhase.FLOP
        
        # Flop action: check, check
        game.process_action(Action(ActionType.CHECK))
        game.process_action(Action(ActionType.CHECK))
        
        assert game.phase == GamePhase.TURN
        
        # Turn action: bet, call
        game.process_action(Action(ActionType.BET, amount=5.0))
        call_amount = game.current_bet - game.get_current_player().current_bet
        game.process_action(Action(ActionType.CALL, amount=call_amount))
        
        assert game.phase == GamePhase.RIVER
        
        # River action: check, check
        game.process_action(Action(ActionType.CHECK))
        game.process_action(Action(ActionType.CHECK))
        
        assert game.phase == GamePhase.SHOWDOWN
    
    def test_fold_wins_hand(self):
        """Test scenario where fold ends the hand early"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        
        game.start_new_hand()
        
        # Alice raises, Bob folds
        game.process_action(Action(ActionType.RAISE, amount=6.0))
        game.process_action(Action(ActionType.FOLD))
        
        # Hand should be over
        active_players = [p for p in game.active_players if not p.has_folded]
        assert len(active_players) == 1
    
    def test_position_rotation(self):
        """Test that positions rotate correctly"""
        game = GameState(small_blind=1.0, big_blind=2.0)
        game.add_player("p1", "Alice", 100.0)
        game.add_player("p2", "Bob", 100.0)
        game.add_player("p3", "Charlie", 100.0)
        
        # Hand 1
        game.start_new_hand()
        hand1_positions = {p.name: p.position for p in game.active_players}
        
        # Hand 2
        game.start_new_hand()
        hand2_positions = {p.name: p.position for p in game.active_players}
        
        # Positions should have rotated
        assert hand1_positions != hand2_positions


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])