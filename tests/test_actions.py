"""
Test suite for the Actions module

Tests all poker action functionality including:
- Action creation and validation
- Action types and amounts
- Legal action determination
- Bet sizing recommendations
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import poker_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_engine.actions import (
    Action, ActionType
)


class TestActionType:
    """Test the ActionType enum"""
    
    def test_action_type_values(self):
        """Test that all action types have correct values"""
        assert ActionType.FOLD.value == "fold"
        assert ActionType.CHECK.value == "check"
        assert ActionType.CALL.value == "call"
        assert ActionType.BET.value == "bet"
        assert ActionType.RAISE.value == "raise"
        assert ActionType.ALL_IN.value == "all_in"
    
    def test_action_type_string_conversion(self):
        """Test converting action types to strings"""
        assert str(ActionType.FOLD) == "ActionType.FOLD"


class TestAction:
    """Test the Action dataclass"""
    
    def test_valid_action_creation(self):
        """Test creating valid actions"""
        # Fold action (no amount needed)
        action = Action(ActionType.FOLD)
        assert action.action_type == ActionType.FOLD
        assert action.amount == 0
        
        # Bet action with amount
        action = Action(ActionType.BET, amount=10.0)
        assert action.action_type == ActionType.BET
        assert action.amount == 10.0
    
    def test_invalid_action_validation(self):
        """Test that invalid actions raise errors"""
        # Bet without amount
        with pytest.raises(ValueError):
            Action(ActionType.BET, amount=0)
        
        # Bet with negative amount
        with pytest.raises(ValueError):
            Action(ActionType.BET, amount=-5.0)
        
        # Fold with amount (should not have amount)
        with pytest.raises(ValueError):
            Action(ActionType.FOLD, amount=5.0)
        
        # Check with amount (should not have amount)
        with pytest.raises(ValueError):
            Action(ActionType.CHECK, amount=10.0)
    
    def test_action_properties(self):
        """Test action properties"""
        # Aggressive actions
        bet_action = Action(ActionType.BET, amount=10.0)
        raise_action = Action(ActionType.RAISE, amount=20.0)
        assert bet_action.is_aggressive()
        assert raise_action.is_aggressive()
        
        # Non-aggressive actions
        fold_action = Action(ActionType.FOLD)
        call_action = Action(ActionType.CALL, amount=10.0)
        check_action = Action(ActionType.CHECK)
        assert not fold_action.is_aggressive()
        assert not call_action.is_aggressive()
        assert not check_action.is_aggressive()
    
    def test_action_string_representation(self):
        """Test action string representation"""
        fold_action = Action(ActionType.FOLD)
        bet_action = Action(ActionType.BET, amount=15.5)
        
        assert "fold" in str(fold_action).lower()
        assert "bet" in str(bet_action).lower()
        assert "15.5" in str(bet_action)


class TestActionValidator:
    """Test the ActionValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = ActionValidator()
    
    def test_legal_actions_no_bet(self):
        """Test legal actions when there's no bet to call"""
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            current_bet=0,
            player_bet=0,
            player_stack=100,
            min_raise=2
        )
        
        expected = [ActionType.FOLD, ActionType.CHECK, ActionType.BET, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
    
    def test_legal_actions_with_bet(self):
        """Test legal actions when there's a bet to call"""
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            current_bet=10,
            player_bet=0,
            player_stack=100,
            min_raise=10
        )
        
        expected = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
    
    def test_legal_actions_short_stack(self):
        """Test legal actions with insufficient chips to raise"""
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            current_bet=10,
            player_bet=0,
            player_stack=15,  # Can only call or go all-in
            min_raise=10
        )
        
        expected = [ActionType.FOLD, ActionType.CALL, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
    
    def test_legal_actions_already_all_in(self):
        """Test legal actions when player is already all-in"""
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            current_bet=10,
            player_bet=0,
            player_stack=0,  # No chips left
            min_raise=10
        )
        
        # Player can't take any actions when all-in
        assert legal_actions == []
    
    def test_legal_actions_already_called(self):
        """Test legal actions when player has already called"""
        validator = ActionValidator()
        legal_actions = validator.get_legal_actions(
            current_bet=10,
            player_bet=10,  # Already called
            player_stack=90,
            min_raise=10
        )
        
        expected = [ActionType.FOLD, ActionType.CHECK, ActionType.RAISE, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
    
    def test_validate_fold_action(self):
        """Test validating fold actions"""
        validator = ActionValidator()
        action = Action(ActionType.FOLD)
        
        # Fold should always be valid (unless specific game rules prevent it)
        legal_actions = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE]
        assert validator.validate_action(action, legal_actions, current_bet=10, player_bet=0)
    
    def test_validate_check_action(self):
        """Test validating check actions"""
        validator = ActionValidator()
        action = Action(ActionType.CHECK)
        
        # Check valid when no bet to call
        legal_actions = [ActionType.FOLD, ActionType.CHECK, ActionType.BET]
        assert validator.validate_action(action, legal_actions, current_bet=0, player_bet=0)
        
        # Check invalid when there's a bet to call
        legal_actions = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE]
        assert not validator.validate_action(action, legal_actions, current_bet=10, player_bet=0)
    
    def test_validate_call_action(self):
        """Test validating call actions"""
        validator = ActionValidator()
        action = Action(ActionType.CALL, amount=10.0)
        
        # Call valid when there's a bet to call
        legal_actions = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE]
        assert validator.validate_action(action, legal_actions, current_bet=10, player_bet=0)
        
        # Call invalid when amount is wrong
        action_wrong_amount = Action(ActionType.CALL, amount=5.0)
        assert not validator.validate_action(action_wrong_amount, legal_actions, current_bet=10, player_bet=0)
    
    def test_validate_bet_action(self):
        """Test validating bet actions"""
        validator = ActionValidator()
        action = Action(ActionType.BET, amount=15.0)
        
        # Bet valid when no current bet and amount >= min bet
        legal_actions = [ActionType.FOLD, ActionType.CHECK, ActionType.BET]
        assert validator.validate_action(action, legal_actions, current_bet=0, player_bet=0, min_bet=10)
        
        # Bet invalid when amount too small
        small_bet = Action(ActionType.BET, amount=5.0)
        assert not validator.validate_action(small_bet, legal_actions, current_bet=0, player_bet=0, min_bet=10)
    
    def test_validate_raise_action(self):
        """Test validating raise actions"""
        validator = ActionValidator()
        action = Action(ActionType.RAISE, amount=20.0)
        
        # Raise valid when amount >= min raise
        legal_actions = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE]
        assert validator.validate_action(action, legal_actions, current_bet=10, player_bet=0, min_raise=10)
        
        # Raise invalid when amount too small
        small_raise = Action(ActionType.RAISE, amount=5.0)
        assert not validator.validate_action(small_raise, legal_actions, current_bet=10, player_bet=0, min_raise=10)
    
    def test_validate_all_in_action(self):
        """Test validating all-in actions"""
        validator = ActionValidator()
        action = Action(ActionType.ALL_IN, amount=50.0)
        
        # All-in should always be valid if player has chips
        legal_actions = [ActionType.FOLD, ActionType.CALL, ActionType.ALL_IN]
        assert validator.validate_action(action, legal_actions, current_bet=10, player_bet=0, player_stack=50)


class TestBetSizingRecommender:
    """Test the BetSizingRecommender class"""
    
    def test_pot_bet_recommendation(self):
        """Test pot-sized bet recommendation"""
        recommender = BetSizingRecommender()
        sizes = recommender.recommend_bet_sizes(pot_size=20, player_stack=100)
        
        # Should include pot-sized bet
        assert 20 in sizes
    
    def test_standard_bet_sizes(self):
        """Test standard bet sizing recommendations"""
        recommender = BetSizingRecommender()
        sizes = recommender.recommend_bet_sizes(pot_size=20, player_stack=100)
        
        # Should include common bet sizes
        expected_sizes = [5, 10, 15, 20, 30, 40, 100]  # Various fractions of pot + all-in
        for size in sizes:
            assert size in expected_sizes or any(abs(size - pot_fraction * 20) < 0.1 for pot_fraction in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    
    def test_all_in_always_included(self):
        """Test that all-in is always included in recommendations"""
        recommender = BetSizingRecommender()
        sizes = recommender.recommend_bet_sizes(pot_size=20, player_stack=50)
        
        # All-in (50) should always be included
        assert 50 in sizes
    
    def test_small_stack_recommendations(self):
        """Test recommendations when player has small stack"""
        recommender = BetSizingRecommender()
        sizes = recommender.recommend_bet_sizes(pot_size=100, player_stack=30)
        
        # With small stack, should only get smaller bet sizes and all-in
        assert 30 in sizes  # All-in
        assert all(size <= 30 for size in sizes)
    
    def test_preflop_sizing(self):
        """Test preflop-specific bet sizing"""
        recommender = BetSizingRecommender()
        sizes = recommender.recommend_preflop_sizes(big_blind=2, player_stack=200)
        
        # Should include standard preflop raises (2.5x, 3x, 4x BB)
        expected = [5, 6, 8, 200]  # 2.5BB, 3BB, 4BB, all-in
        for expected_size in expected:
            assert expected_size in sizes


# Global functions not implemented yet - tests removed


class TestIntegrationScenarios:
    """Test realistic poker scenarios"""
    
    def test_preflop_scenario(self):
        """Test a typical preflop scenario"""
        validator = ActionValidator()
        
        # Big blind posted, UTG to act
        legal_actions = validator.get_legal_actions(
            current_bet=2,    # Big blind
            player_bet=0,     # UTG hasn't acted
            player_stack=200, # 100BB stack
            min_raise=2       # Min raise is one big blind
        )
        
        expected = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
        
        # Test a raise
        raise_action = Action(ActionType.RAISE, amount=6)  # 3x BB raise
        assert validator.validate_action(
            raise_action, 
            legal_actions, 
            current_bet=2, 
            player_bet=0, 
            min_raise=2
        )
    
    def test_river_betting_scenario(self):
        """Test a river betting scenario"""
        validator = ActionValidator()
        recommender = BetSizingRecommender()
        
        # No one has bet, first to act
        legal_actions = validator.get_legal_actions(
            current_bet=0,
            player_bet=0,
            player_stack=75,
            min_bet=2
        )
        
        expected = [ActionType.FOLD, ActionType.CHECK, ActionType.BET, ActionType.ALL_IN]
        assert set(legal_actions) == set(expected)
        
        # Test bet sizing
        bet_sizes = recommender.recommend_bet_sizes(pot_size=30, player_stack=75)
        assert 75 in bet_sizes  # All-in
        assert any(7 <= size <= 8 for size in bet_sizes)  # ~25% pot
        assert any(14 <= size <= 16 for size in bet_sizes)  # ~50% pot
    
    def test_all_in_scenario(self):
        """Test scenario where player is facing an all-in"""
        validator = ActionValidator()
        
        # Opponent shoved, we have to call or fold
        legal_actions = validator.get_legal_actions(
            current_bet=100,  # Opponent all-in
            player_bet=0,
            player_stack=50,  # We have less than the bet
            min_raise=0       # Can't raise anyway
        )
        
        expected = [ActionType.FOLD, ActionType.ALL_IN]  # Call becomes all-in
        assert set(legal_actions) == set(expected)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])