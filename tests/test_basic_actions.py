"""
Basic test suite for the Actions module

Tests just the Action and ActionType classes that are actually implemented.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import poker_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_engine.actions import Action, ActionType


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


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])