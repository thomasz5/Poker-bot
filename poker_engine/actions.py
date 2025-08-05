"""Poker actions and action validation."""

from enum import Enum
from typing import Union, Optional
from dataclasses import dataclass


class ActionType(Enum):
    """Types of poker actions."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class Action:
    """Represents a poker action."""
    action_type: ActionType
    amount: float = 0.0
    player_id: int = 0
    
    def __post_init__(self):
        """Validate action after initialization."""
        if self.action_type in [ActionType.BET, ActionType.RAISE, ActionType.CALL]:
            if self.amount <= 0:
                raise ValueError(f"{self.action_type.value} action must have positive amount")
        
        if self.action_type in [ActionType.FOLD, ActionType.CHECK]:
            if self.amount != 0:
                raise ValueError(f"{self.action_type.value} action should not have amount")
    
    def is_aggressive(self) -> bool:
        """Return True if action is aggressive (bet/raise)."""
        return self.action_type in [ActionType.BET, ActionType.RAISE]
    
    def __str__(self):
        if self.amount > 0:
            return f"{self.action_type.value}({self.amount})"
        return self.action_type.value


class ActionValidator:
    """Validates poker actions based on game state."""
    
    @staticmethod
    def get_legal_actions(player_stack: float, 
                         current_bet: float, 
                         to_call: float,
                         min_raise: float,
                         can_check: bool = False) -> list[ActionType]:
        """
        Get list of legal actions for a player.
        
        Args:
            player_stack: Player's remaining chips
            current_bet: Player's current bet this round
            to_call: Amount needed to call
            min_raise: Minimum raise amount
            can_check: Whether player can check
            
        Returns:
            List of legal ActionTypes
        """
        legal_actions = []
        
        # Can always fold
        legal_actions.append(ActionType.FOLD)
        
        # Can check if no bet to call
        if can_check and to_call == 0:
            legal_actions.append(ActionType.CHECK)
        
        # Can call if there's something to call and player has chips
        if to_call > 0 and player_stack >= to_call:
            legal_actions.append(ActionType.CALL)
        
        # Can bet if no bet to call and player has chips
        if to_call == 0 and player_stack > 0:
            legal_actions.append(ActionType.BET)
        
        # Can raise if there's a bet to call and player has enough for min raise
        if to_call > 0 and player_stack >= to_call + min_raise:
            legal_actions.append(ActionType.RAISE)
        
        # Can go all-in if player has any chips
        if player_stack > 0:
            legal_actions.append(ActionType.ALL_IN)
        
        return legal_actions
    
    @staticmethod
    def validate_action(action: Action,
                       player_stack: float,
                       current_bet: float,
                       to_call: float,
                       min_raise: float,
                       can_check: bool = False) -> bool:
        """
        Validate if an action is legal.
        
        Returns:
            True if action is legal, False otherwise
        """
        legal_actions = ActionValidator.get_legal_actions(
            player_stack, current_bet, to_call, min_raise, can_check
        )
        
        if action.action_type not in legal_actions:
            return False
        
        # Additional amount validations
        if action.action_type == ActionType.CALL:
            return action.amount == to_call
        
        elif action.action_type == ActionType.BET:
            return 0 < action.amount <= player_stack
        
        elif action.action_type == ActionType.RAISE:
            min_total = to_call + min_raise
            max_total = player_stack
            return min_total <= action.amount <= max_total
        
        elif action.action_type == ActionType.ALL_IN:
            return action.amount == player_stack
        
        elif action.action_type in [ActionType.FOLD, ActionType.CHECK]:
            return action.amount == 0
        
        return True
    
    @staticmethod
    def get_bet_sizes(player_stack: float, 
                     pot_size: float,
                     current_bet: float = 0,
                     to_call: float = 0) -> dict[str, float]:
        """
        Get common bet sizes as fractions of pot.
        
        Returns:
            Dictionary of bet size names to amounts
        """
        effective_stack = player_stack
        if to_call > 0:
            effective_stack = player_stack - to_call
        
        bet_sizes = {}
        
        # Common bet sizes as fractions of pot
        for fraction, name in [(0.25, "quarter_pot"), (0.33, "third_pot"), 
                              (0.5, "half_pot"), (0.66, "two_thirds_pot"),
                              (0.75, "three_quarters_pot"), (1.0, "pot"),
                              (1.5, "overbet"), (2.0, "double_pot")]:
            
            size = pot_size * fraction
            if size <= effective_stack:
                bet_sizes[name] = size
        
        # Always include all-in
        bet_sizes["all_in"] = effective_stack
        
        return bet_sizes


# Example usage
if __name__ == "__main__":
    # Test action creation
    fold_action = Action(ActionType.FOLD)
    bet_action = Action(ActionType.BET, amount=50.0)
    raise_action = Action(ActionType.RAISE, amount=100.0)
    
    print(f"Fold: {fold_action}")
    print(f"Bet: {bet_action}")
    print(f"Raise: {raise_action}")
    print(f"Bet is aggressive: {bet_action.is_aggressive()}")
    
    # Test action validation
    validator = ActionValidator()
    
    # Player with 200 chips, facing a bet of 50, min raise is 50
    legal_actions = validator.get_legal_actions(
        player_stack=200, current_bet=0, to_call=50, min_raise=50
    )
    print(f"Legal actions: {[a.value for a in legal_actions]}")
    
    # Test bet sizes
    bet_sizes = validator.get_bet_sizes(
        player_stack=200, pot_size=100, to_call=50
    )
    print(f"Available bet sizes: {bet_sizes}") 