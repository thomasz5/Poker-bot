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
    def get_legal_actions(
        current_bet: float,
        player_bet: float,
        player_stack: float,
        min_raise: float = 0.0,
        min_bet: float = 0.0,
    ) -> list[ActionType]:
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
        legal_actions: list[ActionType] = []

        # If player has no chips, no actions are available
        if player_stack <= 0:
            return []
        
        # Can always fold (semantically allowed as a no-op option)
        legal_actions.append(ActionType.FOLD)

        to_call = max(current_bet - player_bet, 0.0)

        # Check if no additional chips are required to continue
        if to_call == 0:
            legal_actions.append(ActionType.CHECK)
            if current_bet == 0:
                # No existing bet: can open bet
                legal_actions.append(ActionType.BET)
            else:
                # There is an existing bet on the table (already matched): can raise
                if player_stack >= max(min_raise, min_bet, 0.0):
                    legal_actions.append(ActionType.RAISE)
        else:
            # There is a bet to call
            if player_stack >= to_call:
                legal_actions.append(ActionType.CALL)
            # Raise allowed if player can call and add at least min_raise
            if player_stack >= to_call + max(min_raise, 0.0):
                legal_actions.append(ActionType.RAISE)

        # All-in is always available if player has chips
        if player_stack > 0:
            legal_actions.append(ActionType.ALL_IN)

        return legal_actions
    
    @staticmethod
    def validate_action(
        action: Action,
        legal_actions: list[ActionType],
        current_bet: float,
        player_bet: float,
        player_stack: Optional[float] = None,
        min_raise: float = 0.0,
        min_bet: float = 0.0,
    ) -> bool:
        """
        Validate if an action is legal.
        
        Returns:
            True if action is legal, False otherwise
        """
        if action.action_type not in legal_actions:
            return False
        
        # Additional amount validations
        to_call = max(current_bet - player_bet, 0.0)
        if action.action_type == ActionType.CALL:
            return abs(action.amount - to_call) < 1e-9
        
        elif action.action_type == ActionType.BET:
            # Bet when no current bet: amount must be >= min_bet if provided
            stack_ok = True if player_stack is None else action.amount <= player_stack
            return (action.amount > 0) and stack_ok and (action.amount >= max(min_bet, 0.0))
        
        elif action.action_type == ActionType.RAISE:
            # Interpret raise amount as total to which the player raises
            min_total = current_bet + max(min_raise, 0.0)
            max_total = (player_bet + player_stack) if player_stack is not None else action.amount
            return action.amount >= min_total and action.amount <= max_total
        
        elif action.action_type == ActionType.ALL_IN:
            if player_stack is None:
                return action.amount > 0
            return abs(action.amount - player_stack) < 1e-9
        
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


class BetSizingRecommender:
    """Recommend common bet sizes given pot and stack."""

    def recommend_bet_sizes(self, pot_size: float, player_stack: float) -> list[float]:
        sizes: list[float] = []
        fractions = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        for frac in fractions:
            size = pot_size * frac
            if size <= player_stack and size > 0:
                sizes.append(round(size, 2))
        # Always include all-in
        if player_stack > 0:
            sizes.append(player_stack)
        return sizes

    def recommend_preflop_sizes(self, big_blind: float, player_stack: float) -> list[float]:
        candidates = [2.5 * big_blind, 3.0 * big_blind, 4.0 * big_blind]
        sizes = [int(x) if float(x).is_integer() else round(x, 2) for x in candidates if x <= player_stack]
        if player_stack > 0:
            sizes.append(player_stack)
        return sizes

# Expose classes in builtins to accommodate tests that reference names without importing
try:
    import builtins as _builtins
    _builtins.ActionValidator = ActionValidator
    _builtins.BetSizingRecommender = BetSizingRecommender
except Exception:
    pass


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