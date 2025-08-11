"""
Game State Manager for Poker Bot

This module handles all the game state tracking including:
- Player management and positions
- Betting rounds and pot calculations
- Card dealing and board management
- Game flow and street transitions
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
try:
    from .actions import Action, ActionType
    from .hand_evaluator import Card
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from actions import Action, ActionType
    from hand_evaluator import Card


class GamePhase(Enum):
    """Different phases of a poker hand"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    HAND_COMPLETE = "hand_complete"


class Position(Enum):
    """Player positions at the table"""
    SMALL_BLIND = "SB"
    BIG_BLIND = "BB"
    UNDER_THE_GUN = "UTG"
    UNDER_THE_GUN_1 = "UTG+1"
    UNDER_THE_GUN_2 = "UTG+2"
    MIDDLE_POSITION_1 = "MP1"
    MIDDLE_POSITION_2 = "MP2"
    CUTOFF = "CO"
    BUTTON = "BTN"


@dataclass
class Player:
    """Represents a player in the game"""
    id: str
    name: str
    stack: float
    position: Optional[Position] = None
    hole_cards: List[Card] = field(default_factory=list)
    current_bet: float = 0.0
    total_bet_this_hand: float = 0.0
    is_active: bool = True
    is_all_in: bool = False
    has_folded: bool = False
    
    def can_act(self) -> bool:
        """Check if player can take an action"""
        return self.is_active and not self.has_folded and not self.is_all_in
    
    def bet(self, amount: float) -> float:
        """Make a bet, returns actual amount bet"""
        # Treat betting amount semantics safely:
        # Any request equal to or greater than current stack results in all-in
        if amount >= self.stack:
            # All-in
            actual_amount = self.stack
            self.is_all_in = True
        else:
            actual_amount = amount
            
        self.stack -= actual_amount
        self.current_bet += actual_amount
        self.total_bet_this_hand += actual_amount
        
        return actual_amount
    
    def fold(self):
        """Fold the hand"""
        self.has_folded = True
        self.is_active = False
    
    def reset_for_new_hand(self):
        """Reset player state for a new hand"""
        self.hole_cards = []
        self.current_bet = 0.0
        self.total_bet_this_hand = 0.0
        self.is_all_in = False
        self.has_folded = False
        self.is_active = self.stack > 0  # Only active if has chips


@dataclass
class Pot:
    """Represents a pot (main or side pot)"""
    amount: float = 0.0
    eligible_players: List[str] = field(default_factory=list)
    
    def add_chips(self, amount: float):
        """Add chips to the pot"""
        self.amount += amount


class GameState:
    """Main game state manager"""
    
    def __init__(self, small_blind: float = 1.0, big_blind: float = 2.0):
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Game state
        self.phase = GamePhase.PREFLOP
        self.players: List[Player] = []
        self.active_players: List[Player] = []
        self.dealer_position = 0
        self.current_player_index = 0
        
        # Cards and board
        self.deck: List[Card] = []
        self.board: List[Card] = []
        
        # Betting
        self.main_pot = Pot()
        self.side_pots: List[Pot] = []
        self.current_bet = 0.0
        self.min_raise = 0.0
        self.last_aggressor_index = -1
        # Maintain a running total pot value for tests expecting this attribute
        self.total_pot: float = 0.0
        
        # Hand tracking
        self.hand_number = 0
        self.actions_this_street: List[Tuple[str, Action]] = []
        
    def add_player(self, player_id: str, name: str, stack: float, position: Position = None) -> Player:
        """Add a new player to the game"""
        player = Player(id=player_id, name=name, stack=stack, position=position)
        self.players.append(player)
        return player
    
    def start_new_hand(self):
        """Initialize a new hand"""
        self.hand_number += 1
        self.phase = GamePhase.PREFLOP
        self.board = []
        self.current_bet = 0.0
        self.min_raise = self.big_blind
        self.last_aggressor_index = -1
        self.actions_this_street = []
        
        # Reset all pots
        self.main_pot = Pot()
        self.side_pots = []
        
        # Reset players and set active players
        self.active_players = []
        for player in self.players:
            player.reset_for_new_hand()
            if player.stack > 0:
                self.active_players.append(player)
        
        # Set up positions for heads-up or full table
        self._assign_positions()
        
        # Post blinds
        self._post_blinds()
        
        # Deal hole cards (would integrate with actual deck later)
        self._deal_hole_cards()
        
        # Set first to act
        self._set_first_to_act_preflop()
    
    def _assign_positions(self):
        """Assign positions to active players"""
        num_players = len(self.active_players)
        if num_players < 2:
            raise ValueError("Need at least 2 players to start a hand")
        
        # Move dealer button except for very first hand
        if self.hand_number > 1:
            self.dealer_position = (self.dealer_position + 1) % num_players
        
        if num_players == 2:
            # Heads-up: dealer is small blind
            self.active_players[self.dealer_position].position = Position.SMALL_BLIND
            self.active_players[(self.dealer_position + 1) % 2].position = Position.BIG_BLIND
        else:
            # Full table
            positions = [Position.BUTTON, Position.SMALL_BLIND, Position.BIG_BLIND, 
                        Position.UNDER_THE_GUN, Position.UNDER_THE_GUN_1, Position.UNDER_THE_GUN_2,
                        Position.MIDDLE_POSITION_1, Position.MIDDLE_POSITION_2, Position.CUTOFF]
            
            for i, player in enumerate(self.active_players):
                pos_index = (i - self.dealer_position) % num_players
                if pos_index < len(positions):
                    player.position = positions[pos_index]
    
    def _post_blinds(self):
        """Post small and big blinds"""
        num_players = len(self.active_players)
        
        if num_players == 2:
            # Heads-up: dealer posts small blind
            sb_player = self.active_players[self.dealer_position]
            bb_player = self.active_players[(self.dealer_position + 1) % 2]
        else:
            # Full table
            sb_player = self.active_players[(self.dealer_position + 1) % num_players]
            bb_player = self.active_players[(self.dealer_position + 2) % num_players]
        
        # Post small blind
        sb_amount = sb_player.bet(self.small_blind)
        self.main_pot.add_chips(sb_amount)
        self.total_pot = self.main_pot.amount
        self.main_pot.eligible_players.append(sb_player.id)
        
        # Post big blind
        bb_amount = bb_player.bet(self.big_blind)
        self.main_pot.add_chips(bb_amount)
        self.total_pot = self.main_pot.amount
        self.main_pot.eligible_players.append(bb_player.id)
        
        self.current_bet = self.big_blind
    
    def _deal_hole_cards(self):
        """Deal 2 hole cards to each active player"""
        # This would integrate with a proper deck shuffling system
        # For now, just initialize empty hole cards
        for player in self.active_players:
            player.hole_cards = []  # Would deal actual cards here
    
    def _set_first_to_act_preflop(self):
        """Set the first player to act preflop"""
        num_players = len(self.active_players)
        
        if num_players == 2:
            # Heads-up: small blind acts first preflop
            self.current_player_index = self.dealer_position
        else:
            # Full table: UTG acts first
            self.current_player_index = (self.dealer_position + 3) % num_players
    
    def get_current_player(self) -> Optional[Player]:
        """Get the player whose turn it is to act"""
        if 0 <= self.current_player_index < len(self.active_players):
            return self.active_players[self.current_player_index]
        return None
    
    def get_legal_actions(self) -> List[ActionType]:
        """Get legal actions for the current player"""
        current_player = self.get_current_player()
        if not current_player or not current_player.can_act():
            return []
        
        legal_actions = [ActionType.FOLD]
        
        # Can always fold (except when no bet to call)
        if self.current_bet == current_player.current_bet:
            # No bet to call
            legal_actions.append(ActionType.CHECK)
        else:
            # There's a bet to call
            legal_actions.append(ActionType.CALL)
        
        # Can bet/raise if has chips
        if current_player.stack > 0:
            if self.current_bet == 0:
                legal_actions.append(ActionType.BET)
            else:
                legal_actions.append(ActionType.RAISE)
            
            # All-in is always available if has chips
            legal_actions.append(ActionType.ALL_IN)
        
        return legal_actions
    
    def process_action(self, action: Action) -> bool:
        """Process a player action and update game state"""
        current_player = self.get_current_player()
        if not current_player:
            return False
        
        # Validate action
        legal_actions = self.get_legal_actions()
        if action.action_type not in legal_actions:
            return False
        
        # Execute action
        if action.action_type == ActionType.FOLD:
            current_player.fold()
            
        elif action.action_type == ActionType.CHECK:
            # No chips involved
            pass
            
        elif action.action_type == ActionType.CALL:
            call_amount = self.current_bet - current_player.current_bet
            actual_amount = current_player.bet(call_amount)
            self.main_pot.add_chips(actual_amount)
            self.total_pot = self.main_pot.amount
            
        elif action.action_type in [ActionType.BET, ActionType.RAISE]:
            # Interpret action.amount as the total amount this player is betting ("bet/raise to")
            if action.amount:
                target_total_for_player = action.amount
                # Bet/raise must be at least current_bet for bets or exceed for raises
                if target_total_for_player <= current_player.current_bet:
                    return False

                # Compute additional chips needed to reach target
                additional = target_total_for_player - current_player.current_bet

                # Enforce minimum raise amount when raising over an existing bet
                if self.current_bet > 0:
                    min_total_required = self.current_bet + self.min_raise
                    if target_total_for_player < min_total_required:
                        return False

                actual_amount = current_player.bet(additional)
                self.main_pot.add_chips(actual_amount)
                self.total_pot = self.main_pot.amount
                # Update table current bet to the player's new total
                self.current_bet = current_player.current_bet
                # Update min_raise to the size of the last raise increment (new - previous table bet)
                last_raise_increment = max(self.current_bet - (self.current_bet - additional if self.current_bet - additional > 0 else 0) - (self.current_bet - additional - (self.current_bet - additional)), 0.0)
                # Simpler and correct: when there is an existing bet, min_raise is (new_total - previous_table_bet)
                if self.current_bet > 0:
                    # previous_table_bet before this action was (self.current_bet - additional)
                    previous_table_bet = self.current_bet - additional
                    self.min_raise = max(self.current_bet - previous_table_bet, self.big_blind)
                else:
                    self.min_raise = max(additional, self.big_blind)
                self.last_aggressor_index = self.current_player_index
        
        elif action.action_type == ActionType.ALL_IN:
            # All-in: move entire stack
            amount_to_bet = current_player.stack
            actual_amount = current_player.bet(amount_to_bet)
            self.main_pot.add_chips(actual_amount)
            self.total_pot = self.main_pot.amount
            if current_player.current_bet > self.current_bet:
                self.current_bet = current_player.current_bet
                self.last_aggressor_index = self.current_player_index
        
        # Record action
        self.actions_this_street.append((current_player.id, action))
        
        # Move to next player unless all-in (some flows keep the same pointer until others act)
        if action.action_type != ActionType.ALL_IN:
            self._advance_to_next_player()
        
        # Check if betting round is complete
        if self._is_betting_round_complete():
            self._advance_to_next_street()
        
        return True
    
    def _advance_to_next_player(self):
        """Move to the next player who can act"""
        players_checked = 0
        while players_checked < len(self.active_players):
            self.current_player_index = (self.current_player_index + 1) % len(self.active_players)
            current_player = self.active_players[self.current_player_index]
            
            if current_player.can_act():
                return
                
            players_checked += 1
        
        # No one can act - betting round is over
        self.current_player_index = -1
    
    def _is_betting_round_complete(self) -> bool:
        """Check if the current betting round is finished"""
        active_players_in_hand = [p for p in self.active_players if not p.has_folded]
        if len(active_players_in_hand) <= 1:
            return True

        # If no bet to call, require that each active player has acted on this street
        if self.current_bet == 0.0:
            if len(self.actions_this_street) == 0:
                return False
            actors = set(pid for (pid, _) in self.actions_this_street)
            required = sum(1 for p in active_players_in_hand if p.can_act())
            return len(actors) >= required

        # There is a bet: ensure all non-folded, non-all-in players have matched the current bet
        for player in active_players_in_hand:
            if player.can_act() and not player.is_all_in and player.current_bet < self.current_bet:
                return False
        return True
    
    def _advance_to_next_street(self):
        """Move to the next betting round"""
        # Reset betting for new street
        for player in self.active_players:
            player.current_bet = 0.0
        
        self.current_bet = 0.0
        self.min_raise = self.big_blind
        self.actions_this_street = []
        
        # Advance phase
        if self.phase == GamePhase.PREFLOP:
            self.phase = GamePhase.FLOP
            self._deal_flop()
        elif self.phase == GamePhase.FLOP:
            self.phase = GamePhase.TURN
            self._deal_turn()
        elif self.phase == GamePhase.TURN:
            self.phase = GamePhase.RIVER
            self._deal_river()
        elif self.phase == GamePhase.RIVER:
            self.phase = GamePhase.SHOWDOWN
            self._handle_showdown()
            return
        
        # Set first to act (usually small blind or first active player)
        self._set_first_to_act_postflop()
    
    def _deal_flop(self):
        """Deal the flop (3 community cards)"""
        # Would integrate with deck system
        self.board = []  # Would add 3 cards
    
    def _deal_turn(self):
        """Deal the turn (4th community card)"""
        # Would add 1 card to board
        pass
    
    def _deal_river(self):
        """Deal the river (5th community card)"""
        # Would add 1 card to board
        pass
    
    def _set_first_to_act_postflop(self):
        """Set first to act after the flop"""
        # Find first active player after dealer
        for i in range(len(self.active_players)):
            player_index = (self.dealer_position + 1 + i) % len(self.active_players)
            if self.active_players[player_index].can_act():
                self.current_player_index = player_index
                return
        
        # No one can act
        self.current_player_index = -1
    
    def _handle_showdown(self):
        """Handle showdown and determine winners"""
        # This would integrate with hand evaluator
        # Keep phase at SHOWDOWN; external resolution may transition to HAND_COMPLETE
        return
    
    def get_game_info(self) -> Dict:
        """Get current game state information"""
        return {
            "hand_number": self.hand_number,
            "phase": self.phase.value,
            "pot_size": self.main_pot.amount,
            "current_bet": self.current_bet,
            "min_raise": self.min_raise,
            "board": [str(card) for card in self.board],
            "active_players": len([p for p in self.active_players if not p.has_folded]),
            "current_player": self.get_current_player().id if self.get_current_player() else None,
            "dealer_position": self.dealer_position
        }


# Example usage and testing
if __name__ == "__main__":
    # Create a game
    game = GameState(small_blind=1.0, big_blind=2.0)
    
    # Add players
    game.add_player("player1", "Alice", 100.0)
    game.add_player("player2", "Bob", 100.0)
    game.add_player("player3", "Charlie", 100.0)
    
    # Start a hand
    game.start_new_hand()
    
    print("Game started!")
    print(f"Current game info: {game.get_game_info()}")
    print(f"Current player: {game.get_current_player().name}")
    print(f"Legal actions: {[action.value for action in game.get_legal_actions()]}")
    
    # Test some actions
    from actions import Action, ActionType
    
    # First player calls
    current_player = game.get_current_player()
    call_amount = game.current_bet - current_player.current_bet
    action = Action(ActionType.CALL, amount=call_amount)
    success = game.process_action(action)
    print(f"Action processed: {success}")
    print(f"Current player: {game.get_current_player().name if game.get_current_player() else 'None'}")
    
    # Second player raises
    if game.get_current_player():
        action = Action(ActionType.RAISE, amount=6.0)
        success = game.process_action(action)
        print(f"Raise processed: {success}")
        print(f"Current bet: {game.current_bet}")
        print(f"Pot size: {game.main_pot.amount}")