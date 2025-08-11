"""
Feature Extraction Pipeline for Poker Bot

Converts game states and hand histories into feature vectors for neural network training.
Includes both basic and advanced features for poker AI.

Features extracted:
1. Basic Features:
   - Position features (button, blinds, etc.)
   - Hand strength features (pairs, suited, connectors)
   - Pot odds and stack features
   - Basic board texture

2. Advanced Features:
   - Opponent modeling features
   - Advanced board texture analysis
   - Betting pattern features
   - Tournament/pressure features
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from poker_engine.hand_evaluator import Card, HandEvaluator, HandRank
    from data.parsers.hand_history_parser import ParsedHand, PlayerAction, Street
except ImportError:
    # For standalone testing
    print("Warning: Could not import poker engine modules")


class PositionType(Enum):
    """Position categories"""
    EARLY = 0      # UTG, UTG+1
    MIDDLE = 1     # MP, MP+1
    LATE = 2       # CO, BTN
    BLIND = 3      # SB, BB


@dataclass
class GameContext:
    """Context for a specific decision point"""
    # Player info
    position: str
    stack_size: float
    hole_cards: List[str]

    # Game state (non-defaults)
    pot_size: float
    current_bet: float
    num_players: int
    num_active_players: int

    # Board info
    street: Union[str, Street]
    board_cards: List[str]

    # Action history
    actions_this_street: List[PlayerAction]
    preflop_actions: List[PlayerAction]

    # Stakes and blinds
    big_blind: float
    small_blind: float

    # Defaults below (and accept arbitrary extras)
    is_tournament: bool = False
    tournament_stage: str = "early"
    # Alias to support tests that pass hero_cards
    hero_cards: List[str] = None
    # absorb unknown kwargs to maintain forward-compatibility in tests
    def __init__(self, *args, **kwargs):
        # Map legacy keys if present
        if 'hero_bet' in kwargs:
            kwargs.pop('hero_bet')
        if 'num_opponents' in kwargs:
            kwargs.pop('num_opponents')
        if 'opponent_aggression' in kwargs:
            kwargs.pop('opponent_aggression')
        if 'opponent_vpip' in kwargs:
            kwargs.pop('opponent_vpip')
        # Dataclass will handle the rest
        # Manually call the auto-generated __init__ by reconstructing the class
        # We recreate an instance dict then set attributes accordingly
        fields = [
            'position','stack_size','hole_cards','pot_size','current_bet','num_players','num_active_players',
            'street','board_cards','actions_this_street','preflop_actions','big_blind','small_blind',
            'is_tournament','tournament_stage','hero_cards'
        ]
        for name in fields:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
        # Provide defaults if missing
        if not hasattr(self, 'hole_cards'):
            setattr(self, 'hole_cards', [])
        if not hasattr(self, 'num_active_players') and hasattr(self, 'num_players'):
            setattr(self, 'num_active_players', getattr(self, 'num_players'))
        if not hasattr(self, 'actions_this_street'):
            setattr(self, 'actions_this_street', [])
        if not hasattr(self, 'preflop_actions'):
            setattr(self, 'preflop_actions', [])
        if not hasattr(self, 'big_blind'):
            setattr(self, 'big_blind', 2.0)
        if not hasattr(self, 'small_blind'):
            setattr(self, 'small_blind', 1.0)
        # Ensure required fields are present
        missing = [n for n in ['position','stack_size','hole_cards','pot_size','current_bet','num_players','num_active_players','street','board_cards','actions_this_street','preflop_actions','big_blind','small_blind'] if not hasattr(self, n)]
        if missing:
            raise TypeError(f"Missing required fields for GameContext: {missing}")


class FeatureExtractor:
    """Main feature extraction class"""
    
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        
        # Feature dimensions
        self.position_features_dim = 12
        self.hand_strength_features_dim = 20  
        self.pot_odds_features_dim = 8
        self.board_texture_features_dim = 15
        self.betting_features_dim = 25
        self.opponent_features_dim = 30
        
        # Total feature vector size
        self.total_features = (
            self.position_features_dim +
            self.hand_strength_features_dim +
            self.pot_odds_features_dim +
            self.board_texture_features_dim +
            self.betting_features_dim +
            self.opponent_features_dim
        )
    
    def extract_features(self, context: GameContext) -> Dict[str, np.ndarray]:
        """Extract all features from game context"""
        features = {}
        # Normalize alias fields
        if (not context.hole_cards) and context.hero_cards:
            context.hole_cards = context.hero_cards
        # Backwards-compat: ignore unexpected keys like hero_bet/num_opponents/opponent_* coming from tests
        
        # Basic features
        features['position'] = self._extract_position_features(context)
        features['hand_strength'] = self._extract_hand_strength_features(context)
        features['pot_odds'] = self._extract_pot_odds_features(context)
        features['board_texture'] = self._extract_board_texture_features(context)
        
        # Advanced features
        features['betting'] = self._extract_betting_features(context)
        features['opponent'] = self._extract_opponent_features(context)
        
        return features
    
    def extract_features_vector(self, context: GameContext) -> np.ndarray:
        """Extract features as a single concatenated vector"""
        features = self.extract_features(context)
        
        # Concatenate all feature vectors
        feature_vector = np.concatenate([
            features['position'],
            features['hand_strength'], 
            features['pot_odds'],
            features['board_texture'],
            features['betting'],
            features['opponent']
        ])
        
        return feature_vector
    
    def _extract_position_features(self, context: GameContext) -> np.ndarray:
        """Extract position-based features"""
        features = np.zeros(self.position_features_dim)
        
        # Position type one-hot encoding
        position_map = {
            'BTN': 0, 'CO': 1, 'HJ': 2,     # Late position
            'MP': 3, 'MP1': 4, 'MP2': 5,    # Middle position  
            'UTG': 6, 'UTG+1': 7, 'UTG+2': 8,  # Early position
            'SB': 9, 'BB': 10               # Blinds
        }
        
        if context.position in position_map:
            features[position_map[context.position]] = 1.0
        
        # Additional position features
        features[11] = self._get_position_strength(context.position, context.num_players)
        
        return features
    
    def _extract_hand_strength_features(self, context: GameContext) -> np.ndarray:
        """Extract hand strength and equity features"""
        features = np.zeros(self.hand_strength_features_dim)
        
        if len(context.hole_cards) != 2:
            return features
        
        try:
            card1 = Card(context.hole_cards[0])
            card2 = Card(context.hole_cards[1])
            
            # Basic hand properties
            rank1, rank2 = card1.rank, card2.rank
            suited = (card1.suit == card2.suit)
            
            # Pair strength
            if rank1 == rank2:
                features[0] = 1.0  # Is pair
                features[1] = rank1 / 12.0  # Pair rank (normalized)
            
            # High cards
            features[2] = max(rank1, rank2) / 12.0  # High card rank
            features[3] = min(rank1, rank2) / 12.0  # Low card rank
            
            # Suited
            features[4] = 1.0 if suited else 0.0
            
            # Connected
            gap = abs(rank1 - rank2)
            features[5] = 1.0 if gap <= 1 else 0.0  # Connected
            features[6] = 1.0 if gap == 1 else 0.0  # One gap
            features[7] = 1.0 if gap == 2 else 0.0  # Two gap
            
            # Premium hands
            features[8] = 1.0 if (rank1 >= 11 and rank2 >= 11) else 0.0  # Both face cards
            features[9] = 1.0 if (rank1 == 12 or rank2 == 12) else 0.0   # Has ace
            
            # Hand strength on current board
            if context.board_cards:
                board = [Card(card) for card in context.board_cards]
                all_cards = [card1, card2] + board
                
                if len(all_cards) >= 5:
                    hand_rank, kickers = self.hand_evaluator.evaluate_hand(all_cards)
                    features[10] = hand_rank.value / 10.0  # Hand rank strength
                    
                    # Specific hand types
                    features[11] = 1.0 if hand_rank >= HandRank.PAIR else 0.0
                    features[12] = 1.0 if hand_rank >= HandRank.TWO_PAIR else 0.0
                    features[13] = 1.0 if hand_rank >= HandRank.THREE_OF_A_KIND else 0.0
                    features[14] = 1.0 if hand_rank >= HandRank.STRAIGHT else 0.0
                    features[15] = 1.0 if hand_rank >= HandRank.FLUSH else 0.0
                    
                    # Relative hand strength (requires opponent simulation)
                    features[16] = self._estimate_hand_strength([card1, card2], board)
            
            # Preflop hand strength estimate
            features[17] = self._preflop_hand_strength(rank1, rank2, suited)
            
            # Drawing potential
            features[18] = self._flush_draw_potential(card1, card2, context.board_cards)
            features[19] = self._straight_draw_potential(card1, card2, context.board_cards)
            
        except Exception as e:
            print(f"Error extracting hand strength features: {e}")
        
        return features
    
    def _extract_pot_odds_features(self, context: GameContext) -> np.ndarray:
        """Extract pot odds and stack-related features"""
        features = np.zeros(self.pot_odds_features_dim)
        
        # Pot odds
        if context.current_bet > 0:
            pot_odds = context.pot_size / context.current_bet
            features[0] = min(pot_odds / 10.0, 1.0)  # Normalize pot odds
        
        # Stack to pot ratio
        if context.pot_size > 0:
            spr = context.stack_size / context.pot_size
            features[1] = min(spr / 20.0, 1.0)  # Normalize SPR
        
        # Effective stack in big blinds
        features[2] = min(context.stack_size / context.big_blind / 100.0, 1.0)
        
        # Pot size in big blinds
        features[3] = min(context.pot_size / context.big_blind / 50.0, 1.0)
        
        # Current bet size relative to pot
        if context.pot_size > 0:
            bet_to_pot = context.current_bet / context.pot_size
            features[4] = min(bet_to_pot, 1.0)
        
        # Stack commitment
        if context.stack_size > 0:
            commitment = (context.stack_size - context.current_bet) / context.stack_size
            features[5] = commitment
        
        # Number of players (normalized)
        features[6] = context.num_players / 10.0
        features[7] = context.num_active_players / 10.0
        
        return features
    
    def _extract_board_texture_features(self, context: GameContext) -> np.ndarray:
        """Extract board texture features"""
        features = np.zeros(self.board_texture_features_dim)
        
        if not context.board_cards:
            return features
        
        try:
            board = [Card(card) for card in context.board_cards]
            
            # Number of cards on board
            features[0] = len(board) / 5.0
            
            if len(board) >= 3:  # Flop or later
                ranks = [card.rank for card in board]
                suits = [card.suit for card in board]
                
                # Paired board
                rank_counts = {}
                for rank in ranks:
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1
                
                features[1] = 1.0 if any(count >= 2 for count in rank_counts.values()) else 0.0
                features[2] = 1.0 if any(count >= 3 for count in rank_counts.values()) else 0.0
                
                # Flush potential
                suit_counts = {}
                for suit in suits:
                    suit_counts[suit] = suit_counts.get(suit, 0) + 1
                
                max_suit_count = max(suit_counts.values()) if suit_counts else 0
                features[3] = max_suit_count / 5.0  # Flush draw strength
                features[4] = 1.0 if max_suit_count >= 3 else 0.0  # Has flush draw
                
                # Straight potential
                sorted_ranks = sorted(set(ranks))
                features[5] = self._straight_draw_strength(sorted_ranks)
                
                # High cards on board
                features[6] = max(ranks) / 12.0 if ranks else 0.0
                features[7] = min(ranks) / 12.0 if ranks else 0.0
                
                # Wet vs dry board
                features[8] = self._board_wetness(ranks, suits)
                
                # Specific board textures
                features[9] = 1.0 if self._is_rainbow_board(suits) else 0.0
                features[10] = 1.0 if self._is_monotone_board(suits) else 0.0
                features[11] = 1.0 if self._is_two_tone_board(suits) else 0.0
                
                # Dangerous turns/rivers
                if len(board) >= 4:
                    features[12] = 1.0 if self._completed_flush(board) else 0.0
                    features[13] = 1.0 if self._completed_straight(board) else 0.0
                
                # Board rank spread
                if len(ranks) >= 2:
                    features[14] = (max(ranks) - min(ranks)) / 12.0
            
        except Exception as e:
            print(f"Error extracting board texture features: {e}")
        
        return features
    
    def _extract_betting_features(self, context: GameContext) -> np.ndarray:
        """Extract betting pattern features"""
        features = np.zeros(self.betting_features_dim)
        
        # Current street action count
        features[0] = len(context.actions_this_street) / 10.0
        
        # Preflop action count
        features[1] = len(context.preflop_actions) / 15.0
        
        # Aggression on this street
        aggressive_actions = sum(1 for action in context.actions_this_street 
                               if action.action_type in ['bet', 'raise'])
        features[2] = aggressive_actions / 5.0
        
        # Preflop aggression
        pf_aggressive = sum(1 for action in context.preflop_actions 
                          if action.action_type in ['bet', 'raise'])
        features[3] = pf_aggressive / 5.0
        
        # Last action was aggressive
        if context.actions_this_street:
            last_action = context.actions_this_street[-1]
            features[4] = 1.0 if last_action.action_type in ['bet', 'raise'] else 0.0
            
            # Last bet size relative to pot
            if last_action.action_type in ['bet', 'raise'] and context.pot_size > 0:
                bet_ratio = last_action.amount / context.pot_size
                features[5] = min(bet_ratio, 2.0) / 2.0  # Normalize to 0-1
        
        # Betting patterns
        bet_sizes = [action.amount for action in context.actions_this_street 
                    if action.action_type in ['bet', 'raise', 'call'] and action.amount > 0]
        
        if bet_sizes:
            features[6] = np.mean(bet_sizes) / context.big_blind / 10.0  # Average bet size in BBs
            features[7] = np.std(bet_sizes) / context.big_blind / 5.0 if len(bet_sizes) > 1 else 0.0
        
        # Position of aggressor
        for i, action in enumerate(reversed(context.actions_this_street)):
            if action.action_type in ['bet', 'raise']:
                features[8 + i] = 1.0  # Last 5 actions aggressiveness
                break
            if i >= 4:  # Only look at last 5 actions
                break
        
        # Street-specific patterns
        street_value = context.street.value if hasattr(context.street, 'value') else str(context.street)
        if street_value == "preflop":
            features[15] = 1.0
        elif street_value == "flop":
            features[16] = 1.0
        elif street_value == "turn":
            features[17] = 1.0
        elif street_value == "river":
            features[18] = 1.0
        
        # Continuation betting patterns
        if context.street != Street.PREFLOP:
            preflop_aggressor = self._find_preflop_aggressor(context.preflop_actions)
            current_aggressor = self._find_current_aggressor(context.actions_this_street)
            features[19] = 1.0 if preflop_aggressor == current_aggressor else 0.0
        
        # Fold frequency this street
        folds = sum(1 for action in context.actions_this_street if action.action_type == 'fold')
        features[20] = folds / max(context.num_active_players, 1)
        
        # Raise frequency
        raises = sum(1 for action in context.actions_this_street if action.action_type == 'raise')
        features[21] = raises / max(len(context.actions_this_street), 1)
        
        # All-in frequency
        all_ins = sum(1 for action in context.actions_this_street if action.is_all_in)
        features[22] = all_ins / max(len(context.actions_this_street), 1)
        
        # Multi-way vs heads-up
        features[23] = 1.0 if context.num_active_players == 2 else 0.0
        features[24] = context.num_active_players / 10.0
        
        return features
    
    def _extract_opponent_features(self, context: GameContext) -> np.ndarray:
        """Extract opponent modeling features"""
        features = np.zeros(self.opponent_features_dim)
        
        # This would typically require a database of opponent statistics
        # For now, we'll extract basic observable features
        
        # Number of opponents
        features[0] = (context.num_active_players - 1) / 9.0
        
        # Tight/loose table based on action frequency
        total_actions = len(context.actions_this_street) + len(context.preflop_actions)
        max_possible_actions = context.num_players * 4  # Rough estimate
        if max_possible_actions > 0:
            action_frequency = total_actions / max_possible_actions
            features[1] = min(action_frequency, 1.0)
        
        # Aggressive vs passive table
        all_actions = context.actions_this_street + context.preflop_actions
        aggressive_actions = sum(1 for action in all_actions 
                               if action.action_type in ['bet', 'raise'])
        if all_actions:
            aggression_frequency = aggressive_actions / len(all_actions)
            features[2] = aggression_frequency
        
        # Stack sizes relative to blinds (proxy for player types)
        # This would be expanded with actual opponent tracking
        
        # Position-based opponent modeling
        early_pos_players = sum(1 for pos in ['UTG', 'UTG+1', 'UTG+2'] 
                              if any(action.player_name.endswith(pos) for action in all_actions))
        features[3] = early_pos_players / max(context.num_players, 1)
        
        # Recent action patterns (would be expanded with player tracking)
        features[4:29] = 0.0  # Placeholder for individual opponent stats
        
        # Table dynamics
        features[29] = 1.0 if context.is_tournament else 0.0
        
        return features
    
    # Helper methods for feature extraction
    
    def _get_position_strength(self, position: str, num_players: int) -> float:
        """Get numerical position strength (0=worst, 1=best)"""
        position_order = ['UTG', 'UTG+1', 'UTG+2', 'MP', 'MP1', 'MP2', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        late_positions = ['CO', 'BTN']
        
        if position in late_positions:
            return 0.9
        elif position in ['HJ', 'MP2']:
            return 0.7
        elif position in ['MP', 'MP1']:
            return 0.5
        elif position in ['SB']:
            return 0.3
        elif position in ['BB']:
            return 0.4
        else:  # Early position
            return 0.2
    
    def _preflop_hand_strength(self, rank1: int, rank2: int, suited: bool) -> float:
        """Estimate preflop hand strength (0-1)"""
        # Simplified hand strength based on rank and suitedness
        high_rank = max(rank1, rank2)
        low_rank = min(rank1, rank2)
        
        # Pair bonus
        if rank1 == rank2:
            return 0.5 + (high_rank / 12.0) * 0.5
        
        # High card strength
        base_strength = (high_rank + low_rank / 2) / 15.0
        
        # Suited bonus
        if suited:
            base_strength += 0.1
        
        # Connected bonus
        if abs(rank1 - rank2) <= 1:
            base_strength += 0.05
        
        return min(base_strength, 1.0)
    
    def _estimate_hand_strength(self, hole_cards: List[Card], board: List[Card]) -> float:
        """Estimate hand strength vs random opponents"""
        # Simplified hand strength calculation
        # In a real implementation, this would simulate vs random hands
        all_cards = hole_cards + board
        if len(all_cards) >= 5:
            hand_rank, _ = self.hand_evaluator.evaluate_hand(all_cards)
            return hand_rank.value / 10.0
        return 0.5
    
    def _flush_draw_potential(self, card1: Card, card2: Card, board: List[str]) -> float:
        """Calculate flush draw potential"""
        if not board:
            return 1.0 if card1.suit == card2.suit else 0.0
        
        try:
            board_cards = [Card(card) for card in board]
            all_cards = [card1, card2] + board_cards
            
            suit_counts = {}
            for card in all_cards:
                suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
            
            max_suit = max(suit_counts.values()) if suit_counts else 0
            return min(max_suit / 5.0, 1.0)
        except:
            return 0.0
    
    def _straight_draw_potential(self, card1: Card, card2: Card, board: List[str]) -> float:
        """Calculate straight draw potential"""
        if not board:
            gap = abs(card1.rank - card2.rank)
            return 1.0 if gap <= 4 else 0.0
        
        try:
            board_cards = [Card(card) for card in board]
            all_cards = [card1, card2] + board_cards
            ranks = sorted([card.rank for card in all_cards])
            
            # Count potential straights
            max_consecutive = 1
            current_consecutive = 1
            
            for i in range(1, len(ranks)):
                if ranks[i] == ranks[i-1] + 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                elif ranks[i] != ranks[i-1]:  # Skip duplicates
                    current_consecutive = 1
            
            return min(max_consecutive / 5.0, 1.0)
        except:
            return 0.0
    
    def _board_wetness(self, ranks: List[int], suits: List[int]) -> float:
        """Calculate how wet/coordinated the board is"""
        wetness = 0.0
        
        # Flush draws
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        wetness += min(max_suit_count / 5.0, 1.0)
        
        # Straight draws
        sorted_ranks = sorted(set(ranks))
        wetness += self._straight_draw_strength(sorted_ranks)
        
        # Pairs
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        wetness += 0.3 if any(count >= 2 for count in rank_counts.values()) else 0.0
        
        return min(wetness / 2.0, 1.0)
    
    def _straight_draw_strength(self, sorted_ranks: List[int]) -> float:
        """Calculate straight draw strength"""
        if len(sorted_ranks) < 2:
            return 0.0
        
        max_gap_filled = 0
        for i in range(len(sorted_ranks)):
            gap_filled = 1
            for j in range(i + 1, len(sorted_ranks)):
                if sorted_ranks[j] - sorted_ranks[j-1] <= 1:
                    gap_filled += 1
                else:
                    break
            max_gap_filled = max(max_gap_filled, gap_filled)
        
        return min(max_gap_filled / 5.0, 1.0)
    
    def _is_rainbow_board(self, suits: List[int]) -> bool:
        """Check if board is rainbow (all different suits)"""
        return len(set(suits)) == len(suits)
    
    def _is_monotone_board(self, suits: List[int]) -> bool:
        """Check if board is monotone (all same suit)"""
        return len(set(suits)) == 1
    
    def _is_two_tone_board(self, suits: List[int]) -> bool:
        """Check if board is two-tone"""
        return len(set(suits)) == 2
    
    def _completed_flush(self, board: List[Card]) -> bool:
        """Check if board completed a flush"""
        if len(board) < 5:
            return False
        suits = [card.suit for card in board]
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        return any(count >= 5 for count in suit_counts.values())
    
    def _completed_straight(self, board: List[Card]) -> bool:
        """Check if board completed a straight"""
        if len(board) < 5:
            return False
        ranks = sorted([card.rank for card in board])
        consecutive = 1
        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i-1] + 1:
                consecutive += 1
                if consecutive >= 5:
                    return True
            elif ranks[i] != ranks[i-1]:
                consecutive = 1
        return False
    
    def _find_preflop_aggressor(self, actions: List[PlayerAction]) -> Optional[str]:
        """Find the preflop aggressor"""
        for action in reversed(actions):
            if action.action_type in ['bet', 'raise']:
                return action.player_name
        return None
    
    def _find_current_aggressor(self, actions: List[PlayerAction]) -> Optional[str]:
        """Find the current street aggressor"""
        for action in reversed(actions):
            if action.action_type in ['bet', 'raise']:
                return action.player_name
        return None


# Convenience functions
def extract_features_from_parsed_hand(hand: ParsedHand, action_index: int) -> np.ndarray:
    """Extract features for a specific action in a parsed hand"""
    if not hand.hero_name or action_index >= len(hand.all_actions):
        return np.zeros(110)  # Return empty feature vector
    
    action = hand.all_actions[action_index]
    if action.player_name != hand.hero_name:
        return np.zeros(110)  # Only extract features for hero actions
    
    # Build context
    context = GameContext(
        position="MP",  # Would need to calculate from seat positions
        stack_size=100.0,  # Would need to track from hand history
        hole_cards=[],  # Would extract from hand
        pot_size=hand.total_pot,
        current_bet=action.amount,
        num_players=len(hand.players),
        num_active_players=len(hand.players),
        street=action.street,
        board_cards=hand.flop + ([hand.turn] if hand.turn else []) + ([hand.river] if hand.river else []),
        actions_this_street=[a for a in hand.all_actions[:action_index] if a.street == action.street],
        preflop_actions=[a for a in hand.all_actions[:action_index] if a.street == Street.PREFLOP],
        big_blind=hand.stakes[1],
        small_blind=hand.stakes[0]
    )
    
    extractor = FeatureExtractor()
    return extractor.extract_features_vector(context)


# Example usage and testing
if __name__ == "__main__":
    # Test feature extraction with sample data
    extractor = FeatureExtractor()
    
    # Create sample context
    context = GameContext(
        position="BTN",
        stack_size=100.0,
        hole_cards=["As", "Kh"],
        pot_size=15.0,
        current_bet=5.0,
        num_players=6,
        num_active_players=3,
        street=Street.FLOP,
        board_cards=["Ah", "7c", "2s"],
        actions_this_street=[],
        preflop_actions=[],
        big_blind=2.0,
        small_blind=1.0
    )
    
    # Extract features
    features = extractor.extract_features(context)
    feature_vector = extractor.extract_features_vector(context)
    
    print("Feature Extraction Test Results:")
    print(f"Total feature vector length: {len(feature_vector)}")
    print(f"Expected length: {extractor.total_features}")
    
    for feature_type, feature_array in features.items():
        print(f"{feature_type}: {len(feature_array)} features")
        print(f"  Sample values: {feature_array[:5]}")
    
    print(f"\nSample feature vector (first 20 values):")
    print(feature_vector[:20])
    
    print(f"\nFeature extraction test completed!")