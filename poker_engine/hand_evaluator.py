"""Fast poker hand evaluator using lookup tables and bit manipulation."""

from typing import List, Tuple
from enum import IntEnum
import itertools


class HandRank(IntEnum):
    """Hand rankings from weakest to strongest."""
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class Card:
    """Represents a playing card."""
    
    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'  # clubs, diamonds, hearts, spades
    
    def __init__(self, card_str: str):
        """Initialize card from string like 'As' (Ace of spades)."""
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank_char, suit_char = card_str[0], card_str[1]
        
        if rank_char not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank_char}")
        if suit_char not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit_char}")
            
        self.rank = self.RANKS.index(rank_char)
        self.suit = self.SUITS.index(suit_char)
        self.card_str = card_str
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __str__(self):
        return self.card_str
    
    def __repr__(self):
        return f"Card('{self.card_str}')"


class HandEvaluator:
    """Efficient poker hand evaluator."""
    
    def __init__(self):
        """Initialize the hand evaluator with lookup tables."""
        self._initialize_lookup_tables()
    
    def _initialize_lookup_tables(self):
        """Pre-compute lookup tables for fast evaluation."""
        # This is a simplified version - in production, you'd use
        # more sophisticated lookup tables for maximum speed
        self.straight_ranks = [
            [0, 1, 2, 3, 4],      # A-2-3-4-5 (wheel)
            [1, 2, 3, 4, 5],      # 2-3-4-5-6
            [2, 3, 4, 5, 6],      # 3-4-5-6-7
            [3, 4, 5, 6, 7],      # 4-5-6-7-8
            [4, 5, 6, 7, 8],      # 5-6-7-8-9
            [5, 6, 7, 8, 9],      # 6-7-8-9-T
            [6, 7, 8, 9, 10],     # 7-8-9-T-J
            [7, 8, 9, 10, 11],    # 8-9-T-J-Q
            [8, 9, 10, 11, 12],   # 9-T-J-Q-K
            [9, 10, 11, 12, 0],   # T-J-Q-K-A (broadway)
        ]
    
    def evaluate_hand(self, cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a poker hand and return (hand_rank, kickers).
        
        Args:
            cards: List of 5-7 cards
            
        Returns:
            Tuple of (HandRank, list of kicker ranks in descending order)
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate")
        
        # For 6-7 cards, find the best 5-card combination
        if len(cards) > 5:
            best_hand = None
            best_rank = HandRank.HIGH_CARD
            best_kickers = []
            
            for combo in itertools.combinations(cards, 5):
                rank, kickers = self._evaluate_five_cards(list(combo))
                if rank > best_rank or (rank == best_rank and kickers > best_kickers):
                    best_hand = combo
                    best_rank = rank
                    best_kickers = kickers
            
            return best_rank, best_kickers
        
        return self._evaluate_five_cards(cards)
    
    def _evaluate_five_cards(self, cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Evaluate exactly 5 cards."""
        ranks = [card.rank for card in cards]
        suits = [card.suit for card in cards]
        
        # Count ranks
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Sort by count, then by rank (descending)
        sorted_counts = sorted(rank_counts.items(), 
                             key=lambda x: (x[1], x[0]), reverse=True)
        
        counts = [count for _, count in sorted_counts]
        sorted_ranks = [rank for rank, _ in sorted_counts]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        is_straight, straight_high = self._is_straight(ranks)
        
        # Determine hand rank and kickers
        if is_straight and is_flush:
            if straight_high == 12:  # Ace high straight flush
                return HandRank.ROYAL_FLUSH, [12]
            else:
                return HandRank.STRAIGHT_FLUSH, [straight_high]
        
        elif counts == [4, 1]:
            # Four of a kind
            return HandRank.FOUR_OF_A_KIND, sorted_ranks
        
        elif counts == [3, 2]:
            # Full house
            return HandRank.FULL_HOUSE, sorted_ranks
        
        elif is_flush:
            return HandRank.FLUSH, sorted(ranks, reverse=True)
        
        elif is_straight:
            return HandRank.STRAIGHT, [straight_high]
        
        elif counts == [3, 1, 1]:
            # Three of a kind
            return HandRank.THREE_OF_A_KIND, sorted_ranks
        
        elif counts == [2, 2, 1]:
            # Two pair
            return HandRank.TWO_PAIR, sorted_ranks
        
        elif counts == [2, 1, 1, 1]:
            # One pair
            return HandRank.PAIR, sorted_ranks
        
        else:
            # High card
            return HandRank.HIGH_CARD, sorted(ranks, reverse=True)
    
    def _is_straight(self, ranks: List[int]) -> Tuple[bool, int]:
        """Check if ranks form a straight, return (is_straight, high_card)."""
        unique_ranks = sorted(set(ranks))
        
        if len(unique_ranks) != 5:
            return False, 0
        
        # Check for regular straight
        if unique_ranks[-1] - unique_ranks[0] == 4:
            return True, unique_ranks[-1]
        
        # Check for wheel (A-2-3-4-5)
        if unique_ranks == [0, 1, 2, 3, 12]:  # A=12, but low in wheel
            return True, 3  # 5-high straight
        
        return False, 0
    
    def compare_hands(self, hand1: List[Card], hand2: List[Card]) -> int:
        """
        Compare two hands.
        
        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        rank1, kickers1 = self.evaluate_hand(hand1)
        rank2, kickers2 = self.evaluate_hand(hand2)
        
        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1
        else:
            # Same hand rank, compare kickers
            for k1, k2 in zip(kickers1, kickers2):
                if k1 > k2:
                    return 1
                elif k1 < k2:
                    return -1
            return 0
    
    def get_hand_strength(self, hand: List[Card], board: List[Card]) -> float:
        """
        Calculate hand strength (0.0 to 1.0) based on current board.
        
        This is a simplified version - in production you'd use more
        sophisticated equity calculations.
        """
        all_cards = hand + board
        rank, kickers = self.evaluate_hand(all_cards)
        
        # Simple hand strength mapping (needs refinement)
        strength_map = {
            HandRank.HIGH_CARD: 0.1,
            HandRank.PAIR: 0.2,
            HandRank.TWO_PAIR: 0.35,
            HandRank.THREE_OF_A_KIND: 0.5,
            HandRank.STRAIGHT: 0.65,
            HandRank.FLUSH: 0.7,
            HandRank.FULL_HOUSE: 0.85,
            HandRank.FOUR_OF_A_KIND: 0.95,
            HandRank.STRAIGHT_FLUSH: 0.99,
            HandRank.ROYAL_FLUSH: 1.0,
        }
        
        base_strength = strength_map[rank]
        
        # Adjust for kickers (simplified)
        if kickers:
            kicker_bonus = sum(kickers) / (13 * len(kickers)) * 0.1
            base_strength = min(1.0, base_strength + kicker_bonus)
        
        return base_strength


# Example usage and testing
if __name__ == "__main__":
    evaluator = HandEvaluator()
    
    # Test royal flush
    royal_flush = [Card('As'), Card('Ks'), Card('Qs'), Card('Js'), Card('Ts')]
    rank, kickers = evaluator.evaluate_hand(royal_flush)
    print(f"Royal flush: {rank}, kickers: {kickers}")
    
    # Test pair
    pair = [Card('As'), Card('Ah'), Card('Ks'), Card('Qj'), Card('Ts')]
    rank, kickers = evaluator.evaluate_hand(pair)
    print(f"Pair of Aces: {rank}, kickers: {kickers}")
    
    # Test hand comparison
    hand1 = [Card('As'), Card('Ah')]
    hand2 = [Card('Ks'), Card('Kh')]
    board = [Card('2c'), Card('3d'), Card('4s'), Card('5h'), Card('6c')]
    
    result = evaluator.compare_hands(hand1 + board, hand2 + board)
    print(f"AA vs KK comparison: {result}")
    
    # Test hand strength
    strength = evaluator.get_hand_strength(hand1, board[:3])
    print(f"AA hand strength on 234 board: {strength:.3f}") 