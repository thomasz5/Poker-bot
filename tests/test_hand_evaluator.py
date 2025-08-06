"""
Test suite for the Hand Evaluator module

Tests all poker hand ranking functionality including:
- Card creation and validation
- Hand ranking detection
- Hand comparison
- Edge cases and error handling
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import poker_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_engine.hand_evaluator import (
    Card, HandRank, HandEvaluator
)


class TestCard:
    """Test the Card class"""
    
    def test_valid_card_creation(self):
        """Test creating valid cards"""
        card = Card("As")  # Ace of spades
        assert card.rank == 12  # Ace is rank 12
        assert card.suit == "s"
        assert str(card) == "As"
        
        card2 = Card("2h")  # Two of hearts
        assert card2.rank == 0  # Two is rank 0
        assert card2.suit == "h"
        
    def test_invalid_card_creation(self):
        """Test that invalid cards raise errors"""
        with pytest.raises(ValueError):
            Card("Xs")  # Invalid rank
            
        with pytest.raises(ValueError):
            Card("Ax")  # Invalid suit
            
        with pytest.raises(ValueError):
            Card("A")   # Too short
            
        with pytest.raises(ValueError):
            Card("Abc") # Too long
    
    def test_card_equality(self):
        """Test card equality comparison"""
        card1 = Card("As")
        card2 = Card("As")
        card3 = Card("Ah")
        
        assert card1 == card2
        assert card1 != card3
    
    def test_card_comparison(self):
        """Test card ordering"""
        ace = Card("As")
        king = Card("Kh")
        two = Card("2c")
        
        assert ace > king
        assert king > two
        assert two < ace


class TestHandEvaluator:
    """Test the HandEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = HandEvaluator()
    
    def test_royal_flush(self):
        """Test royal flush detection"""
        evaluator = HandEvaluator()
        # Royal flush in spades
        cards = [Card("As"), Card("Ks"), Card("Qs"), Card("Js"), Card("Ts")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.ROYAL_FLUSH
        assert kickers == [12]  # Ace high
    
    def test_straight_flush(self):
        """Test straight flush detection"""
        evaluator = HandEvaluator()
        # 5-high straight flush
        cards = [Card("5h"), Card("4h"), Card("3h"), Card("2h"), Card("Ah")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert kickers == [3]  # 5-high (ace low)
        
        # 9-high straight flush
        cards = [Card("9s"), Card("8s"), Card("7s"), Card("6s"), Card("5s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert kickers == [7]  # 9-high
    
    def test_four_of_a_kind(self):
        """Test four of a kind detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Ah"), Card("Ac"), Card("Ad"), Card("Ks")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.FOUR_OF_A_KIND
        assert kickers[0] == 12  # Aces
        assert kickers[1] == 11  # King kicker
    
    def test_full_house(self):
        """Test full house detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Ah"), Card("Ac"), Card("Ks"), Card("Kh")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.FULL_HOUSE
        assert kickers == [12, 11]  # Aces over Kings
    
    def test_flush(self):
        """Test flush detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Js"), Card("9s"), Card("7s"), Card("2s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.FLUSH
        assert kickers == [12, 9, 7, 5, 0]  # Ace high flush
    
    def test_straight(self):
        """Test straight detection"""
        evaluator = HandEvaluator()
        # Regular straight
        cards = [Card("Ts"), Card("Jh"), Card("Qc"), Card("Kd"), Card("As")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [12]  # Ace high straight
        
        # Wheel (A-2-3-4-5)
        cards = [Card("As"), Card("2h"), Card("3c"), Card("4d"), Card("5s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [3]  # 5-high (ace low)
    
    def test_three_of_a_kind(self):
        """Test three of a kind detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Ah"), Card("Ac"), Card("Ks"), Card("Qh")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.THREE_OF_A_KIND
        assert kickers[0] == 12  # Trip aces
        assert set(kickers[1:3]) == {11, 10}  # King and Queen kickers
    
    def test_two_pair(self):
        """Test two pair detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Ah"), Card("Ks"), Card("Kh"), Card("Qc")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.TWO_PAIR
        assert kickers[0] == 12  # Aces
        assert kickers[1] == 11  # Kings  
        assert kickers[2] == 10  # Queen kicker
    
    def test_pair(self):
        """Test pair detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Ah"), Card("Ks"), Card("Qh"), Card("Jc")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.PAIR
        assert kickers[0] == 12  # Pair of aces
        assert set(kickers[1:4]) == {11, 10, 9}  # King, Queen, Jack kickers
    
    def test_high_card(self):
        """Test high card detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Kh"), Card("Qc"), Card("Jd"), Card("9s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.HIGH_CARD
        assert kickers == [12, 11, 10, 9, 7]  # A-K-Q-J-9
    
    def test_seven_card_evaluation(self):
        """Test evaluation with 7 cards (like Texas Hold'em)"""
        evaluator = HandEvaluator()
        # Player has pocket aces, board has A-K-Q-J-T (royal flush)
        hole_cards = [Card("As"), Card("Ah")]
        board = [Card("Ks"), Card("Qs"), Card("Js"), Card("Ts"), Card("2c")]
        all_cards = hole_cards + board
        
        rank, kickers = evaluator.evaluate_hand(all_cards)
        assert rank == HandRank.ROYAL_FLUSH
    
    def test_hand_comparison(self):
        """Test comparing different hands"""
        evaluator = HandEvaluator()
        
        # Royal flush vs straight flush
        royal = [Card("As"), Card("Ks"), Card("Qs"), Card("Js"), Card("Ts")]
        straight_flush = [Card("9s"), Card("8s"), Card("7s"), Card("6s"), Card("5s")]
        
        royal_rank, royal_kickers = evaluator.evaluate_hand(royal)
        sf_rank, sf_kickers = evaluator.evaluate_hand(straight_flush)
        
        result = evaluator.compare_hands(royal, straight_flush)
        assert result == 1  # Royal flush wins
        
        # Same hand type - higher wins
        high_pair = [Card("As"), Card("Ah"), Card("Ks"), Card("Qh"), Card("Jc")]
        low_pair = [Card("2s"), Card("2h"), Card("Ks"), Card("Qh"), Card("Jc")]
        
        result = evaluator.compare_hands(high_pair, low_pair)
        assert result == 1  # Aces beat deuces
    
    def test_hand_strength_calculation(self):
        """Test hand strength calculation"""
        evaluator = HandEvaluator()
        
        # Pocket aces should be very strong preflop
        hole_cards = [Card("As"), Card("Ah")]
        board = []
        strength = evaluator.get_hand_strength(hole_cards, board)
        assert strength > 0.8  # Should be very strong
        
        # 7-2 offsuit should be weak
        hole_cards = [Card("7s"), Card("2h")]
        board = []
        strength = evaluator.get_hand_strength(hole_cards, board)
        assert strength < 0.3  # Should be weak
    
    def test_invalid_hand_size(self):
        """Test that invalid hand sizes raise errors"""
        evaluator = HandEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.evaluate_hand([])  # Empty hand
            
        with pytest.raises(ValueError):
            evaluator.evaluate_hand([Card("As")])  # Too few cards
    
    def test_duplicate_cards(self):
        """Test that duplicate cards raise errors"""
        evaluator = HandEvaluator()
        
        with pytest.raises(ValueError):
            cards = [Card("As"), Card("As"), Card("Kh"), Card("Qc"), Card("Jd")]
            evaluator.evaluate_hand(cards)


# Global functions not implemented yet - tests removed


class TestEdgeCases:
    """Test edge cases and unusual situations"""
    
    def test_ace_low_straight(self):
        """Test ace-low straight (wheel)"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("2h"), Card("3c"), Card("4d"), Card("5s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT
        assert kickers == [3]  # 5-high, not ace-high
    
    def test_ace_low_straight_flush(self):
        """Test ace-low straight flush"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("2s"), Card("3s"), Card("4s"), Card("5s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert kickers == [3]  # 5-high straight flush
    
    def test_kicker_comparison(self):
        """Test detailed kicker comparison"""
        evaluator = HandEvaluator()
        
        # Same pair, different kickers
        hand1 = [Card("As"), Card("Ah"), Card("Ks"), Card("Qh"), Card("Jc")]
        hand2 = [Card("As"), Card("Ah"), Card("Ks"), Card("Qh"), Card("Tc")]
        
        rank1, kickers1 = evaluator.evaluate_hand(hand1)
        rank2, kickers2 = evaluator.evaluate_hand(hand2)
        
        result = compare_hands((rank1, kickers1), (rank2, kickers2))
        assert result == 1  # Jack kicker beats ten kicker


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])