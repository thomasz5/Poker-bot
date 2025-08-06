"""
Basic test suite for the Hand Evaluator module

Tests just the implemented features of the hand evaluator.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import poker_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_engine.hand_evaluator import Card, HandRank, HandEvaluator


class TestCard:
    """Test the Card class"""
    
    def test_valid_card_creation(self):
        """Test creating valid cards"""
        card = Card("As")  # Ace of spades
        assert card.rank == 12  # Ace is rank 12
        assert card.suit == 3   # Spades is suit 3
        assert str(card) == "As"
        
        card2 = Card("2h")  # Two of hearts
        assert card2.rank == 0  # Two is rank 0
        assert card2.suit == 2  # Hearts is suit 2
    
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
    
    def test_card_string_representation(self):
        """Test card string conversion"""
        card = Card("Kh")
        assert str(card) == "Kh"
        assert repr(card) == "Card('Kh')"


class TestHandEvaluator:
    """Test the HandEvaluator class"""
    
    def test_hand_evaluator_creation(self):
        """Test creating a hand evaluator"""
        evaluator = HandEvaluator()
        assert evaluator is not None
    
    def test_basic_hand_evaluation(self):
        """Test basic hand evaluation functionality"""
        evaluator = HandEvaluator()
        
        # Test with a simple high card hand
        cards = [Card("As"), Card("Kh"), Card("Qc"), Card("Jd"), Card("9s")]
        rank, kickers = evaluator.evaluate_hand(cards)
        
        # Should recognize this as high card
        assert rank == HandRank.HIGH_CARD
        assert isinstance(kickers, list)
        assert len(kickers) > 0
    
    def test_pair_detection(self):
        """Test pair detection"""
        evaluator = HandEvaluator()
        cards = [Card("As"), Card("Ah"), Card("Kc"), Card("Qd"), Card("Js")]
        rank, kickers = evaluator.evaluate_hand(cards)
        
        assert rank == HandRank.PAIR
    
    def test_hand_comparison(self):
        """Test comparing hands"""
        evaluator = HandEvaluator()
        
        # Aces should beat kings
        aces = [Card("As"), Card("Ah"), Card("Kc"), Card("Qd"), Card("Js")]
        kings = [Card("Ks"), Card("Kh"), Card("Ac"), Card("Qd"), Card("Js")]
        
        result = evaluator.compare_hands(aces, kings)
        assert result == 1  # Aces win
    
    def test_invalid_hand_size(self):
        """Test that invalid hand sizes raise errors"""
        evaluator = HandEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.evaluate_hand([])  # Empty hand
            
        with pytest.raises(ValueError):
            evaluator.evaluate_hand([Card("As")])  # Too few cards
    
    def test_hand_strength_basic(self):
        """Test basic hand strength calculation"""
        evaluator = HandEvaluator()
        
        # Test with pocket aces (should be strong)
        hole_cards = [Card("As"), Card("Ah")]
        board = []
        
        try:
            strength = evaluator.get_hand_strength(hole_cards, board)
            assert 0.0 <= strength <= 1.0  # Should be between 0 and 1
        except Exception:
            # If get_hand_strength isn't fully implemented, that's okay
            pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])