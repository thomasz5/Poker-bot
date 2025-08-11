"""
Data Processing Pipeline for Poker Bot

Integrates hand history parsing and feature extraction to create 
training datasets for neural networks.

Pipeline:
1. Parse hand history files
2. Extract game contexts for each decision point
3. Generate feature vectors
4. Save training data to database
5. Create batches for neural network training
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.parsers.hand_history_parser import HandHistoryParser, ParsedHand, PlayerAction, Street
    from training.feature_extractor import FeatureExtractor, GameContext
    from data.storage.game_data_manager import GameDataManager
except ImportError as e:
    print(f"Import error: {e}")


@dataclass
class TrainingExample:
    """Single training example for neural network"""
    features: np.ndarray
    target_action: str
    target_amount: float
    hand_id: str
    action_sequence: int
    street: str
    position: str
    stack_size: float
    pot_size: float


class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self, database_url: str = "sqlite:///poker_training.db"):
        self.hand_parser = HandHistoryParser()
        self.feature_extractor = FeatureExtractor()
        self.data_manager = GameDataManager(database_url)
        
        # Action type mapping for neural network targets
        self.action_to_id = {
            'fold': 0,
            'check': 1, 
            'call': 2,
            'bet': 3,
            'raise': 4,
            'all_in': 5
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}
    
    def process_hand_history_file(self, file_path: str, save_to_db: bool = True) -> List[TrainingExample]:
        """Process a complete hand history file"""
        print(f"Processing hand history file: {file_path}")
        
        # Parse hands
        parsed_hands = self.hand_parser.parse_file(file_path)
        print(f"Parsed {len(parsed_hands)} hands")
        
        # Extract training examples
        training_examples = []
        session_id = None
        
        if save_to_db:
            # Create database session for this file
            session_id = self.data_manager.create_game_session(
                session_type="training_data",
                small_blind=parsed_hands[0].stakes[0] if parsed_hands else 1.0,
                big_blind=parsed_hands[0].stakes[1] if parsed_hands else 2.0
            )
        
        for hand in parsed_hands:
            hand_examples = self.process_single_hand(hand, session_id, save_to_db)
            training_examples.extend(hand_examples)
        
        print(f"Generated {len(training_examples)} training examples")
        return training_examples
    
    def process_single_hand(self, hand: ParsedHand, session_id: str = None, save_to_db: bool = True) -> List[TrainingExample]:
        """Process a single hand and extract training examples"""
        examples = []
        
        # Determine hero if not explicitly provided
        hero: Any = None
        if getattr(hand, 'hero_name', None):
            for player in hand.players:
                if getattr(player, 'name', None) == hand.hero_name:
                    hero = player
                    break
        else:
            # Fallback to player named 'Hero' or with known hole cards
            for player in hand.players:
                if getattr(player, 'name', None) == 'Hero' or (getattr(player, 'hole_cards', None) and len(player.hole_cards) == 2):
                    hero = player
                    break
            # As a last resort, choose first player
            if hero is None and hand.players:
                hero = hand.players[0]
        
        # Find hero player
        # If players include is_hero flag, prefer it
        if hero is None:
            for player in hand.players:
                if getattr(player, 'is_hero', False):
                    hero = player
                    break
        
        if not hero:
            return examples
        
        # Save hand to database if requested
        hand_id_db = None
        session_player_id = None
        
        if save_to_db and session_id:
            try:
                # Add hero to session
                session_player_id = self.data_manager.add_player_to_session(
                    session_id=session_id,
                    player_name=hero.name,
                    seat_number=hero.seat,
                    starting_stack=hero.starting_stack,
                    is_bot=False
                )
                
                # Start hand in database
                hand_id_db = self.data_manager.start_new_hand(
                    session_id=session_id,
                    hand_number=int(hand.hand_id) % 1000000,  # Truncate for storage
                    dealer_position=hand.button_seat,
                    small_blind_seat=1,  # Simplified
                    big_blind_seat=2     # Simplified
                )
            except Exception as e:
                print(f"Error saving hand to database: {e}")
        
        # Process each hero action; if none recorded on player, use all_actions filtered by hero name
        hero_actions = getattr(hero, 'actions', None) or []
        if not hero_actions:
            hero_actions = [a for a in hand.all_actions if getattr(a, 'player_name', None) == getattr(hero, 'name', None)]
        # Process each hero action
        for action_index, action in enumerate(hero_actions):
            try:
                context = self._build_game_context(hand, hero, action, action_index)
                features = self.feature_extractor.extract_features_vector(context)
                
                # Create training example
                example = TrainingExample(
                    features=features,
                    target_action=action.action_type,
                    target_amount=action.amount,
                    hand_id=hand.hand_id,
                    action_sequence=action_index,
                    street=action.street.value,
                    position=context.position,
                    stack_size=context.stack_size,
                    pot_size=context.pot_size
                )
                examples.append(example)
                
                # Save to database
                if save_to_db and session_player_id and hand_id_db:
                    try:
                        action_id = self.data_manager.record_action(
                            hand_id=hand_id_db,
                            session_player_id=session_player_id,
                            action_sequence=action_index,
                            street=action.street.value,
                            action_type=action.action_type,
                            amount=action.amount,
                            pot_size_before=context.pot_size,
                            current_bet=context.current_bet
                        )
                        
                        # Save features
                        feature_dict = self.feature_extractor.extract_features(context)
                        # Convert NumPy arrays to lists for JSON serialization
                        serializable_features = {}
                        for key, value in feature_dict.items():
                            if hasattr(value, 'tolist'):  # NumPy array
                                serializable_features[key] = value.tolist()
                            else:
                                serializable_features[key] = value
                        
                        self.data_manager.save_training_features(
                            action_id=action_id,
                            features=serializable_features,
                            target_action=action.action_type,
                            target_amount=action.amount,
                            feature_version="1.0"
                        )
                    except Exception as e:
                        print(f"Error saving action to database: {e}")
                        
            except Exception as e:
                # Skip actions that cannot build features (e.g., missing optional fields)
                continue
        
        # Fallback: if no examples were created, attempt to create one from the first hero action
        if not examples and hero_actions:
            action = hero_actions[0]
            try:
                context = self._build_game_context(hand, hero, action, 0)
                features = self.feature_extractor.extract_features_vector(context)
                examples.append(TrainingExample(
                    features=features,
                    target_action=action.action_type,
                    target_amount=action.amount,
                    hand_id=hand.hand_id,
                    action_sequence=0,
                    street=(action.street.value if hasattr(action.street, 'value') else str(action.street)),
                    position=context.position,
                    stack_size=context.stack_size,
                    pot_size=context.pot_size
                ))
            except Exception:
                pass
        
        return examples
    
    def _build_game_context(self, hand: ParsedHand, hero: Any, action: PlayerAction, action_index: int) -> GameContext:
        """Build game context for a specific action"""
        
        # Calculate position
        position = self._calculate_position(hero.seat, hand.button_seat, len(hand.players))
        
        # Get board cards up to this street
        board_cards = []
        if action.street in [Street.FLOP, Street.TURN, Street.RIVER]:
            board_cards.extend(hand.flop)
        if action.street in [Street.TURN, Street.RIVER]:
            if hand.turn:
                board_cards.append(hand.turn)
        if action.street == Street.RIVER:
            if hand.river:
                board_cards.append(hand.river)
        
        # Get actions up to this point
        all_actions_before = hand.all_actions[:hand.all_actions.index(action)]
        def _street_name(s):
            return s.value if hasattr(s, 'value') else str(s)
        actions_this_street = [a for a in all_actions_before if _street_name(a.street) == _street_name(action.street)]
        preflop_actions = [a for a in all_actions_before if _street_name(a.street) == 'preflop']
        
        # Estimate pot size at this point (simplified): blinds plus contributions so far
        pot_before_action = float(hand.stakes[0] + hand.stakes[1])
        for prev_action in all_actions_before:
            if prev_action.action_type in ['bet', 'raise', 'call']:
                pot_before_action += float(prev_action.amount)
        
        # Estimate current bet
        current_bet = 0.0
        for prev_action in reversed(actions_this_street):
            if prev_action.action_type in ['bet', 'raise']:
                current_bet = float(prev_action.amount)
                break
        
        # Count active players (simplified)
        folded_players = set()
        for prev_action in all_actions_before:
            if prev_action.action_type == 'fold':
                folded_players.add(prev_action.player_name)
        
        num_active_players = len(hand.players) - len(folded_players)
        
        return GameContext(
            position=position,
            stack_size=float(getattr(hero, 'starting_stack', 100.0)),
            hole_cards=list(getattr(hero, 'hole_cards', [])),
            pot_size=float(pot_before_action),
            current_bet=float(current_bet),
            num_players=len(hand.players),
            num_active_players=int(num_active_players),
            street=action.street,
            board_cards=list(board_cards),
            actions_this_street=list(actions_this_street),
            preflop_actions=list(preflop_actions),
            big_blind=float(hand.stakes[1]),
            small_blind=float(hand.stakes[0])
        )
    
    def _calculate_position(self, seat: int, button_seat: int, num_players: int) -> str:
        """Calculate position name from seat numbers"""
        seats_from_button = (seat - button_seat) % num_players
        
        if num_players == 2:
            return "SB" if seats_from_button == 0 else "BB"
        elif num_players <= 6:
            positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
            return positions[seats_from_button] if seats_from_button < len(positions) else "MP"
        else:
            positions = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP1", "MP2", "CO", "HJ"]
            return positions[seats_from_button] if seats_from_button < len(positions) else "MP"
    
    def create_training_dataset(self, hand_history_files: List[str]) -> Dict[str, np.ndarray]:
        """Create complete training dataset from multiple files"""
        print("Creating training dataset...")
        
        all_examples = []
        for file_path in hand_history_files:
            if os.path.exists(file_path):
                examples = self.process_hand_history_file(file_path, save_to_db=True)
                all_examples.extend(examples)
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not all_examples:
            print("No training examples generated!")
            return {}
        
        # Convert to numpy arrays
        features = np.array([ex.features for ex in all_examples])
        action_targets = np.array([self.action_to_id.get(ex.target_action, 0) for ex in all_examples])
        amount_targets = np.array([ex.target_amount for ex in all_examples])
        
        # Additional metadata
        metadata = {
            'hand_ids': [ex.hand_id for ex in all_examples],
            'streets': [ex.street for ex in all_examples],
            'positions': [ex.position for ex in all_examples],
            'stack_sizes': [ex.stack_size for ex in all_examples],
            'pot_sizes': [ex.pot_size for ex in all_examples]
        }
        
        dataset = {
            'features': features,
            'action_targets': action_targets,
            'amount_targets': amount_targets,
            'metadata': metadata,
            'feature_dim': features.shape[1] if len(features) > 0 else 0,
            'num_actions': len(self.action_to_id),
            'action_mapping': self.action_to_id
        }
        
        print(f"Dataset created with {len(all_examples)} examples")
        print(f"Feature dimension: {dataset['feature_dim']}")
        print(f"Action distribution: {np.bincount(action_targets)}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, np.ndarray], file_path: str):
        """Save dataset to file"""
        np.savez_compressed(file_path, **dataset)
        print(f"Dataset saved to {file_path}")
    
    def load_dataset(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load dataset from file"""
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.keys()}
    
    def get_batch(self, dataset: Dict[str, np.ndarray], batch_size: int = 32, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of training data"""
        features = dataset['features']
        action_targets = dataset['action_targets']
        amount_targets = dataset['amount_targets']
        
        if shuffle:
            indices = np.random.permutation(len(features))
            features = features[indices]
            action_targets = action_targets[indices]
            amount_targets = amount_targets[indices]
        
        for i in range(0, len(features), batch_size):
            yield (
                features[i:i+batch_size],
                action_targets[i:i+batch_size],
                amount_targets[i:i+batch_size]
            )
    
    def analyze_dataset(self, dataset: Dict[str, np.ndarray]):
        """Analyze dataset statistics"""
        print("\n=== Dataset Analysis ===")
        
        features = dataset['features']
        action_targets = dataset['action_targets']
        amount_targets = dataset['amount_targets']
        metadata = dataset['metadata']
        
        print(f"Total examples: {len(features)}")
        print(f"Feature dimension: {features.shape[1]}")
        
        # Action distribution
        print("\nAction distribution:")
        action_counts = np.bincount(action_targets)
        for action_id, count in enumerate(action_counts):
            action_name = self.id_to_action.get(action_id, f"unknown_{action_id}")
            percentage = count / len(action_targets) * 100
            print(f"  {action_name}: {count} ({percentage:.1f}%)")
        
        # Street distribution
        print("\nStreet distribution:")
        street_counts = {}
        for street in metadata['streets']:
            street_counts[street] = street_counts.get(street, 0) + 1
        for street, count in street_counts.items():
            percentage = count / len(metadata['streets']) * 100
            print(f"  {street}: {count} ({percentage:.1f}%)")
        
        # Position distribution
        print("\nPosition distribution:")
        position_counts = {}
        for position in metadata['positions']:
            position_counts[position] = position_counts.get(position, 0) + 1
        for position, count in position_counts.items():
            percentage = count / len(metadata['positions']) * 100
            print(f"  {position}: {count} ({percentage:.1f}%)")
        
        # Amount statistics
        non_zero_amounts = amount_targets[amount_targets > 0]
        if len(non_zero_amounts) > 0:
            print(f"\nBet/Raise amounts:")
            print(f"  Mean: ${np.mean(non_zero_amounts):.2f}")
            print(f"  Median: ${np.median(non_zero_amounts):.2f}")
            print(f"  Min: ${np.min(non_zero_amounts):.2f}")
            print(f"  Max: ${np.max(non_zero_amounts):.2f}")


# Example usage and testing
if __name__ == "__main__":
    # Test data processing pipeline
    processor = DataProcessor("sqlite:///test_training.db")
    
    # Test with sample files
    sample_files = [
        "data/sample_hands/ggpoker_sample.txt",
        "data/sample_hands/pokerstars_sample.txt"
    ]
    
    print("Testing Data Processing Pipeline")
    print("=" * 50)
    
    # Create dataset
    dataset = processor.create_training_dataset(sample_files)
    
    if dataset:
        # Analyze dataset
        processor.analyze_dataset(dataset)
        
        # Save dataset
        processor.save_dataset(dataset, "test_dataset.npz")
        
        # Test batch generation
        print("\nTesting batch generation:")
        batch_count = 0
        for features_batch, action_batch, amount_batch in processor.get_batch(dataset, batch_size=2):
            batch_count += 1
            print(f"Batch {batch_count}: {features_batch.shape}, {action_batch.shape}, {amount_batch.shape}")
            if batch_count >= 3:  # Only show first 3 batches
                break
        
        print(f"\nData processing pipeline test completed!")
        print(f"Generated {len(dataset['features'])} training examples")
    else:
        print("No dataset created - check input files")