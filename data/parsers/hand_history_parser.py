"""
Hand History Parser for Poker Bot

Parses hand history files from major poker sites:
- PokerStars
- GGPoker
- Hold'em Manager exports

Converts raw text files into structured data for AI training.
"""

import re
import os
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class Site(Enum):
    """Supported poker sites"""
    POKERSTARS = "pokerstars"
    GGPOKER = "ggpoker"
    HOLDEM_MANAGER = "holdem_manager"
    UNKNOWN = "unknown"


class Street(Enum):
    """Betting rounds"""
    PREFLOP = "preflop"
    FLOP = "flop" 
    TURN = "turn"
    RIVER = "river"


@dataclass
class PlayerAction:
    """Represents a single player action"""
    player_name: str
    action_type: str  # fold, check, call, bet, raise, all-in
    amount: float = 0.0
    street: Street = Street.PREFLOP
    is_all_in: bool = False
    position: Optional[str] = None


@dataclass
class Player:
    """Player in a hand"""
    name: str
    seat: int
    starting_stack: float
    position: str = ""
    hole_cards: List[str] = field(default_factory=list)
    final_amount: float = 0.0
    is_hero: bool = False
    actions: List[PlayerAction] = field(default_factory=list)


@dataclass
class ParsedHand:
    """Complete parsed poker hand"""
    hand_id: str
    site: Site
    timestamp: datetime
    game_type: str
    stakes: Tuple[float, float]  # (small_blind, big_blind)
    table_name: str
    max_players: int
    button_seat: int
    
    # Players and actions
    players: List[Player] = field(default_factory=list)
    hero_name: str = ""
    
    # Board cards
    flop: List[str] = field(default_factory=list)
    turn: str = ""
    river: str = ""
    
    # Hand result
    total_pot: float = 0.0
    rake: float = 0.0
    winner: str = ""
    winning_hand: str = ""
    
    # All actions in order
    all_actions: List[PlayerAction] = field(default_factory=list)


class HandHistoryParser:
    """Main parser class for different poker sites"""
    
    def __init__(self):
        self.parsers = {
            Site.POKERSTARS: self._parse_pokerstars,
            Site.GGPOKER: self._parse_ggpoker,
        }
    
    def parse_file(self, file_path: str) -> List[ParsedHand]:
        """Parse a hand history file and return list of parsed hands"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hand history file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detect site format
        site = self._detect_site(content)
        
        # Split into individual hands
        hands_text = self._split_hands(content, site)
        
        # Parse each hand
        parsed_hands = []
        for hand_text in hands_text:
            try:
                parsed_hand = self.parsers[site](hand_text)
                parsed_hands.append(parsed_hand)
            except Exception as e:
                print(f"Error parsing hand: {e}")
                continue
        
        return parsed_hands
    
    def _detect_site(self, content: str) -> Site:
        """Detect which poker site format this is"""
        if "PokerStars Hand #" in content:
            return Site.POKERSTARS
        elif "Poker Hand #" in content and "GG" in content.upper():
            return Site.GGPOKER
        elif "Poker Hand #" in content:
            return Site.GGPOKER  # Default to GGPoker for generic format
        else:
            return Site.UNKNOWN
    
    def _split_hands(self, content: str, site: Site) -> List[str]:
        """Split file content into individual hands"""
        if site == Site.POKERSTARS:
            # Split on PokerStars hand headers
            hands = re.split(r'\n\n(?=PokerStars Hand #)', content)
        else:
            # Split on generic hand headers
            hands = re.split(r'\n\n(?=Poker Hand #)', content)
        
        # Remove empty hands
        return [hand.strip() for hand in hands if hand.strip()]
    
    def _parse_pokerstars(self, hand_text: str) -> ParsedHand:
        """Parse a PokerStars hand"""
        lines = hand_text.strip().split('\n')
        
        # Parse header
        header_match = re.match(
            r"PokerStars Hand #(\d+): Hold'em No Limit \(\$([0-9.]+)/\$([0-9.]+) USD\) - (.+)",
            lines[0]
        )
        if not header_match:
            raise ValueError("Invalid PokerStars header")
        
        hand_id = header_match.group(1)
        small_blind = float(header_match.group(2))
        big_blind = float(header_match.group(3))
        timestamp_str = header_match.group(4)
        timestamp = self._parse_timestamp(timestamp_str)
        
        # Parse table info
        table_match = re.match(r"Table '(.+)' (\d+)-max Seat #(\d+) is the button", lines[1])
        if not table_match:
            raise ValueError("Invalid table info")
        
        table_name = table_match.group(1)
        max_players = int(table_match.group(2))
        button_seat = int(table_match.group(3))
        
        # Create hand object
        hand = ParsedHand(
            hand_id=hand_id,
            site=Site.POKERSTARS,
            timestamp=timestamp,
            game_type="No Limit Hold'em",
            stakes=(small_blind, big_blind),
            table_name=table_name,
            max_players=max_players,
            button_seat=button_seat
        )
        
        # Parse players and actions
        self._parse_players_and_actions(hand, lines[2:])
        
        return hand
    
    def _parse_ggpoker(self, hand_text: str) -> ParsedHand:
        """Parse a GGPoker hand"""
        lines = hand_text.strip().split('\n')
        
        # Parse header
        header_match = re.match(
            r"Poker Hand #(\d+): Hold'em No Limit \(\$([0-9.]+)/\$([0-9.]+)\) - (.+)",
            lines[0]
        )
        if not header_match:
            raise ValueError("Invalid GGPoker header")
        
        hand_id = header_match.group(1)
        small_blind = float(header_match.group(2))
        big_blind = float(header_match.group(3))
        timestamp_str = header_match.group(4)
        timestamp = self._parse_timestamp(timestamp_str)
        
        # Parse table info
        table_match = re.match(r"Table '(.+)' (\d+)-max Seat #(\d+) is the button", lines[1])
        if not table_match:
            raise ValueError("Invalid table info")
        
        table_name = table_match.group(1)
        max_players = int(table_match.group(2))
        button_seat = int(table_match.group(3))
        
        # Create hand object
        hand = ParsedHand(
            hand_id=hand_id,
            site=Site.GGPOKER,
            timestamp=timestamp,
            game_type="No Limit Hold'em",
            stakes=(small_blind, big_blind),
            table_name=table_name,
            max_players=max_players,
            button_seat=button_seat
        )
        
        # Parse players and actions
        self._parse_players_and_actions(hand, lines[2:])
        
        return hand
    
    def _parse_players_and_actions(self, hand: ParsedHand, lines: List[str]):
        """Parse players and all actions from hand text"""
        current_street = Street.PREFLOP
        i = 0
        
        # Parse player seats first
        while i < len(lines):
            line = lines[i].strip()
            
            # Player seat line
            seat_match = re.match(r"Seat (\d+): (.+?) \(\$([0-9.]+) in chips\)", line)
            if seat_match:
                seat = int(seat_match.group(1))
                name = seat_match.group(2)
                stack = float(seat_match.group(3))
                
                player = Player(
                    name=name,
                    seat=seat,
                    starting_stack=stack,
                    is_hero=(name == "Hero")
                )
                
                if player.is_hero:
                    hand.hero_name = name
                
                hand.players.append(player)
            
            # Blinds
            elif "posts small blind" in line:
                action_match = re.match(r"(.+?): posts small blind \$([0-9.]+)", line)
                if action_match:
                    player_name = action_match.group(1)
                    amount = float(action_match.group(2))
                    action = PlayerAction(player_name, "small_blind", amount, Street.PREFLOP)
                    hand.all_actions.append(action)
            
            elif "posts big blind" in line:
                action_match = re.match(r"(.+?): posts big blind \$([0-9.]+)", line)
                if action_match:
                    player_name = action_match.group(1)
                    amount = float(action_match.group(2))
                    action = PlayerAction(player_name, "big_blind", amount, Street.PREFLOP)
                    hand.all_actions.append(action)
            
            # Hole cards
            elif "*** HOLE CARDS ***" in line:
                current_street = Street.PREFLOP
            elif line.startswith("Dealt to"):
                cards_match = re.match(r"Dealt to (.+?) \[(.+)\]", line)
                if cards_match:
                    player_name = cards_match.group(1)
                    cards = cards_match.group(2).split()
                    for player in hand.players:
                        if player.name == player_name:
                            player.hole_cards = cards
                            break
            
            # Street transitions
            elif "*** FLOP ***" in line:
                current_street = Street.FLOP
                flop_match = re.search(r"\[(.+?)\]", line)
                if flop_match:
                    hand.flop = flop_match.group(1).split()
            
            elif "*** TURN ***" in line:
                current_street = Street.TURN
                turn_match = re.search(r"\[.+?\] \[(.+?)\]", line)
                if turn_match:
                    hand.turn = turn_match.group(1)
            
            elif "*** RIVER ***" in line:
                current_street = Street.RIVER
                river_match = re.search(r"\[.+?\] \[(.+?)\]", line)
                if river_match:
                    hand.river = river_match.group(1)
            
            # Actions
            elif ":" in line and any(action in line for action in ["folds", "checks", "calls", "raises", "bets"]):
                action = self._parse_action_line(line, current_street)
                if action:
                    hand.all_actions.append(action)
                    # Add to player's actions
                    for player in hand.players:
                        if player.name == action.player_name:
                            player.actions.append(action)
                            break
            
            # Summary
            elif "Total pot" in line:
                pot_match = re.search(r"Total pot \$([0-9.]+)", line)
                rake_match = re.search(r"Rake \$([0-9.]+)", line)
                if pot_match:
                    hand.total_pot = float(pot_match.group(1))
                if rake_match:
                    hand.rake = float(rake_match.group(1))
            
            i += 1
    
    def _parse_action_line(self, line: str, street: Street) -> Optional[PlayerAction]:
        """Parse a single action line"""
        line = line.strip()
        
        # Fold
        if " folds" in line:
            match = re.match(r"(.+?): folds", line)
            if match:
                return PlayerAction(match.group(1), "fold", 0.0, street)
        
        # Check
        elif " checks" in line:
            match = re.match(r"(.+?): checks", line)
            if match:
                return PlayerAction(match.group(1), "check", 0.0, street)
        
        # Call
        elif " calls" in line:
            match = re.match(r"(.+?): calls \$([0-9.]+)", line)
            if match:
                player = match.group(1)
                amount = float(match.group(2))
                is_all_in = "and is all-in" in line
                return PlayerAction(player, "call", amount, street, is_all_in)
        
        # Bet
        elif " bets" in line:
            match = re.match(r"(.+?): bets \$([0-9.]+)", line)
            if match:
                player = match.group(1)
                amount = float(match.group(2))
                is_all_in = "and is all-in" in line
                return PlayerAction(player, "bet", amount, street, is_all_in)
        
        # Raise
        elif " raises" in line:
            match = re.match(r"(.+?): raises \$([0-9.]+) to \$([0-9.]+)", line)
            if match:
                player = match.group(1)
                raise_amount = float(match.group(2))
                total_amount = float(match.group(3))
                is_all_in = "and is all-in" in line
                return PlayerAction(player, "raise", total_amount, street, is_all_in)
        
        return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from hand history"""
        try:
            # Remove timezone for simplicity
            timestamp_str = timestamp_str.replace(" ET", "").replace(" EST", "").replace(" UTC", "")
            return datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            # Return current time if parsing fails
            return datetime.now()
    
    def get_training_data(self, parsed_hands: List[ParsedHand]) -> List[Dict[str, Any]]:
        """Convert parsed hands to training data format"""
        training_data = []
        
        for hand in parsed_hands:
            if not hand.hero_name:
                continue  # Skip hands without hero
            
            # Find hero player
            hero = None
            for player in hand.players:
                if player.is_hero:
                    hero = player
                    break
            
            if not hero:
                continue
            
            # Create training examples for each hero action
            for action in hero.actions:
                example = {
                    "hand_id": hand.hand_id,
                    "site": hand.site.value,
                    "stakes": hand.stakes,
                    "position": self._get_position(hero.seat, hand.button_seat, len(hand.players)),
                    "hole_cards": hero.hole_cards,
                    "board_cards": {
                        "flop": hand.flop,
                        "turn": hand.turn,
                        "river": hand.river
                    },
                    "action": {
                        "type": action.action_type,
                        "amount": action.amount,
                        "street": action.street.value,
                        "is_all_in": action.is_all_in
                    },
                    "pot_size": hand.total_pot,
                    "stack_size": hero.starting_stack,
                    "num_players": len(hand.players)
                }
                training_data.append(example)
        
        return training_data
    
    def _get_position(self, seat: int, button_seat: int, num_players: int) -> str:
        """Calculate position relative to button"""
        seats_from_button = (seat - button_seat) % num_players
        
        if num_players == 2:
            return "SB" if seats_from_button == 0 else "BB"
        elif num_players <= 6:
            positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
            return positions[seats_from_button] if seats_from_button < len(positions) else "MP"
        else:
            # 9-max positions
            positions = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP1", "MP2", "CO", "HJ"]
            return positions[seats_from_button] if seats_from_button < len(positions) else "MP"


# Convenience functions
def parse_hand_history_file(file_path: str) -> List[ParsedHand]:
    """Parse a hand history file"""
    parser = HandHistoryParser()
    return parser.parse_file(file_path)


def get_training_data_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Get training data from a hand history file"""
    parser = HandHistoryParser()
    hands = parser.parse_file(file_path)
    return parser.get_training_data(hands)


# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Test with sample files
    sample_files = [
        "data/sample_hands/ggpoker_sample.txt",
        "data/sample_hands/pokerstars_sample.txt"
    ]
    
    parser = HandHistoryParser()
    
    for file_path in sample_files:
        print(f"\n=== Testing {file_path} ===")
        try:
            # Parse hands
            hands = parser.parse_file(file_path)
            print(f"Parsed {len(hands)} hands")
            
            # Show first hand details
            if hands:
                hand = hands[0]
                print(f"Hand ID: {hand.hand_id}")
                print(f"Site: {hand.site.value}")
                print(f"Stakes: ${hand.stakes[0]}/${hand.stakes[1]}")
                print(f"Players: {len(hand.players)}")
                print(f"Hero: {hand.hero_name}")
                print(f"Total pot: ${hand.total_pot}")
                print(f"Actions: {len(hand.all_actions)}")
                
                # Show hero actions
                if hand.hero_name:
                    hero_actions = [a for a in hand.all_actions if a.player_name == hand.hero_name]
                    print(f"Hero actions: {[f'{a.action_type}({a.amount})' for a in hero_actions]}")
            
            # Generate training data
            training_data = parser.get_training_data(hands)
            print(f"Generated {len(training_data)} training examples")
            
            if training_data:
                print("Sample training example:")
                print(json.dumps(training_data[0], indent=2, default=str))
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nHand history parser test completed!")