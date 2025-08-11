"""
Neural Network for Poker Action Prediction

A simple neural network that learns to predict poker actions from game features.
Starts with basic action classification, can be extended for bet sizing.

Architecture:
- Input: 110-dimensional feature vector
- Hidden layers: Fully connected with ReLU activation
- Output: 6 action types (fold, check, call, bet, raise, all-in)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from training.data_processor import DataProcessor, TrainingExample
except ImportError:
    print("Warning: Could not import data processor")


class ActionPredictorNetwork(nn.Module):
    """Neural network for predicting poker actions"""
    
    def __init__(self, input_dim: int = 110, hidden_dims: List[int] = [256, 128, 64], num_actions: int = 6):
        super(ActionPredictorNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer for action probabilities
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Optional: separate head for bet sizing (when action is bet/raise)
        self.bet_size_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Ensure positive bet sizes
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        # Extract features through main network (except final layer)
        x = features
        for layer in self.network[:-1]:
            x = layer(x)
        
        # Action logits
        action_logits = self.network[-1](x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Bet size prediction (only used when action is bet/raise)
        bet_size = self.bet_size_head(x)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'bet_size': bet_size
        }
    
    def predict_action(self, features: torch.Tensor, legal_actions: Optional[List[int]] = None) -> Tuple[int, float]:
        """Predict the best action given features"""
        self.eval()
        with torch.no_grad():
            # Accept numpy arrays or lists
            if not isinstance(features, torch.Tensor):
                features = torch.as_tensor(features, dtype=torch.float32)
            if features.dim() == 1:
                features = features.unsqueeze(0)

            outputs = self.forward(features)
            action_probs = outputs['action_probs']
            
            # Handle batch dimension - squeeze if single example
            if action_probs.dim() > 1:
                action_probs = action_probs.squeeze(0)
            
            # Mask illegal actions if provided
            if legal_actions is not None:
                mask = torch.zeros_like(action_probs)
                for action in legal_actions:
                    if action < len(mask):  # Ensure action index is valid
                        mask[action] = 1.0
                action_probs = action_probs * mask
                # Avoid division by zero
                if action_probs.sum() > 0:
                    action_probs = action_probs / action_probs.sum()  # Renormalize
                else:
                    # If all actions are masked, use uniform distribution over legal actions
                    action_probs = mask / mask.sum()
            
            # Get best action
            best_action = torch.argmax(action_probs).item()
            confidence = action_probs[best_action].item()
            
            # Get bet size if action is bet/raise
            bet_size = outputs['bet_size'].squeeze().item() if outputs['bet_size'].numel() > 0 else 0.0
            
            return best_action, bet_size, confidence


class ActionPredictorTrainer:
    """Trainer for the action predictor network"""
    
    def __init__(self, model: ActionPredictorNetwork, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.action_criterion = nn.CrossEntropyLoss()
        self.bet_criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        
        # Action mapping
        self.action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
    
    def train_step(self, features: torch.Tensor, action_targets: torch.Tensor, amount_targets: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(features)
        action_logits = outputs['action_logits']
        bet_sizes = outputs['bet_size'].squeeze()
        
        # Action loss
        action_loss = self.action_criterion(action_logits, action_targets)
        
        # Bet size loss (only for bet/raise actions)
        bet_mask = (action_targets >= 3) & (action_targets <= 4)  # bet and raise
        if bet_mask.sum() > 0:
            bet_size_loss = self.bet_criterion(bet_sizes[bet_mask], amount_targets[bet_mask])
        else:
            bet_size_loss = torch.tensor(0.0)
        
        # Combined loss
        total_loss = action_loss + 0.1 * bet_size_loss  # Weight bet size loss lower
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        predicted_actions = torch.argmax(action_logits, dim=-1)
        accuracy = (predicted_actions == action_targets).float().mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'bet_size_loss': bet_size_loss.item() if isinstance(bet_size_loss, torch.Tensor) else 0.0,
            'accuracy': accuracy
        }
    
    def train_epoch(self, dataset: Dict[str, np.ndarray], batch_size: int = 32) -> Dict[str, float]:
        """Train for one epoch"""
        features = torch.FloatTensor(dataset['features'])
        action_targets = torch.LongTensor(dataset['action_targets'])
        amount_targets = torch.FloatTensor(dataset['amount_targets'])
        
        # Shuffle data
        indices = torch.randperm(len(features))
        features = features[indices]
        action_targets = action_targets[indices]
        amount_targets = amount_targets[indices]
        
        epoch_losses = []
        epoch_accuracies = []
        
        # Training batches
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_actions = action_targets[i:i+batch_size]
            batch_amounts = amount_targets[i:i+batch_size]
            
            metrics = self.train_step(batch_features, batch_actions, batch_amounts)
            epoch_losses.append(metrics['total_loss'])
            epoch_accuracies.append(metrics['accuracy'])
        
        # Average metrics
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def evaluate(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model on dataset"""
        self.model.eval()
        
        features = torch.FloatTensor(dataset['features'])
        action_targets = torch.LongTensor(dataset['action_targets'])
        amount_targets = torch.FloatTensor(dataset['amount_targets'])
        
        with torch.no_grad():
            outputs = self.model(features)
            action_logits = outputs['action_logits']
            bet_sizes = outputs['bet_size'].squeeze()
            
            # Action loss and accuracy
            action_loss = self.action_criterion(action_logits, action_targets)
            predicted_actions = torch.argmax(action_logits, dim=-1)
            accuracy = (predicted_actions == action_targets).float().mean().item()
            
            # Bet size loss
            bet_mask = (action_targets >= 3) & (action_targets <= 4)
            if bet_mask.sum() > 0:
                bet_size_loss = self.bet_criterion(bet_sizes[bet_mask], amount_targets[bet_mask])
                bet_size_mae = torch.abs(bet_sizes[bet_mask] - amount_targets[bet_mask]).mean().item()
            else:
                bet_size_loss = torch.tensor(0.0)
                bet_size_mae = 0.0
            
            # Per-action accuracy
            action_accuracies = {}
            for action_id in range(6):
                mask = action_targets == action_id
                if mask.sum() > 0:
                    action_acc = (predicted_actions[mask] == action_targets[mask]).float().mean().item()
                    action_accuracies[self.action_names[action_id]] = action_acc
        
        return {
            'loss': action_loss.item(),
            'accuracy': accuracy,
            'bet_size_loss': bet_size_loss.item() if isinstance(bet_size_loss, torch.Tensor) else 0.0,
            'bet_size_mae': bet_size_mae,
            'action_accuracies': action_accuracies
        }
    
    def train(self, dataset: Dict[str, np.ndarray], epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Full training loop"""
        
        # Split data
        n_samples = len(dataset['features'])
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_dataset = {
            'features': dataset['features'][train_indices],
            'action_targets': dataset['action_targets'][train_indices],
            'amount_targets': dataset['amount_targets'][train_indices]
        }
        
        val_dataset = {
            'features': dataset['features'][val_indices],
            'action_targets': dataset['action_targets'][val_indices],
            'amount_targets': dataset['amount_targets'][val_indices]
        }
        
        print(f"Training on {len(train_dataset['features'])} examples")
        print(f"Validating on {len(val_dataset['features'])} examples")
        
        # Training loop
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_dataset, batch_size)
            
            # Validate
            val_metrics = self.evaluate(val_dataset)
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def save_model(self, file_path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies
        }, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """Load model from file"""
        checkpoint = torch.load(file_path, weights_only=False)  # PyTorch 2.6 compatibility
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        print(f"Model loaded from {file_path}")


class PokerAI:
    """High-level poker AI that uses the trained model to make decisions"""
    
    def __init__(self, model_path: Optional[str] = None, model_config: Optional[Dict] = None):
        # If loading from file, try to determine architecture from the checkpoint
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Try to infer model architecture from the saved state dict
            model_state = checkpoint.get('model_state_dict', {})
            if model_state:
                hidden_dims = self._infer_hidden_dims(model_state)
                input_dim = self._infer_input_dim(model_state)
                num_actions = self._infer_num_actions(model_state)
                
                self.model = ActionPredictorNetwork(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    num_actions=num_actions
                )
            else:
                self.model = ActionPredictorNetwork()
        elif model_config:
            # Create model with specified config
            self.model = ActionPredictorNetwork(**model_config)
        else:
            # Default model
            self.model = ActionPredictorNetwork()
        
        self.trainer = ActionPredictorTrainer(self.model)
        
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
        
        # Action mapping
        self.action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
        self.id_to_action = {i: name for i, name in enumerate(self.action_names)}
    
    def _infer_hidden_dims(self, model_state: Dict) -> List[int]:
        """Infer hidden dimensions from model state dict"""
        hidden_dims = []
        
        # Look for network layers (excluding final output layer)
        layer_idx = 0
        while f'network.{layer_idx}.weight' in model_state:
            if f'network.{layer_idx + 3}.weight' in model_state:  # Not the final layer
                hidden_dims.append(model_state[f'network.{layer_idx}.weight'].shape[0])
            layer_idx += 3  # Skip ReLU and Dropout layers
        
        return hidden_dims if hidden_dims else [256, 128, 64]  # Default
    
    def _infer_input_dim(self, model_state: Dict) -> int:
        """Infer input dimension from model state dict"""
        if 'network.0.weight' in model_state:
            return model_state['network.0.weight'].shape[1]
        return 110  # Default
    
    def _infer_num_actions(self, model_state: Dict) -> int:
        """Infer number of actions from model state dict"""
        # Find the final layer by looking for the highest numbered layer
        max_layer = 0
        for key in model_state.keys():
            if key.startswith('network.') and '.weight' in key:
                layer_num = int(key.split('.')[1])
                max_layer = max(max_layer, layer_num)
        
        final_layer_key = f'network.{max_layer}.weight'
        if final_layer_key in model_state:
            return model_state[final_layer_key].shape[0]
        return 6  # Default
    
    def make_decision(self, features: np.ndarray, legal_actions: Optional[List[str]] = None) -> Dict[str, any]:
        """Make a poker decision given game features"""
        
        # Convert to tensor
        features_tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Convert legal actions to indices
        legal_action_ids = None
        if legal_actions:
            legal_action_ids = [self.action_names.index(action) for action in legal_actions if action in self.action_names]
        
        # Get prediction
        action_id, bet_size, confidence = self.model.predict_action(features_tensor, legal_action_ids)
        action_name = self.id_to_action[action_id]
        
        return {
            'action': action_name,
            'amount': bet_size if action_name in ['bet', 'raise'] else 0.0,
            'confidence': confidence,
            'action_id': action_id
        }
    
    def train_from_dataset(self, dataset: Dict[str, np.ndarray], epochs: int = 50):
        """Train the model from a dataset"""
        return self.trainer.train(dataset, epochs=epochs)
    
    def save(self, file_path: str):
        """Save the AI model"""
        self.trainer.save_model(file_path)
    
    def analyze_decision(self, features: np.ndarray) -> Dict[str, float]:
        """Get detailed analysis of a decision"""
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            action_probs = outputs['action_probs'].squeeze()
            bet_size = outputs['bet_size'].item()
        
        # Create detailed analysis
        analysis = {
            'action_probabilities': {
                self.action_names[i]: prob.item() 
                for i, prob in enumerate(action_probs)
            },
            'recommended_bet_size': bet_size,
            'top_3_actions': []
        }
        
        # Get top 3 actions
        top_actions = torch.topk(action_probs, 3)
        for i, (prob, action_id) in enumerate(zip(top_actions.values, top_actions.indices)):
            analysis['top_3_actions'].append({
                'rank': i + 1,
                'action': self.action_names[action_id],
                'probability': prob.item()
            })
        
        return analysis


# Example usage and testing
if __name__ == "__main__":
    print("Testing Poker AI Neural Network")
    print("=" * 50)
    
    # Load training data
    try:
        processor = DataProcessor()
        sample_files = [
            "data/sample_hands/ggpoker_sample.txt",
            "data/sample_hands/pokerstars_sample.txt"
        ]
        
        dataset = processor.create_training_dataset(sample_files)
        
        if dataset and len(dataset['features']) > 0:
            print(f"Loaded dataset with {len(dataset['features'])} examples")
            
            # Create and train AI
            ai = PokerAI()
            
            # Train (with very few epochs for demo)
            print("\nTraining AI...")
            history = ai.train_from_dataset(dataset, epochs=20)
            
            # Test predictions
            print("\nTesting predictions...")
            test_features = dataset['features'][0]  # First example
            
            # Make decision
            decision = ai.make_decision(test_features)
            print(f"AI Decision: {decision}")
            
            # Analyze decision
            analysis = ai.analyze_decision(test_features)
            print(f"\nDetailed Analysis:")
            print(f"Action Probabilities: {analysis['action_probabilities']}")
            print(f"Top 3 Actions: {analysis['top_3_actions']}")
            
            # Save model
            ai.save("poker_ai_model.pth")
            
            print("\nAI training and testing completed!")
        else:
            print("No training data available")
            
    except Exception as e:
        print(f"Error during AI testing: {e}")
        
        # Test with dummy data
        print("\nTesting with dummy data...")
        dummy_features = np.random.random(110)
        
        ai = PokerAI()
        decision = ai.make_decision(dummy_features, legal_actions=['fold', 'call', 'raise'])
        print(f"Dummy decision: {decision}")
        
        analysis = ai.analyze_decision(dummy_features)
        print(f"Dummy analysis: {analysis['top_3_actions']}")
    
    print("\nNeural network test completed!")