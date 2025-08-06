"""
Simple Action Predictor using Scikit-Learn

A lightweight poker action predictor using scikit-learn instead of PyTorch.
Good for getting started and testing the pipeline without heavy dependencies.

Models:
- Random Forest for action classification
- Linear Regression for bet sizing
"""

import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from training.data_processor import DataProcessor, TrainingExample
except ImportError:
    print("Warning: Could not import data processor")


class SimpleActionPredictor:
    """Simple poker action predictor using scikit-learn"""
    
    def __init__(self):
        # Action classifier
        self.action_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Bet size regressor
        self.bet_regressor = LinearRegression()
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Action mapping
        self.action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
        self.action_to_id = {name: i for i, name in enumerate(self.action_names)}
        self.id_to_action = {i: name for i, name in enumerate(self.action_names)}
        
        # Training history
        self.is_trained = False
        self.training_stats = {}
        
    def prepare_data(self, dataset: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        features = dataset['features']
        action_targets = dataset['action_targets']
        amount_targets = dataset['amount_targets']
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, action_targets, amount_targets
    
    def train(self, dataset: Dict[str, np.ndarray], test_size: float = 0.2) -> Dict[str, Any]:
        """Train the action predictor"""
        print("Training Simple Action Predictor...")
        
        # Prepare data
        features, action_targets, amount_targets = self.prepare_data(dataset)
        
        # Split data
        X_train, X_test, y_action_train, y_action_test, y_amount_train, y_amount_test = train_test_split(
            features, action_targets, amount_targets, test_size=test_size, random_state=42
        )
        
        print(f"Training on {len(X_train)} examples, testing on {len(X_test)} examples")
        
        # Train action classifier
        print("Training action classifier...")
        self.action_classifier.fit(X_train, y_action_train)
        
        # Predict actions on training set to get bet/raise examples
        y_action_pred_train = self.action_classifier.predict(X_train)
        
        # Train bet regressor (only on bet/raise actions)
        bet_mask_train = (y_action_train >= 3) & (y_action_train <= 4)  # bet and raise
        if bet_mask_train.sum() > 0:
            print("Training bet size regressor...")
            self.bet_regressor.fit(X_train[bet_mask_train], y_amount_train[bet_mask_train])
        else:
            print("No bet/raise examples found, skipping bet regressor training")
        
        # Evaluate models
        print("Evaluating models...")
        
        # Action classification metrics
        y_action_pred = self.action_classifier.predict(X_test)
        action_accuracy = accuracy_score(y_action_test, y_action_pred)
        
        # Bet size metrics
        bet_mae = 0.0
        bet_mask_test = (y_action_test >= 3) & (y_action_test <= 4)
        if bet_mask_test.sum() > 0 and hasattr(self.bet_regressor, 'coef_'):
            y_amount_pred = self.bet_regressor.predict(X_test[bet_mask_test])
            bet_mae = mean_absolute_error(y_amount_test[bet_mask_test], y_amount_pred)
        
        # Feature importance
        feature_importance = self.action_classifier.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]  # Top 10 features
        
        # Training statistics
        self.training_stats = {
            'action_accuracy': action_accuracy,
            'bet_mae': bet_mae,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'action_distribution': np.bincount(y_action_train),
            'top_features': top_features.tolist(),
            'feature_importance': feature_importance.tolist()
        }
        
        self.is_trained = True
        
        print(f"Training completed!")
        print(f"Action accuracy: {action_accuracy:.3f}")
        print(f"Bet size MAE: ${bet_mae:.2f}")
        
        # Detailed classification report
        print("\nDetailed Action Classification Report:")
        print(classification_report(y_action_test, y_action_pred, target_names=self.action_names))
        
        return self.training_stats
    
    def predict_action(self, features: np.ndarray, legal_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict the best action given features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get action probabilities
        action_probs = self.action_classifier.predict_proba(features_scaled)[0]
        
        # Apply legal action mask if provided
        if legal_actions:
            legal_mask = np.zeros(len(self.action_names))
            for action in legal_actions:
                if action in self.action_to_id:
                    legal_mask[self.action_to_id[action]] = 1.0
            
            # Zero out illegal actions
            action_probs = action_probs * legal_mask
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()  # Renormalize
        
        # Get best action
        best_action_id = np.argmax(action_probs)
        best_action = self.action_names[best_action_id]
        confidence = action_probs[best_action_id]
        
        # Predict bet size if action is bet/raise
        bet_size = 0.0
        if best_action in ['bet', 'raise'] and hasattr(self.bet_regressor, 'coef_'):
            bet_size = max(0.0, self.bet_regressor.predict(features_scaled)[0])
        
        return {
            'action': best_action,
            'action_id': best_action_id,
            'amount': bet_size,
            'confidence': confidence,
            'action_probabilities': {
                self.action_names[i]: prob for i, prob in enumerate(action_probs)
            }
        }
    
    def analyze_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """Get detailed analysis of a decision"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analysis")
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        action_probs = self.action_classifier.predict_proba(features_scaled)[0]
        bet_size = 0.0
        if hasattr(self.bet_regressor, 'coef_'):
            bet_size = max(0.0, self.bet_regressor.predict(features_scaled)[0])
        
        # Get top 3 actions
        top_3_indices = np.argsort(action_probs)[-3:][::-1]
        top_3_actions = [
            {
                'rank': i + 1,
                'action': self.action_names[idx],
                'probability': action_probs[idx]
            }
            for i, idx in enumerate(top_3_indices)
        ]
        
        # Feature importance analysis
        feature_contributions = features_scaled[0] * self.action_classifier.feature_importances_
        top_feature_indices = np.argsort(np.abs(feature_contributions))[-5:][::-1]
        
        return {
            'action_probabilities': {
                self.action_names[i]: prob for i, prob in enumerate(action_probs)
            },
            'recommended_bet_size': bet_size,
            'top_3_actions': top_3_actions,
            'top_contributing_features': [
                {
                    'feature_index': int(idx),
                    'contribution': float(feature_contributions[idx]),
                    'importance': float(self.action_classifier.feature_importances_[idx])
                }
                for idx in top_feature_indices
            ]
        }
    
    def save_model(self, file_path: str):
        """Save model to file"""
        model_data = {
            'action_classifier': self.action_classifier,
            'bet_regressor': self.bet_regressor,
            'scaler': self.scaler,
            'action_names': self.action_names,
            'action_to_id': self.action_to_id,
            'id_to_action': self.id_to_action,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """Load model from file"""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.action_classifier = model_data['action_classifier']
        self.bet_regressor = model_data['bet_regressor']
        self.scaler = model_data['scaler']
        self.action_names = model_data['action_names']
        self.action_to_id = model_data['action_to_id']
        self.id_to_action = model_data['id_to_action']
        self.is_trained = model_data['is_trained']
        self.training_stats = model_data['training_stats']
        
        print(f"Model loaded from {file_path}")


class SimplePokerAI:
    """Simple poker AI using the lightweight action predictor"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.predictor = SimpleActionPredictor()
        
        if model_path and os.path.exists(model_path):
            self.predictor.load_model(model_path)
    
    def make_decision(self, features: np.ndarray, legal_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Make a poker decision given game features"""
        return self.predictor.predict_action(features, legal_actions)
    
    def analyze_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """Get detailed analysis of a decision"""
        return self.predictor.analyze_decision(features)
    
    def train_from_dataset(self, dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train the AI from a dataset"""
        return self.predictor.train(dataset)
    
    def save(self, file_path: str):
        """Save the AI model"""
        self.predictor.save_model(file_path)
    
    def is_trained(self) -> bool:
        """Check if the AI is trained"""
        return self.predictor.is_trained
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.predictor.training_stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing Simple Poker AI")
    print("=" * 50)
    
    # Test with training data
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
            ai = SimplePokerAI()
            
            # Train
            print("\nTraining AI...")
            training_stats = ai.train_from_dataset(dataset)
            
            # Test predictions
            print("\nTesting predictions...")
            test_features = dataset['features'][0]  # First example
            
            # Make decision
            decision = ai.make_decision(test_features)
            print(f"AI Decision: {decision['action']} (${decision['amount']:.2f})")
            print(f"Confidence: {decision['confidence']:.3f}")
            
            # Test with legal actions constraint
            legal_decision = ai.make_decision(test_features, legal_actions=['fold', 'call', 'raise'])
            print(f"Legal Action Decision: {legal_decision['action']} (${legal_decision['amount']:.2f})")
            
            # Analyze decision
            analysis = ai.analyze_decision(test_features)
            print(f"\nTop 3 Actions:")
            for action_info in analysis['top_3_actions']:
                print(f"  {action_info['rank']}. {action_info['action']}: {action_info['probability']:.3f}")
            
            # Save model
            ai.save("simple_poker_ai.pkl")
            
            # Test loading
            print("\nTesting model loading...")
            ai2 = SimplePokerAI("simple_poker_ai.pkl")
            decision2 = ai2.make_decision(test_features)
            print(f"Loaded AI Decision: {decision2['action']} (${decision2['amount']:.2f})")
            
            print("\nSimple AI training and testing completed!")
            
        else:
            print("No training data available")
            
    except Exception as e:
        print(f"Error during AI testing: {e}")
        import traceback
        traceback.print_exc()
        
        # Test with dummy data
        print("\nTesting with dummy data...")
        dummy_features = np.random.random(110)
        
        ai = SimplePokerAI()
        # Can't make decisions without training
        print("AI needs training before making decisions")
    
    print("\nSimple neural network test completed!")