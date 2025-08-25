#!/usr/bin/env python3
"""
Baseline Model Training Script

Train a comprehensive baseline poker AI model with proper evaluation,
model comparison, and performance analysis.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import Dict, List, Any
from datetime import datetime
import os

USE_MLFLOW = os.environ.get("USE_MLFLOW", "1") == "1"
if USE_MLFLOW:
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        MLFLOW_OK = True
    except Exception:
        MLFLOW_OK = False
else:
    MLFLOW_OK = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_core.neural_networks.action_predictor import PokerAI, ActionPredictorNetwork, ActionPredictorTrainer
from training.data_processor import DataProcessor

class BaselineModelTrainer:
    """Comprehensive baseline model training and evaluation"""
    
    def __init__(self, data_dir: str = "data/sample_hands"):
        self.data_dir = data_dir
        self.results_dir = "models/baseline"
        self.processor = DataProcessor()
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training configurations to test
        self.model_configs = {
            'small': {
                'hidden_dims': [128, 64],
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 8
            },
            'medium': {
                'hidden_dims': [256, 128, 64],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 16
            },
            'large': {
                'hidden_dims': [512, 256, 128, 64],
                'learning_rate': 0.0005,
                'epochs': 150,
                'batch_size': 32
            }
        }
        
        self.results = {}
    
    def load_training_data(self) -> Dict[str, np.ndarray]:
        """Load and prepare training data"""
        print("Loading training data...")
        
        # Find all sample files
        sample_files = []
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('.txt'):
                    sample_files.append(os.path.join(self.data_dir, file))
        
        if not sample_files:
            print(f"No training files found in {self.data_dir}")
            return {}
        
        print(f"Found {len(sample_files)} training files")
        
        # Create dataset
        dataset = self.processor.create_training_dataset(sample_files)
        
        if dataset:
            print(f"Dataset loaded: {len(dataset['features'])} examples")
            print(f"Feature dimension: {dataset['feature_dim']}")
            
            # Analyze dataset
            self.processor.analyze_dataset(dataset)
            
            return dataset
        else:
            print("Failed to create dataset")
            return {}
    
    def train_model(self, config_name: str, config: Dict, dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train a single model configuration"""
        print(f"\n{'='*60}")
        print(f"Training {config_name} model")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Create model
        model = ActionPredictorNetwork(
            input_dim=dataset['feature_dim'],
            hidden_dims=config['hidden_dims'],
            num_actions=dataset['num_actions']
        )
        
        # Compute class weights to mitigate imbalance (inverse frequency)
        action_targets_arr = dataset['action_targets']
        binc = np.bincount(action_targets_arr, minlength=dataset['num_actions'])
        inv = 1.0 / np.maximum(binc, 1)
        class_weights = inv * (len(binc) / inv.sum())

        # Create trainer with class weights
        trainer = ActionPredictorTrainer(model, learning_rate=config['learning_rate'], class_weights=class_weights)
        
        # Train model
        start_time = datetime.now()
        if MLFLOW_OK:
            mlflow.set_experiment("poker-baseline")
            mlflow.start_run(run_name=f"baseline-{config_name}")
            mlflow.log_params({
                'hidden_dims': config['hidden_dims'],
                'learning_rate': config['learning_rate'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
            })

        # Try stratified split by (street, position)
        try:
            splits = self.processor.stratified_split_indices(dataset['metadata'], validation_split=0.2)
            train_idx = splits['train']
            val_idx = splits['val']

            train_dataset = {
                'features': dataset['features'][train_idx],
                'action_targets': dataset['action_targets'][train_idx],
                'amount_targets': dataset['amount_targets'][train_idx],
            }
            if 'bet_bucket_targets' in dataset:
                train_dataset['bet_bucket_targets'] = dataset['bet_bucket_targets'][train_idx]

            val_dataset = {
                'features': dataset['features'][val_idx],
                'action_targets': dataset['action_targets'][val_idx],
                'amount_targets': dataset['amount_targets'][val_idx],
            }
            if 'bet_bucket_targets' in dataset:
                val_dataset['bet_bucket_targets'] = dataset['bet_bucket_targets'][val_idx]

            val_losses = []
            val_accuracies = []
            for epoch in range(config['epochs']):
                train_metrics = trainer.train_epoch(train_dataset, batch_size=config['batch_size'])
                val_metrics = trainer.evaluate(val_dataset)
                val_losses.append(val_metrics['loss'])
                val_accuracies.append(val_metrics['accuracy'])
                if epoch % 10 == 0 or epoch == config['epochs'] - 1:
                    print(f"Epoch {epoch:3d}: Train Loss {train_metrics['loss']:.4f}, Acc {train_metrics['accuracy']:.4f} | Val Loss {val_metrics['loss']:.4f}, Acc {val_metrics['accuracy']:.4f}")

            history = {
                'train_losses': trainer.train_losses,
                'train_accuracies': trainer.train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            }
        except Exception as e:
            # Fallback to original trainer's internal split
            history = trainer.train(
                dataset,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=0.2
            )
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds()
        
        # Evaluate final model
        val_split = int(len(dataset['features']) * 0.2)
        indices = np.random.permutation(len(dataset['features']))
        val_indices = indices[-val_split:]
        
        val_dataset = {
            'features': dataset['features'][val_indices],
            'action_targets': dataset['action_targets'][val_indices],
            'amount_targets': dataset['amount_targets'][val_indices]
        }
        
        final_metrics = trainer.evaluate(val_dataset)
        if MLFLOW_OK:
            mlflow.log_metrics({
                'val_loss': float(final_metrics['loss']),
                'val_accuracy': float(final_metrics['accuracy']),
                'val_bet_size_loss': float(final_metrics.get('bet_size_loss', 0.0)),
                'val_bucket_loss': float(final_metrics.get('bucket_loss', 0.0)),
                'val_bet_size_mae': float(final_metrics.get('bet_size_mae', 0.0)),
            })
        
        # Save model
        model_path = os.path.join(self.results_dir, f"{config_name}_model.pth")
        trainer.save_model(model_path)
        if MLFLOW_OK:
            mlflow.log_artifact(model_path)
        
        # Compile results
        results = {
            'config': config,
            'training_time_seconds': training_time,
            'final_metrics': final_metrics,
            'training_history': history,
            'model_path': model_path,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        }
        
        print(f"\n{config_name} Model Results:")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Final accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Final loss: {final_metrics['loss']:.4f}")
        print(f"  Model parameters: {results['num_parameters']:,}")
        print(f"  Model size: {results['model_size_mb']:.2f} MB")
        
        if MLFLOW_OK:
            mlflow.end_run()
        return results
    
    def run_comprehensive_evaluation(self, dataset: Dict[str, np.ndarray]):
        """Run comprehensive evaluation of all model configurations"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BASELINE MODEL EVALUATION")
        print("="*80)
        
        for config_name, config in self.model_configs.items():
            try:
                results = self.train_model(config_name, config, dataset)
                self.results[config_name] = results
            except Exception as e:
                print(f"Error training {config_name} model: {e}")
                self.results[config_name] = {'error': str(e)}
        
        # Generate comparison report
        self.generate_comparison_report()
        
        # Save all results
        results_file = os.path.join(self.results_dir, f"baseline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.results.items():
                serializable_results[key] = self._make_json_serializable(value)
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def generate_comparison_report(self):
        """Generate a detailed comparison report"""
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        
        # Extract key metrics
        comparison_data = []
        for config_name, results in self.results.items():
            if 'error' not in results:
                comparison_data.append({
                    'name': config_name,
                    'accuracy': results['final_metrics']['accuracy'],
                    'loss': results['final_metrics']['loss'],
                    'training_time': results['training_time_seconds'],
                    'parameters': results['num_parameters'],
                    'size_mb': results['model_size_mb']
                })
        
        if not comparison_data:
            print("No successful models to compare")
            return
        
        # Sort by accuracy (descending)
        comparison_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"{'Model':<10} {'Accuracy':<10} {'Loss':<10} {'Time(s)':<10} {'Params':<10} {'Size(MB)':<10}")
        print("-" * 70)
        
        for data in comparison_data:
            print(f"{data['name']:<10} {data['accuracy']:<10.4f} {data['loss']:<10.4f} "
                  f"{data['training_time']:<10.1f} {data['parameters']:<10,} {data['size_mb']:<10.2f}")
        
        # Best model analysis
        best_model = comparison_data[0]
        print(f"\n🏆 Best Model: {best_model['name']}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        print(f"   Loss: {best_model['loss']:.4f}")
        print(f"   Training Time: {best_model['training_time']:.1f}s")
        
        # Action-specific performance
        print(f"\n📊 Action-Specific Performance (Best Model):")
        best_results = self.results[best_model['name']]
        if 'action_accuracies' in best_results['final_metrics']:
            for action, acc in best_results['final_metrics']['action_accuracies'].items():
                print(f"   {action}: {acc:.4f}")
    
    def test_trained_models(self, dataset: Dict[str, np.ndarray]):
        """Test all trained models on sample decisions"""
        print("\n" + "="*80)
        print("TESTING TRAINED MODELS")
        print("="*80)
        
        # Use first few examples for testing
        test_features = dataset['features'][:3]
        test_actions = dataset['action_targets'][:3]
        
        for config_name, results in self.results.items():
            if 'error' in results:
                continue
                
            print(f"\n--- {config_name} Model Predictions ---")
            
            try:
                # Load the trained model
                ai = PokerAI(results['model_path'])
                
                for i, (features, true_action) in enumerate(zip(test_features, test_actions)):
                    decision = ai.make_decision(features)
                    analysis = ai.analyze_decision(features)
                    
                    true_action_name = self.processor.id_to_action.get(true_action, f"unknown_{true_action}")
                    correct = decision['action'] == true_action_name
                    
                    print(f"  Example {i+1}:")
                    print(f"    True action: {true_action_name}")
                    print(f"    Predicted: {decision['action']} (confidence: {decision['confidence']:.3f})")
                    print(f"    Correct: {'✓' if correct else '✗'}")
                    top_3_str = [f"{a['action']}({a['probability']:.3f})" for a in analysis['top_3_actions']]
                    print(f"    Top 3: {top_3_str}")
                    
            except Exception as e:
                print(f"Error testing {config_name} model: {e}")


def main():
    """Main training script"""
    print("🚀 Starting Comprehensive Baseline Model Training")
    print("="*80)
    
    trainer = BaselineModelTrainer()
    
    # Load data
    dataset = trainer.load_training_data()
    if not dataset:
        print("❌ No training data available. Exiting.")
        return
    
    # Run comprehensive evaluation
    trainer.run_comprehensive_evaluation(dataset)
    
    # Test all models
    trainer.test_trained_models(dataset)
    
    print("\n🎉 Baseline model training completed!")
    print(f"📁 Results saved in: {trainer.results_dir}")


if __name__ == "__main__":
    main()