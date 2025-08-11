#!/usr/bin/env python3
"""
Test Trained Models

Quick test script to verify our trained models can be loaded and make predictions.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_core.neural_networks.action_predictor import PokerAI

def test_model(model_path: str, model_name: str = "Model"):
    """Test a single trained model (helper, not a pytest test)."""
    print(f"\n--- Testing {model_name} ---")
    try:
        ai = PokerAI(model_path)
        print(f"✓ Model loaded successfully")
        dummy_features = np.random.random(110)
        decision = ai.make_decision(dummy_features)
        print(f"✓ Basic prediction: {decision['action']} (confidence: {decision['confidence']:.3f})")
        legal_actions = ['fold', 'call', 'raise']
        constrained_decision = ai.make_decision(dummy_features, legal_actions=legal_actions)
        print(f"✓ Constrained prediction: {constrained_decision['action']} (confidence: {constrained_decision['confidence']:.3f})")
        analysis = ai.analyze_decision(dummy_features)
        top_action = analysis['top_3_actions'][0]
        print(f"✓ Analysis: Top action is {top_action['action']} with {top_action['probability']:.3f} probability")
        return True
    except Exception as e:
        print(f"✗ Error testing {model_name}: {e}")
        return False

def main():
    """Test all trained baseline models"""
    print("🧪 Testing Trained Baseline Models")
    print("=" * 50)
    
    models_dir = "models/baseline"
    models_to_test = [
        ("small_model.pth", "Small Model"),
        ("medium_model.pth", "Medium Model"), 
        ("large_model.pth", "Large Model")
    ]
    
    successful_tests = 0
    total_tests = len(models_to_test)
    
    for model_file, model_name in models_to_test:
        model_path = os.path.join(models_dir, model_file)
        
        if os.path.exists(model_path):
            if test_model(model_path, model_name):
                successful_tests += 1
        else:
            print(f"\n--- Testing {model_name} ---")
            print(f"✗ Model file not found: {model_path}")
    
    print(f"\n{'=' * 50}")
    print(f"🏁 Testing Complete: {successful_tests}/{total_tests} models passed")
    
    if successful_tests == total_tests:
        print("🎉 All models are working correctly!")
    else:
        print("⚠️  Some models failed testing")

if __name__ == "__main__":
    main()