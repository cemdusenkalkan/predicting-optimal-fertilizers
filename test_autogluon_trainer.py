#!/usr/bin/env python3
"""
Test script for autogluon_trainer module
"""

import sys
import os
sys.path.append('.')

def test_autogluon_trainer():
    """Test autogluon trainer functionality"""
    print("🚀 Testing AutoGluonTrainer...")
    
    try:
        from src.autogluon_trainer import AutoGluonTrainer
        from src.feature_engineering import FeatureEngineer
        from src.data_loader import DataLoader
        from src.config import config
        
        # Load sample data
        loader = DataLoader(config)
        data = loader.load_all()
        
        # Prepare features (small sample for testing)
        fe = FeatureEngineer(config)
        X_train = fe.create_base_features(data['X_train'].head(200))
        y_train = data['y_train'].head(200)
        X_test = fe.create_base_features(data['X_test'].head(100))
        
        # Initialize trainer
        trainer = AutoGluonTrainer(config)
        print(f"✅ AutoGluonTrainer initialized")
        
        # Test quick evaluation (without full training)
        print("Testing quick evaluation...")
        score = trainer.quick_evaluate(X_train, y_train, X_train.head(50), y_train.head(50))
        print(f"✅ Quick evaluation completed: {score:.4f}")
        
        return True
        
    except ImportError:
        print("⚠️  AutoGluon not installed, skipping test")
        return True
        
    except Exception as e:
        print(f"❌ AutoGluonTrainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_autogluon_trainer()
    sys.exit(0 if success else 1) 