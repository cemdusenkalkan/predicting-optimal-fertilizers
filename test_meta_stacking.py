#!/usr/bin/env python3
"""
Test script for meta_stacking module
"""

import sys
import os
sys.path.append('.')

from src.meta_stacking import MetaStackingEnsemble, DataAugmentation
from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader
from src.config import config

def test_meta_stacking():
    """Test meta-stacking functionality"""
    print("📈 Testing MetaStackingEnsemble...")
    
    try:
        # Load sample data
        loader = DataLoader(config)
        data = loader.load_all()
        
        # Prepare features (small sample for testing)
        fe = FeatureEngineer(config)
        X_train = fe.create_base_features(data['X_train'].head(1000))
        y_train = data['y_train'].head(1000)
        
        # Test data augmentation
        print("Testing data augmentation...")
        augmenter = DataAugmentation(expansion_factor=2)
        X_aug, y_aug = augmenter.expand_training_data(X_train.head(100), y_train.head(100))
        
        print(f"✅ Data augmentation completed")
        print(f"   Original: {len(X_train)} -> Augmented: {len(X_aug)}")
        
        # Test meta-stacking (with reduced folds for speed)
        print("Testing meta-stacking ensemble...")
        ensemble = MetaStackingEnsemble(config, n_folds=3)
        
        # Use small sample for quick test
        X_small = X_train.head(300)
        y_small = y_train.head(300)
        
        ensemble.fit(X_small, y_small)
        
        print(f"✅ Meta-stacking training completed")
        
        # Test prediction
        predictions = ensemble.predict_proba(X_train.head(50))
        print(f"✅ Predictions generated: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MetaStacking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_meta_stacking()
    sys.exit(0 if success else 1) 