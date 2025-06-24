#!/usr/bin/env python3
"""
Test script for feature_engineering module
"""

import sys
import os
sys.path.append('.')

from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader
from src.config import config

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("🔧 Testing FeatureEngineer...")
    
    try:
        # Load sample data
        loader = DataLoader(config)
        data = loader.load_all()
        
        # Initialize feature engineer
        fe = FeatureEngineer(config)
        
        # Test base feature creation
        print("Testing base feature creation...")
        X_train_features = fe.create_base_features(data['X_train'].head(1000))
        X_test_features = fe.create_base_features(data['X_test'].head(500))
        
        print(f"✅ Base features created")
        print(f"   Train features: {X_train_features.shape}")
        print(f"   Test features: {X_test_features.shape}")
        
        # Test target encoding
        print("Testing target encoding...")
        X_train_encoded, X_test_encoded, _ = fe.add_target_encoding(
            X_train_features, 
            data['y_train'].head(1000), 
            X_test_features, 
            None
        )
        
        print(f"✅ Target encoding completed")
        print(f"   Final train features: {X_train_encoded.shape}")
        print(f"   Final test features: {X_test_encoded.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FeatureEngineer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_engineering()
    sys.exit(0 if success else 1) 