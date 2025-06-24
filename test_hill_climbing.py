#!/usr/bin/env python3
"""
Test script for hill_climbing module
"""

import sys
import os
sys.path.append('.')

from src.hill_climbing import HillClimbingOptimizer
from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader
from src.config import config

def test_hill_climbing():
    """Test hill climbing functionality"""
    print("🧗 Testing HillClimbingOptimizer...")
    
    try:
        # Load sample data
        loader = DataLoader(config)
        data = loader.load_all()
        
        # Initialize optimizer
        optimizer = HillClimbingOptimizer(config)
        print(f"✅ HillClimbingOptimizer initialized")
        
        # Test feature generation
        print("Testing iterative feature generation...")
        fe = FeatureEngineer(config)
        X_base = fe.create_base_features(data['X_train'].head(200))
        
        # Test one iteration of feature generation
        X_enhanced = optimizer.generate_iteration_features(X_base, iteration=1)
        print(f"✅ Feature generation completed")
        print(f"   Base features: {X_base.shape[1]}")
        print(f"   Enhanced features: {X_enhanced.shape[1]}")
        
        # Test experiment caching
        print("Testing experiment caching...")
        optimizer.cache_experiment(X_base, X_base, X_base, 0.5, 1)
        print(f"✅ Experiment caching completed")
        
        return True
        
    except Exception as e:
        print(f"❌ HillClimbing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hill_climbing()
    sys.exit(0 if success else 1) 