#!/usr/bin/env python3
"""
Test script for data_loader module
"""

import sys
import os
sys.path.append('.')

from src.data_loader import DataLoader
from src.config import config

def test_data_loader():
    """Test data loading functionality"""
    print("🔄 Testing DataLoader...")
    
    try:
        # Initialize data loader
        loader = DataLoader(config)
        
        # Test loading competition data
        train, test = loader.load_competition_data()
        print(f"✅ Loaded train: {train.shape}, test: {test.shape}")
        
        # Test loading all data
        data = loader.load_all()
        print(f"✅ All data loaded successfully")
        print(f"   Train: {data['X_train'].shape}")
        print(f"   Test: {data['X_test'].shape}")
        print(f"   Target classes: {len(data['y_train'].unique())}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loader()
    sys.exit(0 if success else 1) 