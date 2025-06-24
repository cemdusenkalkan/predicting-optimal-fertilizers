#!/usr/bin/env python3
"""
Test script for ranking_models module
"""

import sys
import os
sys.path.append('.')

from src.ranking_models import RankingOptimizedModels, ProbabilityCalibration, SnapshotEnsemble
from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader
from src.config import config

def test_ranking_models():
    """Test ranking models functionality"""
    print("🎯 Testing RankingOptimizedModels...")
    
    try:
        # Load sample data
        loader = DataLoader(config)
        data = loader.load_all()
        
        # Prepare features (small sample for testing)
        fe = FeatureEngineer(config)
        X_train = fe.create_base_features(data['X_train'].head(500))
        y_train = data['y_train'].head(500)
        
        # Test probability calibration
        print("Testing probability calibration...")
        calibrator = ProbabilityCalibration()
        print(f"✅ ProbabilityCalibration initialized")
        
        # Test snapshot ensemble
        print("Testing snapshot ensemble...")
        snapshot = SnapshotEnsemble(snapshot_intervals=[100, 200])
        snapshots = snapshot.train_with_snapshots(X_train.head(200), y_train.head(200))
        
        print(f"✅ Snapshot ensemble trained: {len(snapshots)} models")
        
        # Test predictions
        predictions = snapshot.predict_ensemble(X_train.head(50))
        print(f"✅ Snapshot predictions: {predictions.shape}")
        
        # Test ranking models (simplified)
        print("Testing ranking models...")
        ranking = RankingOptimizedModels(config)
        
        # Use very small sample for XGBoost ranker test
        X_small = X_train.head(100)
        y_small = y_train.head(100)
        
        models = ranking.train_ranking_ensemble(X_small, y_small)
        print(f"✅ Ranking models trained: {len(models)} models")
        
        return True
        
    except Exception as e:
        print(f"❌ RankingModels test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ranking_models()
    sys.exit(0 if success else 1) 