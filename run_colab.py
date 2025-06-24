#!/usr/bin/env python3
"""
Colab-optimized competitive fertilizer prediction runner

This script is designed to run efficiently in Google Colab with:
- Reduced dataset sizes for faster execution
- Simplified model configurations
- Clear progress tracking
"""

import pandas as pd
import numpy as np
import logging
import time
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_colab_environment():
    """Setup Colab environment"""
    logger.info("🔧 Setting up Colab environment...")
    
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    # Install requirements if needed
    try:
        import xgboost
        import lightgbm
        import catboost
    except ImportError:
        logger.info("📦 Installing required packages...")
        os.system("pip install xgboost lightgbm catboost scipy")
    
    logger.info("✅ Colab environment ready")

def run_colab_pipeline():
    """Run optimized pipeline for Colab"""
    logger.info("🏆 Starting Colab Competitive Pipeline")
    start_time = time.time()
    
    try:
        # Import modules
        from src.predictor import CompetitiveFertilizerPredictor
        from src.config import config
        
        # Initialize predictor
        predictor = CompetitiveFertilizerPredictor(config)
        
        # Run pipeline with meta-stacking (most stable)
        logger.info("🚀 Running meta-stacking pipeline...")
        submission = predictor.run_competitive_pipeline(
            use_ranking=False,  # Disable ranking models for stability
            use_meta_stacking=True
        )
        
        # Display results
        logger.info(f"\n📋 Submission Generated!")
        logger.info(f"   Shape: {submission.shape}")
        logger.info(f"   Columns: {list(submission.columns)}")
        logger.info(f"   Unique predictions: {submission['Fertilizer Name'].nunique()}")
        
        # Show sample
        logger.info(f"\n🔍 Sample predictions:")
        print(submission.head(10))
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ Pipeline completed in {total_time:.1f}s")
        logger.info(f"📁 Submission saved to: submissions/submission.csv")
        
        return submission
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_quick_test():
    """Run quick functionality test"""
    logger.info("🧪 Running quick functionality test...")
    
    try:
        # Test imports
        from src.data_loader import DataLoader
        from src.feature_engineering import FeatureEngineer
        from src.meta_stacking import MetaStackingEnsemble
        from src.config import config
        
        logger.info("✅ All imports successful")
        
        # Test data loading
        loader = DataLoader(config)
        data = loader.load_all()
        logger.info(f"✅ Data loaded: train {data['X_train'].shape}, test {data['X_test'].shape}")
        
        # Test feature engineering
        fe = FeatureEngineer(config)
        X_features = fe.create_base_features(data['X_train'].head(100))
        logger.info(f"✅ Features created: {X_features.shape}")
        
        logger.info("🎉 Quick test passed! Ready for full pipeline.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Colab execution"""
    print("🚀 Competitive Fertilizer Prediction - Colab Runner")
    print("=" * 50)
    
    # Setup environment
    setup_colab_environment()
    
    # Run quick test first
    if not run_quick_test():
        logger.error("Quick test failed. Please check your setup.")
        return
    
    # Ask user what to run
    print("\nChoose execution mode:")
    print("1. Quick test only (already completed)")
    print("2. Full competitive pipeline")
    print("3. Generate submission")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except:
        choice = "3"  # Default to submission generation
    
    if choice == "2" or choice == "3":
        submission = run_colab_pipeline()
        if submission is not None:
            print("\n🎉 Success! Check submissions/submission.csv")
        else:
            print("\n❌ Pipeline failed. Check logs for details.")
    else:
        print("\n✅ Quick test completed successfully!")

if __name__ == "__main__":
    main() 