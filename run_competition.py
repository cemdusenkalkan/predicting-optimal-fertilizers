#!/usr/bin/env python3
"""
Competitive Fertilizer Prediction Pipeline

Targeting 0.36+ MAP@3 using proven techniques:
- Direct MAP@3 optimization with ranking models
- Meta-stacking with diverse base learners
- High-fold stratified CV
- Probability calibration and threshold optimization

Usage:
    python run_competition.py --mode competitive              # Full competitive pipeline
    python run_competition.py --mode ranking                  # Ranking models only
    python run_competition.py --mode meta-stacking            # Meta-stacking only
    python run_competition.py --mode snapshot                 # Snapshot ensemble
    python run_competition.py --mode evaluate                 # Evaluate model
"""

import argparse
import logging
import sys
import os
import pandas as pd
from src.predictor import CompetitiveFertilizerPredictor
from src.config import config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/competition.log', mode='a')
        ]
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

def run_competitive_pipeline():
    """Run full competitive pipeline with all techniques"""
    logger = logging.getLogger(__name__)
    logger.info("🏆 Starting Full Competitive Pipeline")
    
    try:
        predictor = CompetitiveFertilizerPredictor()
        submission = predictor.run_competitive_pipeline(
            use_ranking=True,
            use_meta_stacking=True
        )
        
        logger.info("✅ Competitive pipeline completed successfully")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Target MAP@3: 0.36+")
        
        return submission
        
    except Exception as e:
        logger.error(f"Competitive pipeline failed: {e}")
        raise

def run_ranking_pipeline():
    """Run ranking-optimized models only"""
    logger = logging.getLogger(__name__)
    logger.info("🎯 Starting Ranking Pipeline")
    
    try:
        predictor = CompetitiveFertilizerPredictor()
        submission = predictor.run_competitive_pipeline(
            use_ranking=True,
            use_meta_stacking=False
        )
        
        logger.info("✅ Ranking pipeline completed successfully")
        logger.info(f"Submission shape: {submission.shape}")
        
        return submission
        
    except Exception as e:
        logger.error(f"Ranking pipeline failed: {e}")
        raise

def run_meta_stacking_pipeline():
    """Run meta-stacking ensemble only"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Starting Meta-Stacking Pipeline")
    
    try:
        predictor = CompetitiveFertilizerPredictor()
        submission = predictor.run_competitive_pipeline(
            use_ranking=False,
            use_meta_stacking=True
        )
        
        logger.info("✅ Meta-stacking pipeline completed successfully")
        logger.info(f"Submission shape: {submission.shape}")
        
        return submission
        
    except Exception as e:
        logger.error(f"Meta-stacking pipeline failed: {e}")
        raise

def run_snapshot_pipeline():
    """Run snapshot ensemble only"""
    logger = logging.getLogger(__name__)
    logger.info("📸 Starting Snapshot Pipeline")
    
    try:
        predictor = CompetitiveFertilizerPredictor()
        submission = predictor.run_competitive_pipeline(
            use_ranking=False,
            use_meta_stacking=False
        )
        
        logger.info("✅ Snapshot pipeline completed successfully")
        logger.info(f"Submission shape: {submission.shape}")
        
        return submission
        
    except Exception as e:
        logger.error(f"Snapshot pipeline failed: {e}")
        raise

def evaluate_model():
    """Evaluate trained model performance"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Evaluating Model Performance")
    
    try:
        predictor = CompetitiveFertilizerPredictor()
        metrics = predictor.evaluate_model()
        
        logger.info("📈 Model Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, list):
                logger.info(f"   {metric}: {value}")
            else:
                logger.info(f"   {metric}: {value:.6f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise

def main():
    """Main competition runner"""
    parser = argparse.ArgumentParser(description='Competitive Fertilizer Prediction Pipeline')
    
    parser.add_argument('--mode', choices=['competitive', 'ranking', 'meta-stacking', 'snapshot', 'evaluate'], 
                       default='competitive', help='Pipeline mode to run')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Competitive Fertilizer Prediction Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info("Target: 0.36+ MAP@3 using proven techniques")
    
    try:
        if args.mode == 'competitive':
            logger.info("🏆 Running Full Competitive Pipeline")
            submission = run_competitive_pipeline()
            
        elif args.mode == 'ranking':
            logger.info("🎯 Running Ranking Models Pipeline")
            submission = run_ranking_pipeline()
            
        elif args.mode == 'meta-stacking':
            logger.info("🔧 Running Meta-Stacking Pipeline")
            submission = run_meta_stacking_pipeline()
            
        elif args.mode == 'snapshot':
            logger.info("📸 Running Snapshot Ensemble Pipeline")
            submission = run_snapshot_pipeline()
            
        elif args.mode == 'evaluate':
            logger.info("📊 Evaluating Model")
            metrics = evaluate_model()
            return
        
        # Display results
        logger.info(f"\n📋 Final Submission Summary:")
        logger.info(f"   Rows: {len(submission)}")
        logger.info(f"   Columns: {list(submission.columns)}")
        logger.info(f"   Unique predictions: {submission[config.competition.target_column].nunique()}")
        
        # Show sample predictions
        logger.info(f"\n🔍 Sample Predictions:")
        sample_size = min(10, len(submission))
        for _, row in submission.head(sample_size).iterrows():
            logger.info(f"   ID {row[config.competition.id_column]}: {row[config.competition.target_column]}")
        
        logger.info(f"\n✅ Competition pipeline completed successfully!")
        logger.info(f"📁 Submission saved to: {config.competition.submissions_dir}/submission.csv")
        
        # Expected performance
        logger.info(f"\n🎯 Expected Performance:")
        logger.info(f"   Baseline (current): 0.32-0.33 MAP@3")
        logger.info(f"   Target (competitive): 0.36+ MAP@3")
        logger.info(f"   Techniques used:")
        logger.info(f"     ✓ Agricultural domain features")
        logger.info(f"     ✓ Direct MAP@3 optimization") 
        logger.info(f"     ✓ Meta-stacking ensemble")
        logger.info(f"     ✓ Probability calibration")
        logger.info(f"     ✓ Threshold optimization")
        logger.info(f"     ✓ Agricultural data augmentation")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 