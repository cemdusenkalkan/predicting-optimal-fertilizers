#!/usr/bin/env python3
"""
Fertilizer Competition - Main Execution Script

AutoGluon + Hill Climbing approach for maximum competitive performance
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.predictor import FertilizerPredictor

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs(config.competition.logs_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{config.competition.logs_dir}/competition_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete: {log_file}")
    return logger

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Fertilizer Competition Pipeline')
    parser.add_argument('--skip-hill-climbing', action='store_true',
                       help='Skip hill climbing optimization and use base features')
    parser.add_argument('--quick', action='store_true',
                       help='Quick prediction using existing model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance only')
    parser.add_argument('--config', type=str,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config.load_from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Create directories
    config.create_directories()
    
    # Initialize predictor
    predictor = FertilizerPredictor(config)
    
    try:
        if args.evaluate:
            # Evaluation mode
            logger.info("🔍 Running evaluation mode...")
            metrics = predictor.evaluate_model()
            
            logger.info("📊 Model Performance:")
            for metric, value in metrics.items():
                logger.info(f"   {metric}: {value}")
            
        elif args.quick:
            # Quick prediction mode
            logger.info("⚡ Running quick prediction mode...")
            submission = predictor.quick_predict()
            logger.info(f"✅ Quick prediction complete: {len(submission)} predictions")
            
        else:
            # Full pipeline mode
            logger.info("🏆 Running complete competition pipeline...")
            submission = predictor.run_complete_pipeline(
                skip_hill_climbing=args.skip_hill_climbing
            )
            
            # Log final results
            logger.info("🎯 Final Results:")
            logger.info(f"   Submission samples: {len(submission)}")
            logger.info(f"   Sample predictions:\n{submission.head()}")
            
            # Optional: Show model info
            feature_importance = predictor.get_feature_importance()
            if feature_importance is not None:
                logger.info(f"   Top 5 features:\n{feature_importance.head()}")
            
            leaderboard = predictor.get_model_leaderboard()
            if leaderboard is not None:
                logger.info(f"   Model leaderboard:\n{leaderboard.head()}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    logger.info("✅ Execution completed successfully!")

if __name__ == "__main__":
    main() 