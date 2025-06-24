import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Dict, Any
from .config import config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .hill_climbing import HillClimbingOptimizer
from .autogluon_trainer import AutoGluonTrainer

class FertilizerPredictor:
    """
    Main predictor class that orchestrates the complete competition pipeline
    
    Pipeline:
    1. Load and preprocess data
    2. Run hill climbing optimization for best features
    3. Train final AutoGluon ensemble
    4. Generate submission predictions
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader(cfg)
        self.feature_engineer = FeatureEngineer(cfg)
        self.hill_climber = HillClimbingOptimizer(cfg)
        self.trainer = AutoGluonTrainer(cfg)
        
        # Data storage
        self.data = None
        self.best_features = None
        self.model = None
        
    def run_complete_pipeline(self, skip_hill_climbing: bool = False) -> pd.DataFrame:
        """
        Run the complete competition pipeline
        
        Args:
            skip_hill_climbing: If True, uses pre-optimized features
            
        Returns:
            submission DataFrame ready for competition
        """
        self.logger.info("🏆 Starting Complete Fertilizer Competition Pipeline")
        start_time = time.time()
        
        # Step 1: Load data
        self.logger.info("\n📊 Step 1: Loading Data")
        self.data = self.data_loader.load_all()
        
        # Step 2: Feature optimization (hill climbing or load best)
        if not skip_hill_climbing:
            self.logger.info("\n🧗 Step 2: Hill Climbing Feature Optimization")
            best_features = self._run_hill_climbing()
        else:
            self.logger.info("\n📂 Step 2: Loading Best Features")
            best_features = self._load_or_create_features()
        
        # Step 3: Train final model
        self.logger.info("\n🚀 Step 3: Training Final AutoGluon Ensemble")
        self.model = self._train_final_model(best_features)
        
        # Step 4: Generate submission
        self.logger.info("\n🎯 Step 4: Generating Final Submission")
        submission = self._generate_submission(best_features)
        
        # Save submission
        self._save_submission(submission)
        
        total_time = time.time() - start_time
        self.logger.info(f"\n✅ Pipeline Complete! Total time: {total_time:.1f}s")
        self.logger.info(f"Submission saved: {self.cfg.competition.submissions_dir}/submission.csv")
        
        return submission
    
    def _run_hill_climbing(self) -> Dict[str, pd.DataFrame]:
        """Run hill climbing optimization"""
        try:
            # Check if best experiment already exists
            existing_features = self.hill_climber.load_best_experiment()
            if existing_features is not None:
                X_train_best, X_test_best, X_orig_best = existing_features
                self.logger.info("✅ Loaded existing hill climbing results")
                return {
                    'X_train': X_train_best,
                    'X_test': X_test_best, 
                    'X_orig': X_orig_best
                }
            
            # Run hill climbing optimization
            X_train_best, X_test_best, X_orig_best, results = self.hill_climber.optimize(
                self.data['X_train'],
                self.data['y_train'],
                self.data['X_test'],
                self.data['X_orig'],
                self.data['y_orig']
            )
            
            self.logger.info(f"✅ Hill climbing completed:")
            self.logger.info(f"   Best score: {results['best_score']:.6f}")
            self.logger.info(f"   Improvement: {results['total_improvement']:+.6f}")
            self.logger.info(f"   Best iteration: {results['best_iteration']}")
            
            return {
                'X_train': X_train_best,
                'X_test': X_test_best,
                'X_orig': X_orig_best
            }
            
        except Exception as e:
            self.logger.error(f"Hill climbing failed: {e}")
            self.logger.info("Falling back to base features...")
            return self._load_or_create_features()
    
    def _load_or_create_features(self) -> Dict[str, pd.DataFrame]:
        """Load existing features or create base features"""
        self.logger.info("Creating base proven features...")
        
        # Create base features with all proven techniques
        X_train_base = self.feature_engineer.create_base_features(self.data['X_train'])
        X_test_base = self.feature_engineer.create_base_features(self.data['X_test'])
        X_orig_base = self.feature_engineer.create_base_features(self.data['X_orig']) if self.data['X_orig'] is not None else None
        
        # Add target encoding
        X_train_encoded, X_test_encoded, X_orig_encoded = self.feature_engineer.add_target_encoding(
            X_train_base, self.data['y_train'], X_test_base, X_orig_base
        )
        
        self.logger.info(f"✅ Base features created: {X_train_encoded.shape[1]} features")
        
        return {
            'X_train': X_train_encoded,
            'X_test': X_test_encoded,
            'X_orig': X_orig_encoded
        }
    
    def _train_final_model(self, features: Dict[str, pd.DataFrame]) -> Any:
        """Train final AutoGluon model"""
        try:
            model = self.trainer.train_final_model(
                features['X_train'],
                self.data['y_train'],
                features['X_orig'],
                self.data['y_orig']
            )
            
            self.logger.info("✅ Final model training completed")
            return model
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    def _generate_submission(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate final submission predictions"""
        try:
            submission = self.trainer.predict_submission(
                features['X_test'],
                self.data['test'][self.cfg.competition.id_column]
            )
            
            self.logger.info(f"✅ Generated {len(submission)} predictions")
            return submission
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            raise
    
    def _save_submission(self, submission: pd.DataFrame):
        """Save submission with timestamp"""
        import os
        from datetime import datetime
        
        # Create submissions directory
        os.makedirs(self.cfg.competition.submissions_dir, exist_ok=True)
        
        # Save main submission
        submission_path = f"{self.cfg.competition.submissions_dir}/submission.csv"
        submission.to_csv(submission_path, index=False)
        
        # Save timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.cfg.competition.submissions_dir}/submission_{timestamp}.csv"
        submission.to_csv(backup_path, index=False)
        
        self.logger.info(f"Submission saved: {submission_path}")
        self.logger.info(f"Backup saved: {backup_path}")
    
    def quick_predict(self, X_test: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Quick prediction using existing best features and model
        
        Useful for fast re-runs without retraining
        """
        if X_test is None:
            if self.data is None:
                self.data = self.data_loader.load_all()
            X_test = self.data['X_test']
            test_ids = self.data['test'][self.cfg.competition.id_column]
        else:
            test_ids = X_test.index
        
        # Load or create features
        if self.best_features is None:
            self.best_features = self._load_or_create_features()
        
        # Load or train model
        if self.model is None:
            try:
                model_path = f"{self.cfg.competition.models_dir}/final_autogluon_model"
                self.model = self.trainer.load_model(model_path)
            except:
                self.logger.info("No existing model found, training new model...")
                self.model = self._train_final_model(self.best_features)
        
        # Generate predictions
        submission = self.trainer.predict_submission(self.best_features['X_test'], test_ids)
        return submission
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate model performance using cross-validation
        
        Returns performance metrics
        """
        if self.data is None:
            self.data = self.data_loader.load_all()
        
        if self.best_features is None:
            self.best_features = self._load_or_create_features()
        
        # Quick cross-validation evaluation
        cv_score = self.trainer.quick_evaluate(
            self.best_features['X_train'],
            self.data['y_train'],
            self.best_features['X_train'][:1000],  # Sample for speed
            self.data['y_train'][:1000]
        )
        
        return {
            'cv_map3': cv_score,
            'features_count': self.best_features['X_train'].shape[1],
            'training_samples': len(self.best_features['X_train'])
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from trained model"""
        if self.model is not None:
            try:
                return self.model.feature_importance(data=None, silent=True)
            except Exception as e:
                self.logger.warning(f"Could not get feature importance: {e}")
        return None
    
    def get_model_leaderboard(self) -> Optional[pd.DataFrame]:
        """Get model leaderboard from AutoGluon"""
        if self.model is not None:
            try:
                return self.model.leaderboard(silent=True)
            except Exception as e:
                self.logger.warning(f"Could not get model leaderboard: {e}")
        return None 