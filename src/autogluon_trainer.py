import pandas as pd
import numpy as np
import logging
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder
import gc
from .config import config

def map_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """Calculate MAP@K metric"""
    if len(y_true) == 0:
        return 0.0
    
    # Get top-k predictions
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    
    map_score = 0.0
    for i, true_label in enumerate(y_true):
        predicted_labels = top_k_indices[i]
        
        # Calculate average precision for this sample
        precision_sum = 0.0
        relevant_items = 0
        
        for j, pred_label in enumerate(predicted_labels):
            if pred_label == true_label:
                relevant_items += 1
                precision_sum += relevant_items / (j + 1)
        
        # Average precision for this sample
        if relevant_items > 0:
            map_score += precision_sum / min(relevant_items, k)
    
    return map_score / len(y_true)

class AutoGluonTrainer:
    """
    AutoGluon ensemble trainer with competitive optimizations
    
    Features:
    - Multi-algorithm ensemble (XGBoost, LightGBM, CatBoost, Neural Networks)
    - Automated stacking and bagging
    - Advanced hyperparameter optimization
    - Memory-efficient training
    - MAP@3 evaluation
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        self.predictor = None
        
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_orig: Optional[pd.DataFrame] = None, 
                         y_orig: Optional[pd.Series] = None) -> TabularPredictor:
        """
        Train final competitive AutoGluon model
        
        Uses all proven techniques:
        - Data expansion
        - Multi-algorithm ensemble
        - Advanced stacking
        - Optimal hyperparameters
        """
        self.logger.info("🚀 Training final AutoGluon ensemble...")
        
        # Prepare data with expansion
        from .data_loader import DataLoader
        data_loader = DataLoader(self.cfg)
        X_expanded, y_expanded = data_loader.expand_training_data(X_train, y_train, X_orig, y_orig)
        
        self.logger.info(f"Final training data: {X_expanded.shape}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_expanded)
        
        # Create training dataframe
        train_df = X_expanded.copy()
        train_df[self.cfg.models.predictor_label] = y_encoded
        
        # Enhanced configuration for final model
        final_hyperparameters = self._get_final_hyperparameters()
        
        # Train predictor with maximum performance settings
        save_path = f"{self.cfg.competition.models_dir}/final_autogluon_model"
        os.makedirs(save_path, exist_ok=True)
        
        self.predictor = TabularPredictor(
            label=self.cfg.models.predictor_label,
            problem_type='multiclass',
            eval_metric=self.cfg.models.eval_metric,
            path=save_path,
            verbosity=self.cfg.models.verbosity
        )
        
        # Final training with maximum quality
        self.predictor.fit(
            train_data=train_df,
            hyperparameters=final_hyperparameters,
            time_limit=self.cfg.models.final_time_limit,
            presets=self.cfg.models.presets,
            auto_stack=self.cfg.models.auto_stack,
            num_bag_folds=5,  # Maximum for final model
            num_stack_levels=2,  # Maximum stacking
            excluded_model_types=self.cfg.models.excluded_model_types,
            refit_full=True  # Use all data for final model
        )
        
        # Log model performance
        self._log_model_info()
        
        self.logger.info("✅ Final AutoGluon model trained")
        return self.predictor
    
    def quick_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Quick model evaluation for hill climbing
        
        Uses faster settings for speed while maintaining accuracy
        """
        try:
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val)
            
            # Create training dataframe
            train_df = X_train.copy()
            train_df[self.cfg.models.predictor_label] = y_train_encoded
            
            # Quick hyperparameters for evaluation
            quick_hyperparameters = self._get_quick_hyperparameters()
            
            # Create temporary predictor
            temp_path = f"{self.cfg.competition.models_dir}/temp_eval_{int(time.time())}"
            os.makedirs(temp_path, exist_ok=True)
            
            predictor = TabularPredictor(
                label=self.cfg.models.predictor_label,
                problem_type='multiclass', 
                eval_metric=self.cfg.models.eval_metric,
                path=temp_path,
                verbosity=0  # Silent for speed
            )
            
            # Quick training
            predictor.fit(
                train_data=train_df,
                hyperparameters=quick_hyperparameters,
                time_limit=self.cfg.models.time_limit_per_fold,
                presets='medium_quality_faster_train',
                auto_stack=False,  # No stacking for speed
                num_bag_folds=self.cfg.hill_climbing.cv_folds,
                excluded_model_types=self.cfg.models.excluded_model_types + ['NN_TORCH', 'FASTAI']  # Exclude slow models
            )
            
            # Predict and evaluate
            y_pred_proba = predictor.predict_proba(X_val)
            
            # Calculate MAP@3
            map3_score = map_at_k(y_val_encoded, y_pred_proba.values, k=3)
            
            # Cleanup
            del predictor
            gc.collect()
            
            # Remove temp directory
            import shutil
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
            
            return map3_score
            
        except Exception as e:
            self.logger.warning(f"Quick evaluation failed: {e}")
            return 0.0
    
    def predict_submission(self, X_test: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
        """
        Generate final submission predictions
        
        Returns space-separated top-3 predictions
        """
        if self.predictor is None:
            raise ValueError("Model not trained. Call train_final_model first.")
        
        self.logger.info(f"Generating predictions for {len(X_test)} samples...")
        
        # Get prediction probabilities
        y_pred_proba = self.predictor.predict_proba(X_test)
        
        # Get top 3 predictions
        top3_indices = np.argsort(y_pred_proba.values, axis=1)[:, ::-1][:, :3]
        
        # Convert back to fertilizer names
        top3_fertilizers = self.label_encoder.inverse_transform(top3_indices.flatten()).reshape(-1, 3)
        
        # Create space-separated predictions
        predictions = []
        for i in range(len(top3_fertilizers)):
            pred_str = ' '.join(top3_fertilizers[i])
            predictions.append(pred_str)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            self.cfg.competition.id_column: test_ids,
            self.cfg.competition.target_column: predictions
        })
        
        self.logger.info("✅ Submission predictions generated")
        self.logger.info(f"Sample predictions:\n{submission.head()}")
        
        return submission
    
    def _get_final_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters optimized for final competitive model"""
        return {
            'GBM': [
                # XGBoost variants
                {
                    'extra_trees': True,
                    'ag_args': {'name_suffix': 'XT', 'priority': 1}
                },
                {
                    'boosting': 'gbtree',
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'ag_args': {'name_suffix': 'OptimalXGB', 'priority': 1}
                },
                # LightGBM variants
                {
                    'boosting': 'dart',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'ag_args': {'name_suffix': 'DART', 'priority': 1}
                },
                {
                    'boosting': 'goss',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'max_depth': 8,
                    'learning_rate': 0.03,
                    'ag_args': {'name_suffix': 'GOSS', 'priority': 1}
                },
            ],
            'CAT': {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 8,
                'l2_leaf_reg': 3,
                'border_count': 254,
                'thread_count': -1,
                'ag_args': {'name_suffix': 'Optimal', 'priority': 1}
            },
            'NN_TORCH': [
                {
                    'num_epochs': 100,
                    'learning_rate': 0.01,
                    'batch_size': 512,
                    'dropout_prob': 0.3,
                    'ag_args': {'name_suffix': 'Deep', 'priority': 0}
                }
            ],
            'FASTAI': {
                'epochs': 50,
                'lr': 0.01,
                'ag_args': {'name_suffix': 'TabNet', 'priority': 0}
            },
            'RF': [
                {
                    'n_estimators': 500,
                    'max_features': 0.8,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'ag_args': {'name_suffix': 'Ensemble', 'priority': 0}
                }
            ]
        }
    
    def _get_quick_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters optimized for speed in hill climbing"""
        return {
            'GBM': [
                # Fast XGBoost
                {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'ag_args': {'name_suffix': 'Fast'}
                },
                # Fast LightGBM
                {
                    'boosting': 'gbdt',
                    'objective': 'multiclass',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'ag_args': {'name_suffix': 'FastLGB'}
                }
            ],
            'CAT': {
                'iterations': 200,
                'learning_rate': 0.1,
                'depth': 6,
                'ag_args': {'name_suffix': 'FastCAT'}
            },
            'RF': [
                {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'ag_args': {'name_suffix': 'FastRF'}
                }
            ]
        }
    
    def _log_model_info(self):
        """Log information about trained model"""
        if self.predictor is not None:
            try:
                leaderboard = self.predictor.leaderboard(silent=True)
                self.logger.info(f"Model leaderboard:\n{leaderboard.head(10)}")
                
                # Feature importance
                feature_importance = self.predictor.feature_importance(data=None, silent=True)
                self.logger.info(f"Top 10 features:\n{feature_importance.head(10)}")
                
            except Exception as e:
                self.logger.warning(f"Could not log model info: {e}")
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if self.predictor is not None:
            if path is None:
                path = f"{self.cfg.competition.models_dir}/final_model"
            
            # AutoGluon automatically saves during training
            self.logger.info(f"Model saved at: {self.predictor.path}")
        else:
            self.logger.warning("No model to save")
    
    def load_model(self, path: str) -> TabularPredictor:
        """Load saved model"""
        try:
            self.predictor = TabularPredictor.load(path)
            self.logger.info(f"✅ Model loaded from {path}")
            return self.predictor
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {e}")
            raise 