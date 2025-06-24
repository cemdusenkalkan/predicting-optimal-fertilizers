import pandas as pd
import numpy as np
import time
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from .config import config
from .feature_engineering import FeatureEngineer
from .autogluon_trainer import AutoGluonTrainer

class HillClimbingOptimizer:
    """
    Hill Climbing optimization engine for iterative performance improvement
    
    Strategy:
    1. Start with base proven features (52 features)
    2. For each iteration, generate new feature sets
    3. Train AutoGluon models and evaluate performance
    4. Keep only features that improve CV score
    5. Save best experiments and track progress
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        self.feature_engineer = FeatureEngineer(cfg)
        self.trainer = AutoGluonTrainer(cfg)
        
        # Track best performance
        self.best_score = 0.0
        self.best_features = None
        self.best_iteration = -1
        self.iteration_history = []
        
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, X_orig: Optional[pd.DataFrame] = None,
                y_orig: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict]:
        """
        Run complete hill climbing optimization
        
        Returns:
            Best X_train_features, X_test_features, X_orig_features, optimization_results
        """
        self.logger.info("🧗 Starting Hill Climbing Optimization...")
        self.logger.info(f"Max iterations: {self.cfg.hill_climbing.max_iterations}")
        self.logger.info(f"Improvement threshold: {self.cfg.hill_climbing.improvement_threshold}")
        
        # Start with base proven features
        self.logger.info("Creating base proven features...")
        X_train_base = self.feature_engineer.create_base_features(X_train)
        X_test_base = self.feature_engineer.create_base_features(X_test)
        X_orig_base = self.feature_engineer.create_base_features(X_orig) if X_orig is not None else None
        
        # Add target encoding
        X_train_base, X_test_base, X_orig_base = self.feature_engineer.add_target_encoding(
            X_train_base, y_train, X_test_base, X_orig_base
        )
        
        self.logger.info(f"Base features: {X_train_base.shape[1]} columns")
        
        # Evaluate baseline performance
        baseline_score = self._evaluate_features(X_train_base, y_train, X_orig_base, y_orig)
        self.best_score = baseline_score
        self.best_features = X_train_base.copy()
        self.best_iteration = -1
        
        self.logger.info(f"✅ Baseline MAP@3: {baseline_score:.6f}")
        
        # Save baseline experiment
        self._save_iteration_results(-1, X_train_base, baseline_score, "baseline_proven_features")
        
        # Hill climbing iterations
        current_features_train = X_train_base.copy()
        current_features_test = X_test_base.copy()
        current_features_orig = X_orig_base.copy() if X_orig_base is not None else None
        
        for iteration in range(self.cfg.hill_climbing.max_iterations):
            self.logger.info(f"\n🔄 Hill Climbing Iteration {iteration}")
            
            improved_train, improved_test, improved_orig, iteration_score = self._run_iteration(
                iteration, current_features_train, current_features_test, current_features_orig,
                X_train, y_train, X_test, X_orig, y_orig
            )
            
            # Check if improvement exceeds threshold
            improvement = iteration_score - self.best_score
            self.logger.info(f"Iteration {iteration} score: {iteration_score:.6f} (improvement: {improvement:+.6f})")
            
            if improvement > self.cfg.hill_climbing.improvement_threshold:
                self.logger.info(f"✅ Iteration {iteration} ACCEPTED - improvement: {improvement:+.6f}")
                
                # Update best
                self.best_score = iteration_score
                self.best_features = improved_train.copy()
                self.best_iteration = iteration
                
                # Use improved features for next iteration
                current_features_train = improved_train
                current_features_test = improved_test
                current_features_orig = improved_orig
                
                # Save successful iteration
                self._save_iteration_results(iteration, improved_train, iteration_score, 
                                           f"iteration_{iteration}_accepted")
            else:
                self.logger.info(f"❌ Iteration {iteration} REJECTED - insufficient improvement: {improvement:+.6f}")
                
                # Save rejected iteration
                self._save_iteration_results(iteration, improved_train, iteration_score, 
                                           f"iteration_{iteration}_rejected")
        
        # Save best experiment
        best_test_features = self._align_features(current_features_test, current_features_train.columns)
        best_orig_features = self._align_features(current_features_orig, current_features_train.columns) if current_features_orig is not None else None
        
        self._save_best_experiment(current_features_train, best_test_features, best_orig_features)
        
        # Create optimization summary
        results = {
            'best_score': self.best_score,
            'best_iteration': self.best_iteration,
            'baseline_score': baseline_score,
            'total_improvement': self.best_score - baseline_score,
            'iterations_run': self.cfg.hill_climbing.max_iterations,
            'final_features': current_features_train.shape[1],
            'history': self.iteration_history
        }
        
        self.logger.info(f"\n🏆 Hill Climbing Complete!")
        self.logger.info(f"Best score: {self.best_score:.6f} (iteration {self.best_iteration})")
        self.logger.info(f"Total improvement: {results['total_improvement']:+.6f}")
        self.logger.info(f"Final features: {results['final_features']}")
        
        return current_features_train, best_test_features, best_orig_features, results
    
    def _run_iteration(self, iteration: int, current_train: pd.DataFrame, current_test: pd.DataFrame, 
                      current_orig: Optional[pd.DataFrame], X_train_raw: pd.DataFrame, y_train: pd.Series,
                      X_test_raw: pd.DataFrame, X_orig_raw: Optional[pd.DataFrame], 
                      y_orig: Optional[pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], float]:
        """Run single hill climbing iteration"""
        
        self.logger.info(f"Generating new features for iteration {iteration}...")
        
        # Generate hill climbing features for this iteration
        new_train_features = self.feature_engineer.create_hill_climbing_features(X_train_raw, iteration)
        new_test_features = self.feature_engineer.create_hill_climbing_features(X_test_raw, iteration)
        new_orig_features = self.feature_engineer.create_hill_climbing_features(X_orig_raw, iteration) if X_orig_raw is not None else None
        
        # Combine with current features
        combined_train = self._combine_features(current_train, new_train_features)
        combined_test = self._combine_features(current_test, new_test_features)
        combined_orig = self._combine_features(current_orig, new_orig_features) if current_orig is not None and new_orig_features is not None else current_orig
        
        self.logger.info(f"Combined features: {current_train.shape[1]} -> {combined_train.shape[1]}")
        
        # Evaluate combined features
        iteration_score = self._evaluate_features(combined_train, y_train, combined_orig, y_orig)
        
        # Store iteration history
        self.iteration_history.append({
            'iteration': iteration,
            'score': iteration_score,
            'features_before': current_train.shape[1],
            'features_after': combined_train.shape[1],
            'improvement': iteration_score - self.best_score
        })
        
        return combined_train, combined_test, combined_orig, iteration_score
    
    def _evaluate_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_orig: Optional[pd.DataFrame] = None, 
                          y_orig: Optional[pd.Series] = None) -> float:
        """
        Evaluate feature set using cross-validation
        
        Returns MAP@3 score
        """
        self.logger.info(f"Evaluating features: {X_train.shape}")
        
        # Apply data expansion
        from .data_loader import DataLoader
        data_loader = DataLoader(self.cfg)
        X_expanded, y_expanded = data_loader.expand_training_data(X_train, y_train, X_orig, y_orig)
        
        # Cross-validation evaluation
        cv_scores = []
        skf = StratifiedKFold(n_splits=self.cfg.hill_climbing.cv_folds, 
                             shuffle=True, random_state=self.cfg.competition.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_expanded, y_expanded)):
            try:
                X_fold_train = X_expanded.iloc[train_idx]
                y_fold_train = y_expanded.iloc[train_idx]
                X_fold_val = X_expanded.iloc[val_idx]
                y_fold_val = y_expanded.iloc[val_idx]
                
                # Train quick AutoGluon model for evaluation
                fold_score = self.trainer.quick_evaluate(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                cv_scores.append(fold_score)
                
                self.logger.debug(f"Fold {fold}: {fold_score:.6f}")
                
            except Exception as e:
                self.logger.warning(f"Fold {fold} failed: {e}")
                cv_scores.append(0.0)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        self.logger.info(f"CV MAP@3: {mean_score:.6f} ± {std_score:.6f}")
        return mean_score
    
    def _combine_features(self, current_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Combine current features with new features"""
        if current_df is None:
            return new_df
        if new_df is None:
            return current_df
            
        # Get new columns that don't exist in current
        current_cols = set(current_df.columns)
        new_cols = [col for col in new_df.columns if col not in current_cols]
        
        if not new_cols:
            self.logger.debug("No new features to add")
            return current_df
        
        # Add only new columns
        combined = current_df.copy()
        for col in new_cols:
            combined[col] = new_df[col]
        
        self.logger.debug(f"Added {len(new_cols)} new features: {new_cols[:5]}...")
        return combined
    
    def _align_features(self, df: Optional[pd.DataFrame], target_columns: List[str]) -> Optional[pd.DataFrame]:
        """Align dataframe columns with target columns"""
        if df is None:
            return None
            
        # Keep only columns that exist in both
        common_cols = [col for col in target_columns if col in df.columns]
        return df[common_cols].copy()
    
    def _save_iteration_results(self, iteration: int, features_df: pd.DataFrame, 
                               score: float, experiment_name: str):
        """Save iteration results for analysis"""
        try:
            iteration_dir = self.cfg.get_iteration_path(iteration) if iteration >= 0 else self.cfg.get_best_experiment_path()
            os.makedirs(iteration_dir, exist_ok=True)
            
            # Save feature info
            feature_info = {
                'iteration': iteration,
                'score': score,
                'num_features': features_df.shape[1],
                'feature_names': list(features_df.columns),
                'experiment_name': experiment_name,
                'timestamp': time.time()
            }
            
            with open(f"{iteration_dir}/feature_info.json", 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            # Save feature importance if available
            feature_types = {}
            for col in features_df.columns:
                if '_cat' in col or '_zone' in col:
                    feature_types[col] = 'categorical_binned'
                elif '_ratio' in col:
                    feature_types[col] = 'npk_ratio'
                elif '_score' in col:
                    feature_types[col] = 'fertilizer_score'
                elif 'env_' in col:
                    feature_types[col] = 'environmental'
                elif col in ['const']:
                    feature_types[col] = 'constant'
                else:
                    feature_types[col] = 'original'
            
            with open(f"{iteration_dir}/feature_types.json", 'w') as f:
                json.dump(feature_types, f, indent=2)
            
            self.logger.debug(f"Saved iteration {iteration} results to {iteration_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not save iteration {iteration} results: {e}")
    
    def _save_best_experiment(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             X_orig: Optional[pd.DataFrame]):
        """Save best experiment data"""
        try:
            best_dir = self.cfg.get_best_experiment_path()
            os.makedirs(best_dir, exist_ok=True)
            
            # Save datasets
            X_train.to_parquet(f"{best_dir}/X_train_best.parquet")
            X_test.to_parquet(f"{best_dir}/X_test_best.parquet")
            
            if X_orig is not None:
                X_orig.to_parquet(f"{best_dir}/X_orig_best.parquet")
            
            # Save optimization summary
            summary = {
                'best_score': self.best_score,
                'best_iteration': self.best_iteration,
                'final_features': X_train.shape[1],
                'optimization_completed': True,
                'timestamp': time.time()
            }
            
            with open(f"{best_dir}/optimization_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"✅ Best experiment saved to {best_dir}")
            
        except Exception as e:
            self.logger.error(f"Could not save best experiment: {e}")
    
    def load_best_experiment(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]]:
        """Load best experiment if available"""
        try:
            best_dir = self.cfg.get_best_experiment_path()
            
            if not os.path.exists(f"{best_dir}/optimization_summary.json"):
                return None
            
            X_train = pd.read_parquet(f"{best_dir}/X_train_best.parquet")
            X_test = pd.read_parquet(f"{best_dir}/X_test_best.parquet")
            
            X_orig = None
            if os.path.exists(f"{best_dir}/X_orig_best.parquet"):
                X_orig = pd.read_parquet(f"{best_dir}/X_orig_best.parquet")
            
            with open(f"{best_dir}/optimization_summary.json", 'r') as f:
                summary = json.load(f)
            
            self.best_score = summary['best_score']
            self.best_iteration = summary['best_iteration']
            
            self.logger.info(f"✅ Loaded best experiment: {self.best_score:.6f} MAP@3")
            return X_train, X_test, X_orig
            
        except Exception as e:
            self.logger.warning(f"Could not load best experiment: {e}")
            return None 