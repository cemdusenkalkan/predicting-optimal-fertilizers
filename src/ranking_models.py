import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging
from typing import Dict, List, Tuple, Any, Optional
from .config import config

def map_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """Calculate MAP@K metric"""
    if len(y_true) == 0:
        return 0.0
    
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    
    map_score = 0.0
    for i, true_label in enumerate(y_true):
        predicted_labels = top_k_indices[i]
        
        precision_sum = 0.0
        relevant_items = 0
        
        for j, pred_label in enumerate(predicted_labels):
            if pred_label == true_label:
                relevant_items += 1
                precision_sum += relevant_items / (j + 1)
        
        if relevant_items > 0:
            map_score += precision_sum / min(relevant_items, k)
    
    return map_score / len(y_true)

class RankingOptimizedModels:
    """
    Models optimized specifically for MAP@3 ranking
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        
    def create_xgb_ranker(self) -> xgb.XGBRanker:
        """XGBoost with rank:pairwise objective"""
        return xgb.XGBRanker(
            objective='rank:pairwise',
            eval_metric='map@3',
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            enable_categorical=True
        )
    
    def create_catboost_ranker(self) -> cb.CatBoostRanker:
        """CatBoost with YetiRankPairwise loss"""
        return cb.CatBoostRanker(
            loss_function='YetiRankPairwise',
            custom_metric=['MAP:top=3'],
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            thread_count=-1
        )
    
    def create_lgb_ranker(self) -> lgb.LGBMRanker:
        """LightGBM LambdaMART for ranking"""
        return lgb.LGBMRanker(
            objective='lambdarank',
            metric='map',
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    
    def prepare_ranking_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for ranking models"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create group sizes (each sample is its own group for this problem)
        group_sizes = np.ones(len(X), dtype=int)
        
        # Handle categorical features
        X_processed = X.copy()
        cat_features = []
        for col in X.select_dtypes(include=['category']).columns:
            cat_features.append(col)
            # Keep as category for CatBoost, convert to codes for others
        
        return X_processed, y_encoded, group_sizes
    
    def train_ranking_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble of ranking models"""
        self.logger.info("Training ranking-optimized ensemble")
        
        X_processed, y_encoded, group_sizes = self.prepare_ranking_data(X, y)
        
        models = {}
        
        # XGBoost Ranker
        self.logger.info("Training XGBoost Ranker")
        xgb_ranker = self.create_xgb_ranker()
        
        # Convert all categorical and object columns to codes for XGBoost
        X_xgb = X_processed.copy()
        for col in X_processed.select_dtypes(include=['category', 'object']).columns:
            if X_xgb[col].dtype.name == 'category':
                X_xgb[col] = X_xgb[col].cat.codes
            else:
                # Convert object to category then to codes
                X_xgb[col] = pd.Categorical(X_xgb[col]).codes
        
        xgb_ranker.fit(X_xgb, y_encoded, group=group_sizes)
        models['xgb_ranker'] = xgb_ranker
        
        # CatBoost Ranker
        self.logger.info("Training CatBoost Ranker")
        cat_ranker = self.create_catboost_ranker()
        cat_features = X_processed.select_dtypes(include=['category']).columns.tolist()
        
        # Get all categorical and object features for CatBoost
        all_cat_features = X_processed.select_dtypes(include=['category', 'object']).columns.tolist()
        
        if len(all_cat_features) > 0:
            cat_ranker.fit(X_processed, y_encoded, group_id=np.arange(len(X)), cat_features=all_cat_features)
        else:
            cat_ranker.fit(X_processed, y_encoded, group_id=np.arange(len(X)))
        models['cat_ranker'] = cat_ranker
        
        # LightGBM Ranker
        self.logger.info("Training LightGBM Ranker")
        lgb_ranker = self.create_lgb_ranker()
        
        # Convert all categorical and object columns to codes for LightGBM
        X_lgb = X_processed.copy()
        for col in X_processed.select_dtypes(include=['category', 'object']).columns:
            if X_lgb[col].dtype.name == 'category':
                X_lgb[col] = X_lgb[col].cat.codes
            else:
                # Convert object to category then to codes
                X_lgb[col] = pd.Categorical(X_lgb[col]).codes
        
        lgb_ranker.fit(X_lgb, y_encoded, group=group_sizes)
        models['lgb_ranker'] = lgb_ranker
        
        return models
    
    def predict_ensemble(self, models: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions from ranking models"""
        predictions = []
        
        for model_name, model in models.items():
            X_processed = X.copy()
            
            # Handle categorical features based on model type
            if model_name == 'cat_ranker':
                # CatBoost can handle categoricals natively
                pass
            else:
                # Convert categorical and object to codes for XGB and LGB
                for col in X.select_dtypes(include=['category', 'object']).columns:
                    if X_processed[col].dtype.name == 'category':
                        X_processed[col] = X_processed[col].cat.codes
                    else:
                        # Convert object to category then to codes
                        X_processed[col] = pd.Categorical(X_processed[col]).codes
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_processed)
            else:
                # For rankers, predict returns rankings/scores
                scores = model.predict(X_processed)
                # Convert to probabilities using softmax
                pred = self._scores_to_proba(scores)
            
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def _scores_to_proba(self, scores: np.ndarray) -> np.ndarray:
        """Convert ranking scores to probabilities"""
        n_classes = len(self.label_encoder.classes_)
        
        # For single-dimensional scores, create probability matrix
        if scores.ndim == 1:
            # Create uniform probabilities (rankers output scores, not probabilities)
            proba = np.ones((len(scores), n_classes)) / n_classes
            return proba
        else:
            return scores

class ProbabilityCalibration:
    """
    Probability calibration for improved MAP@3 performance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temperature_params = {}
        
    def fit_temperature_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               n_classes: int) -> Dict[int, float]:
        """Fit class-wise temperature scaling parameters"""
        from scipy.optimize import minimize_scalar
        
        temperatures = {}
        
        for class_idx in range(n_classes):
            # Get samples for this class
            class_mask = (y_true == class_idx)
            if not np.any(class_mask):
                temperatures[class_idx] = 1.0
                continue
            
            class_probs = y_pred_proba[class_mask, class_idx]
            
            def temperature_loss(temp):
                if temp <= 0:
                    return float('inf')
                
                # Apply temperature scaling
                scaled_probs = class_probs ** (1.0 / temp)
                scaled_probs = scaled_probs / (scaled_probs + (1 - class_probs) ** (1.0 / temp))
                
                # Log loss
                eps = 1e-15
                scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
                loss = -np.mean(np.log(scaled_probs))
                
                return loss
            
            # Optimize temperature
            result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
            temperatures[class_idx] = result.x
            
        self.temperature_params = temperatures
        return temperatures
    
    def apply_temperature_scaling(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply fitted temperature scaling"""
        calibrated_proba = y_pred_proba.copy()
        
        for class_idx, temp in self.temperature_params.items():
            if temp != 1.0:
                # Apply temperature to this class
                class_probs = calibrated_proba[:, class_idx]
                calibrated_proba[:, class_idx] = class_probs ** (1.0 / temp)
        
        # Renormalize
        row_sums = calibrated_proba.sum(axis=1, keepdims=True)
        calibrated_proba = calibrated_proba / row_sums
        
        return calibrated_proba
    
    def optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Optimize probability thresholds for MAP@3"""
        from scipy.optimize import minimize
        
        n_classes = y_pred_proba.shape[1]
        
        def threshold_objective(thresholds):
            # Apply thresholds to probabilities
            adjusted_proba = y_pred_proba * thresholds
            
            # Renormalize
            row_sums = adjusted_proba.sum(axis=1, keepdims=True)
            adjusted_proba = adjusted_proba / row_sums
            
            # Calculate negative MAP@3 (minimize negative = maximize positive)
            map_score = map_at_k(y_true, adjusted_proba, k=3)
            return -map_score
        
        # Initial thresholds (all equal)
        initial_thresholds = np.ones(n_classes)
        
        # Constraints: all thresholds positive, sum to n_classes
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - n_classes}
        ]
        bounds = [(0.1, 5.0) for _ in range(n_classes)]
        
        # Optimize
        result = minimize(
            threshold_objective,
            initial_thresholds,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_thresholds = result.x
        
        return {
            'thresholds': optimal_thresholds,
            'map_improvement': -result.fun - map_at_k(y_true, y_pred_proba, k=3)
        }

class SnapshotEnsemble:
    """
    Snapshot ensembling for XGBoost
    """
    
    def __init__(self, snapshot_intervals: List[int] = None):
        self.snapshot_intervals = snapshot_intervals or [300, 500, 700, 900, 1100, 1300]
        self.snapshots = []
        self.logger = logging.getLogger(__name__)
    
    def train_with_snapshots(self, X: pd.DataFrame, y: pd.Series) -> List[Any]:
        """Train XGBoost with snapshot saving"""
        self.logger.info(f"Training XGBoost with snapshots at: {self.snapshot_intervals}")
        
        # Encode labels and features
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_processed = X.copy()
        for col in X.select_dtypes(include=['category']).columns:
            X_processed[col] = X_processed[col].cat.codes
        
        snapshots = []
        
        for n_trees in self.snapshot_intervals:
            self.logger.info(f"Training snapshot with {n_trees} trees")
            
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=n_trees,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                enable_categorical=True
            )
            
            model.fit(X_processed, y_encoded)
            snapshots.append(model)
        
        self.snapshots = snapshots
        return snapshots
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions from snapshots"""
        if not self.snapshots:
            raise ValueError("No snapshots available. Train model first.")
        
        X_processed = X.copy()
        for col in X.select_dtypes(include=['category']).columns:
            X_processed[col] = X_processed[col].cat.codes
        
        predictions = []
        for model in self.snapshots:
            pred = model.predict_proba(X_processed)
            predictions.append(pred)
        
        # Soft voting
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred 