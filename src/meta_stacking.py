import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import logging
from typing import Dict, List, Tuple, Any
from .config import config
from .ranking_models import map_at_k

class MetaStackingEnsemble:
    """
    Meta-stacking with XGBoost over diverse base learners
    """
    
    def __init__(self, cfg=None, n_folds: int = 15):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        self.n_folds = n_folds  # High-fold stratified CV
        self.base_models = {}
        self.meta_model = None
        self.label_encoder = LabelEncoder()
        
    def create_base_models(self) -> Dict[str, Any]:
        """Create diverse base learners"""
        models = {
            'xgb': xgb.XGBClassifier(
                objective='multi:softprob',
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
            ),
            
            'lgb': lgb.LGBMClassifier(
                objective='multiclass',
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
            ),
            
            'cat': cb.CatBoostClassifier(
                iterations=1000,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                thread_count=-1
            )
        }
        
        return models
    
    def create_meta_model(self) -> xgb.XGBClassifier:
        """Create XGBoost meta-learner"""
        return xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
    
    def generate_meta_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Generate meta-features using high-fold cross-validation"""
        self.logger.info(f"Generating meta-features with {self.n_folds}-fold CV")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(np.unique(y_encoded))
        
        # Create base models
        base_models = self.create_base_models()
        
        # High-fold stratified CV
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(base_models) * n_classes))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            self.logger.info(f"Fold {fold + 1}/{self.n_folds}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y_encoded[train_idx]
            X_val_fold = X.iloc[val_idx]
            
            col_start = 0
            for model_name, model in base_models.items():
                # Prepare data for model
                X_train_prep = X_train_fold.copy()
                X_val_prep = X_val_fold.copy()
                
                # Handle categorical features
                if model_name == 'cat':
                    # CatBoost handles categoricals and objects natively
                    cat_features = X_train_prep.select_dtypes(include=['category', 'object']).columns.tolist()
                else:
                    # Other models need encoded categoricals and objects
                    for col in X_train_prep.select_dtypes(include=['category', 'object']).columns:
                        if X_train_prep[col].dtype.name == 'category':
                            X_train_prep[col] = X_train_prep[col].cat.codes
                            X_val_prep[col] = X_val_prep[col].cat.codes
                        else:
                            # Convert object to category then to codes
                            X_train_prep[col] = pd.Categorical(X_train_prep[col]).codes
                            X_val_prep[col] = pd.Categorical(X_val_prep[col]).codes
                
                # Train model
                if model_name == 'cat' and len(cat_features) > 0:
                    model.fit(X_train_prep, y_train_fold, cat_features=cat_features)
                else:
                    model.fit(X_train_prep, y_train_fold)
                
                # Generate predictions
                y_pred_proba = model.predict_proba(X_val_prep)
                
                # Store meta-features (out-of-fold predictions only)
                col_end = col_start + n_classes
                meta_features[val_idx, col_start:col_end] = y_pred_proba
                col_start = col_end
        
        # Store trained base models for final training
        self.base_models = base_models
        
        return meta_features
    
    def train_meta_model(self, meta_features: np.ndarray, y: pd.Series) -> xgb.XGBClassifier:
        """Train meta-model on meta-features"""
        self.logger.info("Training XGBoost meta-learner")
        
        y_encoded = self.label_encoder.transform(y)
        
        # Create and train meta-model
        meta_model = self.create_meta_model()
        meta_model.fit(meta_features, y_encoded)
        
        # Evaluate meta-model
        meta_pred = meta_model.predict_proba(meta_features)
        meta_score = map_at_k(y_encoded, meta_pred, k=3)
        self.logger.info(f"Meta-model MAP@3: {meta_score:.6f}")
        
        self.meta_model = meta_model
        return meta_model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MetaStackingEnsemble':
        """Fit the complete meta-stacking ensemble"""
        self.logger.info("Training meta-stacking ensemble")
        
        # Generate meta-features
        meta_features = self.generate_meta_features(X, y)
        
        # Train meta-model
        self.train_meta_model(meta_features, y)
        
        # Retrain base models on full data
        y_encoded = self.label_encoder.transform(y)
        
        for model_name, model in self.base_models.items():
            self.logger.info(f"Retraining {model_name} on full data")
            
            X_prep = X.copy()
            
            if model_name == 'cat':
                cat_features = X_prep.select_dtypes(include=['category', 'object']).columns.tolist()
            else:
                for col in X_prep.select_dtypes(include=['category', 'object']).columns:
                    if X_prep[col].dtype.name == 'category':
                        X_prep[col] = X_prep[col].cat.codes
                    else:
                        X_prep[col] = pd.Categorical(X_prep[col]).codes
            
            if model_name == 'cat' and len(cat_features) > 0:
                model.fit(X_prep, y_encoded, cat_features=cat_features)
            else:
                model.fit(X_prep, y_encoded)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using meta-stacking"""
        if self.meta_model is None:
            raise ValueError("Model not trained. Call fit first.")
        
        n_classes = len(self.label_encoder.classes_)
        meta_features = np.zeros((len(X), len(self.base_models) * n_classes))
        
        col_start = 0
        for model_name, model in self.base_models.items():
            X_prep = X.copy()
            
            if model_name != 'cat':
                for col in X_prep.select_dtypes(include=['category', 'object']).columns:
                    if X_prep[col].dtype.name == 'category':
                        X_prep[col] = X_prep[col].cat.codes
                    else:
                        X_prep[col] = pd.Categorical(X_prep[col]).codes
            
            y_pred_proba = model.predict_proba(X_prep)
            
            col_end = col_start + n_classes
            meta_features[:, col_start:col_end] = y_pred_proba
            col_start = col_end
        
        # Meta-model prediction
        final_proba = self.meta_model.predict_proba(meta_features)
        
        return final_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions"""
        proba = self.predict_proba(X)
        return self.label_encoder.inverse_transform(np.argmax(proba, axis=1))

class DataAugmentation:
    """
    Agricultural-specific data augmentation
    """
    
    def __init__(self, expansion_factor: int = 3):
        self.expansion_factor = expansion_factor
        self.logger = logging.getLogger(__name__)
    
    def expand_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Expand training data by repeating rows with slight NPK jitter"""
        self.logger.info(f"Expanding training data by {self.expansion_factor}x")
        
        X_expanded = []
        y_expanded = []
        
        # Original data
        X_expanded.append(X.copy())
        y_expanded.append(y.copy())
        
        # Generate expanded copies
        for expansion in range(self.expansion_factor - 1):
            X_copy = X.copy()
            
            # Apply minimal jitter to NPK values (±1%)
            npk_columns = [col for col in X.columns if any(nutrient in col.lower() 
                          for nutrient in ['nitrogen', 'phosphorous', 'potassium', 'npk'])]
            
            for col in npk_columns:
                if X_copy[col].dtype in ['float64', 'int64']:
                    # Add small Gaussian noise
                    noise = np.random.normal(0, 0.01, size=len(X_copy))
                    X_copy[col] = X_copy[col] * (1 + noise)
                    
                    # Ensure non-negative values
                    X_copy[col] = np.maximum(X_copy[col], 0)
            
            X_expanded.append(X_copy)
            y_expanded.append(y.copy())
        
        # Combine all expansions
        X_final = pd.concat(X_expanded, ignore_index=True)
        y_final = pd.concat(y_expanded, ignore_index=True)
        
        self.logger.info(f"Expanded dataset: {len(X)} -> {len(X_final)} samples")
        
        return X_final, y_final
    
    def apply_mixup(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.2) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply mixup augmentation to numerical features within crop-soil strata"""
        self.logger.info(f"Applying mixup augmentation with alpha={alpha}")
        
        # Identify crop and soil columns
        crop_col = None
        soil_col = None
        for col in X.columns:
            if 'crop' in col.lower():
                crop_col = col
            elif 'soil' in col.lower():
                soil_col = col
        
        if crop_col is None or soil_col is None:
            self.logger.warning("Cannot find crop/soil columns for stratified mixup")
            return X, y
        
        # Numerical columns for mixup
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            return X, y
        
        X_augmented = []
        y_augmented = []
        
        # Original data
        X_augmented.append(X.copy())
        y_augmented.append(y.copy())
        
        # Create mixup samples within each crop-soil combination
        for (crop_val, soil_val), group in X.groupby([crop_col, soil_col]):
            if len(group) < 2:
                continue
                
            group_indices = group.index.tolist()
            group_y = y.loc[group_indices]
            
            # Generate mixup samples
            n_mixup = min(len(group) // 2, 10)  # Limit mixup samples
            
            for _ in range(n_mixup):
                # Sample two rows from same stratum
                idx1, idx2 = np.random.choice(group_indices, 2, replace=False)
                
                # Mixup lambda
                lam = np.random.beta(alpha, alpha)
                
                # Create mixed sample
                x_mixed = X.loc[idx1].copy()
                for col in num_cols:
                    x_mixed[col] = lam * X.loc[idx1, col] + (1 - lam) * X.loc[idx2, col]
                
                # Mixed label (take from stronger component)
                y_mixed = y.loc[idx1] if lam > 0.5 else y.loc[idx2]
                
                X_augmented.append(pd.DataFrame([x_mixed]))
                y_augmented.append(pd.Series([y_mixed]))
        
        # Combine all augmentations
        X_final = pd.concat(X_augmented, ignore_index=True)
        y_final = pd.concat(y_augmented, ignore_index=True)
        
        self.logger.info(f"Mixup augmentation: {len(X)} -> {len(X_final)} samples")
        
        return X_final, y_final 