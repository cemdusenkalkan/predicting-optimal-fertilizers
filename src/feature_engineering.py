import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Optional
import logging
from .config import config

class FeatureEngineer:
    """
    Comprehensive feature engineering with ALL proven competitive techniques
    
    Implements:
    - Categorical treatment of ALL features (+0.006 improvement)
    - Constant feature (+0.005 improvement) 
    - NPK ratios (hidden signal)
    - Environmental features (env_max, temp_suitability)
    - Target encoding with proper CV (no leakage)
    - Hill climbing iterative features
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all base proven features (52 features total)
        
        This includes ALL forum-proven techniques for maximum competitive advantage
        """
        df = df.copy()
        self.logger.info(f"Creating base features from {df.shape[1]} columns...")
        
        # 1. CRITICAL: Categorical versions of ALL numerical features (+0.006)
        if self.cfg.features.treat_all_as_categorical:
            df = self._create_categorical_features(df)
        
        # 2. PROVEN: Constant feature (+0.005)
        if self.cfg.features.add_constant_feature:
            df['const'] = 1
            
        # 3. PROVEN: Environmental features
        if self.cfg.features.add_env_max:
            df = self._create_environmental_features(df)
            
        # 4. CRITICAL: NPK ratios (hidden signal)
        if self.cfg.features.add_npk_ratios:
            df = self._create_npk_features(df)
            
        # 5. PROVEN: Temperature suitability
        if self.cfg.features.add_temp_suitability:
            df = self._create_temperature_suitability(df)
            
        # 6. HIGH IMPORTANCE: Crop-Soil interactions
        if self.cfg.features.add_crop_soil_combo:
            df = self._create_crop_soil_features(df)
            
        # 7. Ensure proper categorical dtypes for AutoGluon
        df = self._ensure_categorical_dtypes(df)
        
        self.logger.info(f"Base features created: {df.shape[1]} total features")
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """PROVEN TECHNIQUE: Convert ALL numerical features to categorical (+0.006)"""
        numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
        
        for col in numerical_cols:
            if col in df.columns:
                # CRITICAL: Use quantile-based binning (not equal-width)
                if self.cfg.features.categorical_binning_strategy == "quantile":
                    df[f'{col}_cat'] = pd.qcut(df[col], q=self.cfg.features.categorical_bins, 
                                             labels=False, duplicates='drop')
                else:
                    df[f'{col}_cat'] = pd.cut(df[col], bins=self.cfg.features.categorical_bins, 
                                            labels=False)
                
                # Threshold patterns
                df[f'{col}_high'] = (df[col] > df[col].median()).astype(int)
        
        # Make ALL other numerical features categorical too
        other_numerical = df.select_dtypes(include=[np.number]).columns
        for col in other_numerical:
            if col not in [f'{c}_cat' for c in numerical_cols] and col not in ['const']:
                try:
                    df[f'{col}_cat'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
                except:
                    df[f'{col}_cat'] = pd.cut(df[col], bins=10, labels=False)
        
        return df
    
    def _create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """PROVEN: Environmental features from forum intelligence"""
        if all(col in df.columns for col in ['Temperature', 'Humidity', 'Moisture']):
            # PROVEN: env_max feature
            df['env_max'] = df[['Temperature', 'Humidity', 'Moisture']].max(axis=1)
            
            if self.cfg.features.add_temp_humidity_index:
                df['temp_humidity_index'] = df['Temperature'] * df['Humidity'] / 100
                
            if self.cfg.features.add_climate_comfort:
                df['climate_comfort'] = (df['Temperature'] + df['Humidity'] + df['Moisture']) / 3
        
        # Environmental zone categorization
        for env_col in ['Temperature', 'Humidity', 'Moisture']:
            if env_col in df.columns:
                df[f'{env_col.lower()}_zone'] = pd.qcut(df[env_col], q=5, labels=[0,1,2,3,4], duplicates='drop')
                
                # Stress indicators
                if env_col == 'Temperature':
                    df['temp_stress'] = ((df[env_col] < 20) | (df[env_col] > 35)).astype(int)
                elif env_col == 'Humidity':
                    df['humidity_stress'] = ((df[env_col] < 40) | (df[env_col] > 80)).astype(int)
                elif env_col == 'Moisture':
                    df['moisture_stress'] = ((df[env_col] < 30) | (df[env_col] > 70)).astype(int)
        
        return df
    
    def _create_npk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRITICAL: NPK ratios - the hidden signal from forum intelligence"""
        epsilon = self.cfg.features.npk_ratio_epsilon
        npk_cols = ['Nitrogen', 'Phosphorous', 'Potassium']
        
        if all(col in df.columns for col in npk_cols):
            # Basic ratios (PROVEN)
            df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
            df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)  
            df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
            
            # Total and balance
            if self.cfg.features.add_total_npk:
                df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
                
            if self.cfg.features.add_npk_balance:
                df['NPK_balance'] = df[npk_cols].std(axis=1)
            
            # Fertilizer-specific chemistry scoring (CRITICAL for synthetic data)
            df['NPK_17_17_17_score'] = 1 / (1 + np.abs(df['N_P_ratio'] - 1) + np.abs(df['N_K_ratio'] - 1))
            df['NPK_28_28_score'] = 1 / (1 + np.abs(df['N_P_ratio'] - 1))
            df['NPK_10_26_26_score'] = 1 / (1 + np.abs(df['N_P_ratio'] - 0.38) + np.abs(df['P_K_ratio'] - 1))
            df['NPK_20_20_score'] = 1 / (1 + np.abs(df['N_P_ratio'] - 1))
            df['NPK_14_35_14_score'] = 1 / (1 + np.abs(df['N_P_ratio'] - 0.4) + np.abs(df['N_K_ratio'] - 1))
            df['DAP_score'] = 1 / (1 + np.abs(df['N_P_ratio'] - 0.78))  # DAP is ~18-46
            df['Urea_score'] = df['Nitrogen'] / (df['Total_NPK'] + epsilon)  # Urea is high N
            
            # Clip extreme ratios
            for col in ['N_P_ratio', 'N_K_ratio', 'P_K_ratio']:
                df[col] = np.clip(df[col], 0, 10)
        
        return df
    
    def _create_temperature_suitability(self, df: pd.DataFrame) -> pd.DataFrame:
        """PROVEN: Temperature suitability feature"""
        if 'Temperature' in df.columns and 'Crop Type' in df.columns:
            def get_temp_suitability(row):
                crop = row['Crop Type']
                temp = row['Temperature']
                if crop in self.cfg.features.crop_temp_ranges:
                    min_temp, max_temp = self.cfg.features.crop_temp_ranges[crop]
                    return 1 if min_temp <= temp <= max_temp else 0
                return 1 if 20 <= temp <= 32 else 0  # Default range
            
            df['temp_suitability'] = df.apply(get_temp_suitability, axis=1)
        
        return df
    
    def _create_crop_soil_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """HIGH IMPORTANCE: Crop-Soil interaction features"""
        if 'Crop Type' in df.columns and 'Soil Type' in df.columns:
            df['Crop_Soil_combo'] = df['Crop Type'].astype(str) + '_' + df['Soil Type'].astype(str)
            
            # Agricultural compatibility patterns
            crop_soil_strength = {
                'Maize_Loamy': 1.0, 'Sugarcane_Black': 1.0, 'Cotton_Black': 1.0,
                'Paddy_Clayey': 1.0, 'Wheat_Loamy': 1.0, 'Tobacco_Red': 1.0,
                'Maize_Black': 0.8, 'Sugarcane_Red': 0.8, 'Cotton_Red': 0.8,
                'Paddy_Loamy': 0.8, 'Wheat_Black': 0.8, 'Tobacco_Loamy': 0.8
            }
            df['crop_soil_strength'] = df['Crop_Soil_combo'].map(crop_soil_strength).fillna(0.5)
        
        return df
    
    def _ensure_categorical_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ALL categorical columns have proper dtypes for AutoGluon"""
        # Categorical feature columns
        categorical_cols = [col for col in df.columns if '_cat' in col or '_zone' in col]
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        
        # String columns
        string_cols = ['Crop Type', 'Soil Type', 'Crop_Soil_combo']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Binary features as categorical
        binary_cols = [col for col in df.columns if '_high' in col or '_stress' in col or '_suitability' in col]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def add_target_encoding(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, X_orig: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Add CV-based target encoding to prevent data leakage
        
        CRITICAL: Uses proper CV strategy to avoid leakage
        """
        if not self.cfg.features.add_target_encoding or 'Crop_Soil_combo' not in X_train.columns:
            return X_train, X_test, X_orig
        
        self.logger.info("Adding CV-based target encoding for Crop_Soil_combo...")
        
        # Convert to string for mapping
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # Create encoding map from full training data
        crop_soil_str = X_train['Crop_Soil_combo'].astype(str)
        y_encoded = pd.factorize(y_train)[0]  # Convert string labels to numeric
        
        encoding_map = crop_soil_str.groupby(crop_soil_str).apply(
            lambda x: pd.Series(y_encoded).iloc[x.index].mean()
        )
        
        # Apply CV-based encoding to training data
        X_train = self._add_target_encoding_cv(X_train, y_encoded, 'Crop_Soil_combo')
        
        # Apply same encoding map to test data (NO LEAKAGE)
        test_crop_soil_str = X_test['Crop_Soil_combo'].astype(str)
        X_test['Crop_Soil_combo_target_encoded'] = (
            test_crop_soil_str.map(encoding_map).fillna(np.mean(y_encoded))
        )
        
        # Handle original dataset
        if X_orig is not None:
            X_orig = X_orig.copy()
            orig_crop_soil_str = X_orig['Crop_Soil_combo'].astype(str)
            X_orig['Crop_Soil_combo_target_encoded'] = (
                orig_crop_soil_str.map(encoding_map).fillna(np.mean(y_encoded))
            )
        
        # Ensure Crop_Soil_combo is categorical
        X_train['Crop_Soil_combo'] = X_train['Crop_Soil_combo'].astype('category')
        X_test['Crop_Soil_combo'] = X_test['Crop_Soil_combo'].astype('category')
        if X_orig is not None:
            X_orig['Crop_Soil_combo'] = X_orig['Crop_Soil_combo'].astype('category')
        
        self.logger.info(f"Target encoding added: {X_train.shape[1]} features")
        return X_train, X_test, X_orig
    
    def _add_target_encoding_cv(self, X: pd.DataFrame, y: np.ndarray, feature_col: str) -> pd.DataFrame:
        """CV-based target encoding to prevent leakage"""
        X = X.copy()
        encoded_col = f'{feature_col}_target_encoded'
        X[encoded_col] = 0.0
        
        # Convert categorical to string
        if X[feature_col].dtype.name == 'category':
            X[feature_col] = X[feature_col].astype(str)
        
        # Stratified K-fold
        skf = StratifiedKFold(n_splits=self.cfg.features.target_encoding_folds, 
                             shuffle=True, random_state=self.cfg.competition.random_state)
        
        for train_idx, val_idx in skf.split(X, y):
            # Calculate encoding on training fold only
            encoding_map = X.iloc[train_idx].groupby(feature_col).apply(lambda x: y[x.index].mean())
            
            # Apply to validation fold
            val_encoded = X.loc[val_idx, feature_col].map(encoding_map)
            val_encoded = val_encoded.fillna(y.mean())
            X.loc[val_idx, encoded_col] = val_encoded
        
        return X
    
    def create_hill_climbing_features(self, df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """
        Generate additional features for hill climbing iterations
        
        Iteration 0: Polynomial interactions
        Iteration 1: Advanced domain features  
        Iteration 2: Harmonic means and interaction terms
        """
        df = df.copy()
        
        iteration_features = self.cfg.hill_climbing.iteration_features.get(iteration, [])
        
        if "polynomial_npk" in iteration_features:
            df = self._add_polynomial_npk_features(df)
            
        if "environmental_polynomials" in iteration_features:
            df = self._add_environmental_polynomials(df)
            
        if "advanced_crop_suitability" in iteration_features:
            df = self._add_advanced_crop_suitability(df)
            
        if "fertilizer_effectiveness" in iteration_features:
            df = self._add_fertilizer_effectiveness(df)
            
        if "harmonic_means" in iteration_features:
            df = self._add_harmonic_means(df)
            
        if "interaction_terms" in iteration_features:
            df = self._add_interaction_terms(df)
        
        self.logger.info(f"Hill climbing iteration {iteration}: {df.shape[1]} features")
        return df
    
    def _add_polynomial_npk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial NPK interactions"""
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
            df['NPK_polynomial'] = (df['Nitrogen'] ** 2) + (df['Phosphorous'] ** 2) + (df['Potassium'] ** 2)
            df['NPK_log_sum'] = np.log1p(df['Nitrogen'] + df['Phosphorous'] + df['Potassium'])
        return df
    
    def _add_environmental_polynomials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add environmental polynomial features"""
        if all(col in df.columns for col in ['Temperature', 'Humidity', 'Moisture']):
            df['env_polynomial'] = (df['Temperature'] ** 2) + (df['Humidity'] ** 2) + (df['Moisture'] ** 2)
            df['temp_humidity_interaction'] = df['Temperature'] * df['Humidity'] / 1000
        return df
    
    def _add_advanced_crop_suitability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced crop suitability scoring"""
        if 'Crop Type' in df.columns and 'Temperature' in df.columns:
            def advanced_crop_suitability(row):
                crop_optimal_temps = {
                    'Sugarcane': 30, 'Maize': 28, 'Wheat': 25, 
                    'Paddy': 30, 'Cotton': 30, 'Tobacco': 25
                }
                optimal = crop_optimal_temps.get(row['Crop Type'], 27)
                return 1 / (1 + abs(row['Temperature'] - optimal))
            
            df['advanced_crop_suitability'] = df.apply(advanced_crop_suitability, axis=1)
        return df
    
    def _add_fertilizer_effectiveness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fertilizer effectiveness based on NPK balance"""
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
            df['fertilizer_effectiveness'] = np.exp(-0.1 * df[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1))
        return df
    
    def _add_harmonic_means(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add harmonic means for NPK"""
        if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
            df['NPK_harmonic_mean'] = 3 / (1/np.maximum(df['Nitrogen'], 1) + 
                                          1/np.maximum(df['Phosphorous'], 1) + 
                                          1/np.maximum(df['Potassium'], 1))
        return df
    
    def _add_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction terms between key features"""
        if all(col in df.columns for col in ['env_max', 'Total_NPK']):
            df['env_npk_interaction'] = df['env_max'] * df['Total_NPK'] / 1000
        return df 