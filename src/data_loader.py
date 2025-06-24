import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
import logging
from .config import config

class DataLoader:
    """
    Data loading and preprocessing with competitive strategy techniques
    
    Implements:
    - Data expansion (3x training + 2x original)
    - Column name fixes (Temparature -> Temperature)
    - Original dataset integration
    - Proper validation splits
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.logger = logging.getLogger(__name__)
        
    def load_competition_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load main competition train and test data"""
        try:
            train_df = pd.read_csv(self.cfg.data.train_path)
            test_df = pd.read_csv(self.cfg.data.test_path)
            
            self.logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
            
            # Fix column name typo if present
            if self.cfg.data.fix_temperature_typo:
                train_df = self._fix_temperature_typo(train_df)
                test_df = self._fix_temperature_typo(test_df)
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading competition data: {e}")
            raise
    
    def load_original_data(self) -> Optional[pd.DataFrame]:
        """Load original 100-sample dataset if available"""
        try:
            if os.path.exists(self.cfg.data.original_path):
                original_df = pd.read_csv(self.cfg.data.original_path)
                
                # Fix column names
                if self.cfg.data.fix_temperature_typo:
                    original_df = self._fix_temperature_typo(original_df)
                
                self.logger.info(f"Loaded original dataset: {original_df.shape}")
                return original_df
            else:
                self.logger.warning("Original dataset not found - continuing without it")
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not load original dataset: {e}")
            return None
    
    def _fix_temperature_typo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix 'Temparature' -> 'Temperature' typo"""
        if 'Temparature' in df.columns:
            df = df.rename(columns={'Temparature': 'Temperature'})
            self.logger.debug("Fixed temperature column name typo")
        return df
    
    def prepare_features_and_target(self, train_df: pd.DataFrame, 
                                   original_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features and target from datasets
        
        Returns:
            X_train, y_train, X_orig, y_orig
        """
        # Competition data
        X_train = train_df.drop([self.cfg.competition.id_column, self.cfg.competition.target_column], axis=1)
        y_train = train_df[self.cfg.competition.target_column]
        
        # Original data (if available)
        X_orig, y_orig = None, None
        if original_df is not None:
            # Check if target column exists and filter known classes
            if self.cfg.competition.target_column in original_df.columns:
                known_fertilizers = set(y_train.unique())
                mask = original_df[self.cfg.competition.target_column].isin(known_fertilizers)
                
                if mask.sum() > 0:
                    filtered_original = original_df[mask]
                    X_orig = filtered_original.drop([self.cfg.competition.target_column], axis=1)
                    y_orig = filtered_original[self.cfg.competition.target_column]
                    
                    self.logger.info(f"Original dataset filtered: {len(filtered_original)} valid samples")
                else:
                    self.logger.warning("No overlapping fertilizer types in original dataset")
            else:
                self.logger.warning("Target column not found in original dataset")
        
        return X_train, y_train, X_orig, y_orig
    
    def expand_training_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_orig: Optional[pd.DataFrame] = None, 
                           y_orig: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        PROVEN TECHNIQUE: Expand training data for performance boost
        
        Strategy from forum:
        - 3x multiplication of competition training data
        - 2x multiplication of original data (if available)
        """
        # 3x expansion of competition data
        X_expanded = pd.concat([X_train] * self.cfg.data.training_multiplier, ignore_index=True)
        y_expanded = pd.concat([y_train] * self.cfg.data.training_multiplier, ignore_index=True)
        
        self.logger.info(f"Training data expanded: {X_train.shape} -> {X_expanded.shape}")
        
        # Add original data with 2x multiplication
        if X_orig is not None and y_orig is not None:
            # Ensure same features
            common_features = [col for col in X_orig.columns if col in X_expanded.columns]
            X_orig_subset = X_orig[common_features]
            
            # 2x multiplication of original data
            X_orig_2x = pd.concat([X_orig_subset] * self.cfg.data.original_multiplier, ignore_index=True)
            y_orig_2x = pd.concat([y_orig] * self.cfg.data.original_multiplier, ignore_index=True)
            
            # Combine
            X_expanded = pd.concat([X_expanded, X_orig_2x], ignore_index=True)
            y_expanded = pd.concat([y_expanded, y_orig_2x], ignore_index=True)
            
            self.logger.info(f"Added original data: final shape {X_expanded.shape}")
        
        return X_expanded, y_expanded
    
    def validate_data_quality(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate data quality and consistency"""
        try:
            # Check for missing values
            train_missing = train_df.isnull().sum()
            test_missing = test_df.isnull().sum()
            
            if train_missing.sum() > 0:
                self.logger.warning(f"Missing values in train: {train_missing[train_missing > 0].to_dict()}")
            
            if test_missing.sum() > 0:
                self.logger.warning(f"Missing values in test: {test_missing[test_missing > 0].to_dict()}")
            
            # Check feature consistency
            train_features = set(train_df.columns) - {self.cfg.competition.id_column, self.cfg.competition.target_column}
            test_features = set(test_df.columns) - {self.cfg.competition.id_column}
            
            if train_features != test_features:
                missing_in_test = train_features - test_features
                extra_in_test = test_features - train_features
                
                if missing_in_test:
                    self.logger.error(f"Features missing in test: {missing_in_test}")
                if extra_in_test:
                    self.logger.warning(f"Extra features in test: {extra_in_test}")
                
                return False
            
            # Check target distribution
            target_dist = train_df[self.cfg.competition.target_column].value_counts()
            self.logger.info(f"Target distribution:\n{target_dist}")
            
            # Check if balanced (should be for synthetic data)
            if len(target_dist.unique()) == 1:
                self.logger.info("✅ Perfectly balanced dataset (synthetic characteristic)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def load_all(self) -> dict:
        """
        Load all data and return structured dictionary
        
        Returns:
            {
                'train': train_df,
                'test': test_df, 
                'original': original_df,
                'X_train': X_train,
                'y_train': y_train,
                'X_orig': X_orig,
                'y_orig': y_orig,
                'X_test': X_test
            }
        """
        self.logger.info("🔄 Loading all datasets...")
        
        # Load main competition data
        train_df, test_df = self.load_competition_data()
        
        # Load original data
        original_df = self.load_original_data()
        
        # Validate data quality
        if not self.validate_data_quality(train_df, test_df):
            self.logger.warning("Data quality issues detected")
        
        # Prepare features and targets
        X_train, y_train, X_orig, y_orig = self.prepare_features_and_target(train_df, original_df)
        
        # Test features
        X_test = test_df.drop([self.cfg.competition.id_column], axis=1)
        
        data = {
            'train': train_df,
            'test': test_df,
            'original': original_df,
            'X_train': X_train,
            'y_train': y_train,
            'X_orig': X_orig,
            'y_orig': y_orig,
            'X_test': X_test
        }
        
        self.logger.info("✅ All data loaded successfully")
        return data 