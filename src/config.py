import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import yaml

@dataclass
class DataConfig:
    """Data paths and preprocessing configuration"""
    train_path: str = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path: str = "/kaggle/input/playground-series-s5e6/test.csv"
    original_path: str = "/kaggle/input/fertilizer-recommendation/Fertilizer_Prediction.csv"
    
    # Data expansion strategy (PROVEN TECHNIQUE)
    training_multiplier: int = 3  # Forum proven: 3x expansion
    original_multiplier: int = 2  # Forum proven: 2x original data
    
    # Column name fixes
    fix_temperature_typo: bool = True  # 'Temparature' -> 'Temperature'

@dataclass 
class FeatureConfig:
    """Feature engineering configuration - ALL PROVEN TECHNIQUES"""
    
    # CRITICAL: Categorical treatment (+0.006 improvement)
    categorical_binning_strategy: str = "quantile"  # quantile vs equal-width
    categorical_bins: int = 20
    treat_all_as_categorical: bool = True
    
    # PROVEN: Constant feature (+0.005 improvement)
    add_constant_feature: bool = True
    
    # PROVEN: Environmental features
    add_env_max: bool = True  # max(temp, humidity, moisture)
    add_temp_humidity_index: bool = True
    add_climate_comfort: bool = True
    
    # CRITICAL: NPK ratios (hidden signal)
    add_npk_ratios: bool = True
    add_npk_balance: bool = True
    add_total_npk: bool = True
    npk_ratio_epsilon: float = 1e-8
    
    # PROVEN: Temperature suitability
    add_temp_suitability: bool = True
    crop_temp_ranges: Dict[str, Tuple[int, int]] = None
    
    # HIGH IMPORTANCE: Crop-Soil interactions
    add_crop_soil_combo: bool = True
    add_target_encoding: bool = True
    target_encoding_folds: int = 5
    
    # Advanced features for hill climbing
    add_polynomial_features: bool = False  # Added in iterations
    add_domain_features: bool = False      # Added in iterations
    
    def __post_init__(self):
        if self.crop_temp_ranges is None:
            self.crop_temp_ranges = {
                'Sugarcane': (26, 35), 'Maize': (25, 32), 'Wheat': (20, 30),
                'Paddy': (25, 35), 'Cotton': (25, 35), 'Tobacco': (20, 30),
                'Barley': (15, 25), 'Millets': (25, 35), 'Pulses': (20, 30),
                'Oil seeds': (20, 30), 'Ground Nuts': (25, 32)
            }

@dataclass
class ModelConfig:
    """AutoGluon and ensemble configuration"""
    
    # AutoGluon settings
    predictor_label: str = "target"
    eval_metric: str = "accuracy"  # AutoGluon doesn't have MAP@3
    verbosity: int = 1
    
    # Training configuration
    time_limit_per_fold: int = 180  # 3 minutes per CV fold
    final_time_limit: int = 600     # 10 minutes for final model
    presets: str = "best_quality"
    auto_stack: bool = True
    num_bag_folds: int = 3          # CV folds: 3 for speed, 5 for final
    num_stack_levels: int = 1       # Stacking levels: 1 for speed, 2 for final
    
    # Model hyperparameters (PROVEN ENSEMBLE)
    hyperparameters: Dict = None
    excluded_model_types: List[str] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {
                'GBM': [
                    {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                    {},  # Default XGBoost
                    {'boosting': 'dart', 'ag_args': {'name_suffix': 'DART'}},
                    {'boosting': 'goss', 'ag_args': {'name_suffix': 'GOSS'}},
                ],
                'CAT': {},  # CatBoost - excellent for categoricals
                'NN_TORCH': [{'num_epochs': 50, 'learning_rate': 0.01}],
                'FASTAI': {},
                'RF': [{'n_estimators': 300}],
            }
        
        if self.excluded_model_types is None:
            self.excluded_model_types = ['KNN', 'LR']  # Too slow/simple

@dataclass
class HillClimbingConfig:
    """Hill climbing optimization configuration"""
    max_iterations: int = 3
    cv_folds: int = 3  # Speed vs accuracy tradeoff
    improvement_threshold: float = 0.001  # Minimum improvement to keep features
    memory_limit_gb: float = 8.0
    random_state: int = 42
    
    # Feature generation strategy per iteration
    iteration_features: Dict[int, List[str]] = None
    
    def __post_init__(self):
        if self.iteration_features is None:
            self.iteration_features = {
                0: ["polynomial_npk", "environmental_polynomials"],
                1: ["advanced_crop_suitability", "fertilizer_effectiveness"],
                2: ["harmonic_means", "interaction_terms"]
            }

@dataclass
class CompetitionConfig:
    """Main competition configuration"""
    target_score: float = 0.38  # Beat champion 0.383
    baseline_score: float = 0.33
    random_state: int = 42
    
    # Paths
    output_dir: str = "experiments"
    models_dir: str = "models"
    logs_dir: str = "logs"
    data_dir: str = "data"
    submissions_dir: str = "data/submissions"
    
    # Competition specific
    num_classes: int = 7
    target_column: str = "Fertilizer Name"
    id_column: str = "id"
    
    # Submission format
    submission_format: str = "space_separated"  # TOP-3 space-separated
    
class Config:
    """Master configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.models = ModelConfig()
        self.hill_climbing = HillClimbingConfig()
        self.competition = CompetitionConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations from YAML
        for section, values in config_dict.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def save_to_yaml(self, config_path: str):
        """Save current configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'models': self.models.__dict__,
            'hill_climbing': self.hill_climbing.__dict__,
            'competition': self.competition.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs_to_create = [
            self.competition.output_dir,
            self.competition.models_dir,
            self.competition.logs_dir,
            self.competition.data_dir,
            self.competition.submissions_dir,
            f"{self.competition.output_dir}/iteration_0",
            f"{self.competition.output_dir}/iteration_1", 
            f"{self.competition.output_dir}/iteration_2",
            f"{self.competition.output_dir}/best_experiment",
            f"{self.competition.data_dir}/processed"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_iteration_path(self, iteration: int) -> str:
        """Get path for specific iteration"""
        return f"{self.competition.output_dir}/iteration_{iteration}"
    
    def get_best_experiment_path(self) -> str:
        """Get path for best experiment"""
        return f"{self.competition.output_dir}/best_experiment"

# Global configuration instance
config = Config() 