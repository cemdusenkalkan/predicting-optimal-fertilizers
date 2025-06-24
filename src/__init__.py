"""
Competitive Fertilizer Prediction Package

Implements proven techniques for agricultural fertilizer recommendation
targeting 0.36+ MAP@3 performance.
"""

from .config import config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .ranking_models import RankingOptimizedModels, ProbabilityCalibration, SnapshotEnsemble
from .meta_stacking import MetaStackingEnsemble, DataAugmentation
from .predictor import CompetitiveFertilizerPredictor

__version__ = "2.0.0"
__author__ = "Competition Team"

# Make primary classes available at package level
__all__ = [
    'config',
    'DataLoader',
    'FeatureEngineer',
    'RankingOptimizedModels',
    'ProbabilityCalibration',
    'SnapshotEnsemble',
    'MetaStackingEnsemble',
    'DataAugmentation',
    'CompetitiveFertilizerPredictor'
] 