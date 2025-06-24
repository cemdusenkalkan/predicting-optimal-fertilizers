"""
Fertilizer Competition - AutoGluon + Hill Climbing Pipeline

A competitive machine learning pipeline for fertilizer recommendation
using AutoGluon ensemble methods and hill climbing optimization.
"""

__version__ = "1.0.0"
__author__ = "Competition Team"

from .config import config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .hill_climbing import HillClimbingOptimizer
from .autogluon_trainer import AutoGluonTrainer
from .predictor import FertilizerPredictor

__all__ = [
    'config',
    'DataLoader',
    'FeatureEngineer', 
    'HillClimbingOptimizer',
    'AutoGluonTrainer',
    'FertilizerPredictor'
] 