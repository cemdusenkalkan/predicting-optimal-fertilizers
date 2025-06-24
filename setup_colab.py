#!/usr/bin/env python3
"""
Google Colab Setup Script for Competitive Fertilizer Prediction

Run this first in Colab to set up the environment.
"""

import os
import sys
import subprocess

def install_packages():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0", 
        "catboost>=1.2.0",
        "scipy>=1.10.0",
        "optuna>=3.4.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed")

def setup_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "logs",
        "models", 
        "submissions",
        "data/processed",
        "experiments"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created: {dir_name}")
    
    print("✅ Directories created")

def download_datasets():
    """Download datasets if needed"""
    print("📊 Checking datasets...")
    
    if not os.path.exists("datasets"):
        print("⚠️  No datasets directory found.")
        print("Please upload your datasets to a 'datasets' folder with:")
        print("  - train.csv")
        print("  - test.csv") 
        print("  - sample_submission.csv")
        return False
    
    required_files = ["train.csv", "test.csv"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(f"datasets/{file}"):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required datasets found")
    return True

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.data_loader import DataLoader
        from src.feature_engineering import FeatureEngineer
        from src.meta_stacking import MetaStackingEnsemble
        from src.predictor import CompetitiveFertilizerPredictor
        from src.config import config
        
        print("✅ All modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Competitive Fertilizer Prediction for Colab")
    print("=" * 60)
    
    # Install packages
    install_packages()
    
    # Setup directories
    setup_directories()
    
    # Check datasets
    datasets_ok = download_datasets()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 60)
    if datasets_ok and imports_ok:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python run_colab.py")
        print("2. Or: python run_competition.py --mode meta-stacking")
        print("3. Or: ./run_all.sh (to test all modules)")
    else:
        print("⚠️  Setup completed with warnings.")
        if not datasets_ok:
            print("   - Please upload datasets")
        if not imports_ok:
            print("   - Check import errors above")

if __name__ == "__main__":
    main() 