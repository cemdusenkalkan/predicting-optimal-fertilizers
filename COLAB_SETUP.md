# Google Colab Setup Guide

## 🚀 Quick Start for Colab

### Step 1: Clone Repository
```bash
!git clone https://github.com/cemdusenkalkan/predicting-optimal-fertilizers.git
%cd predicting-optimal-fertilizers
```

### Step 2: Upload Datasets
Upload your competition datasets to a `datasets/` folder:
- `train.csv`
- `test.csv`
- `sample_submission.csv` (optional)

### Step 3: Setup Environment
```python
!python setup_colab.py
```

### Step 4: Run Pipeline
```python
!python run_colab.py
```

## 📋 Alternative Commands

### Test All Modules
```bash
!chmod +x run_all.sh
!./run_all.sh
```

### Run Specific Pipeline
```python
# Meta-stacking only (recommended for Colab)
!python run_competition.py --mode meta-stacking

# Evaluate model performance
!python run_competition.py --mode evaluate
```

### Test Individual Modules
```python
!python test_data_loader.py
!python test_feature_engineering.py
!python test_meta_stacking.py
```

## 🔧 Manual Setup (if needed)

### Install Dependencies
```python
!pip install xgboost>=2.0.0 lightgbm>=4.0.0 catboost>=1.2.0 scipy>=1.10.0 optuna>=3.4.0
```

### Create Directories
```python
import os
for dir_name in ['logs', 'models', 'submissions', 'data/processed', 'experiments']:
    os.makedirs(dir_name, exist_ok=True)
```

## 📊 Expected Output

After running the pipeline, you should see:
```
🏆 Starting Competitive Pipeline (Target: 0.36+ MAP@3)
📊 Step 1: Loading Data
✅ Data loaded: train (750000, 8), test (250000, 8)

🔧 Step 2: Agricultural Feature Engineering
✅ Agricultural features: 52 features

📈 Step 3: Agricultural Data Augmentation
✅ Data augmentation: 750000 -> 2252750 samples

🚀 Step 4: Training Competitive Model
✅ Meta-stacking training completed

📋 Step 5: Generating Submission
✅ Submission saved: submissions/submission.csv
```

## 🎯 Performance Targets

- **Baseline**: 0.32-0.33 MAP@3
- **With proven techniques**: 0.34-0.35 MAP@3
- **Target competitive**: 0.36+ MAP@3

## 🔍 Troubleshooting

### Memory Issues
If you encounter memory issues in Colab:
1. Use smaller data samples in test scripts
2. Reduce CV folds: `n_folds=5` instead of `n_folds=15`
3. Use meta-stacking mode only (avoid ranking models)

### Import Errors
```python
# Check if all modules import correctly
from src.predictor import CompetitiveFertilizerPredictor
from src.config import config
print("✅ Imports successful")
```

### Dataset Issues
Ensure your datasets are in the correct format:
- CSV files with proper headers
- Train set should have 'Fertilizer Name' column
- Test set should have 'ID' column

## 📁 File Structure

```
predicting-optimal-fertilizers/
├── datasets/                 # Upload your data here
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/                      # Core modules
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── meta_stacking.py
│   ├── predictor.py
│   └── config.py
├── setup_colab.py           # Colab setup script
├── run_colab.py             # Colab-optimized runner
├── run_competition.py       # Main competition script
├── run_all.sh              # Test all modules
├── test_*.py               # Individual module tests
└── submissions/            # Generated submissions
```

## 🏆 Competitive Techniques Used

1. **Agricultural Domain Features**: NPK chemistry, crop-soil compatibility
2. **Categorical Treatment**: ALL features treated as categorical (+0.006 MAP@3)
3. **Meta-Stacking**: XGBoost + LightGBM + CatBoost ensemble
4. **Data Augmentation**: NPK jitter within agricultural constraints
5. **High-Fold CV**: 15-fold stratified cross-validation for stability

## 🎉 Success Indicators

You've successfully set up the pipeline when you see:
- ✅ All modules import without errors
- ✅ Data loads correctly with proper shapes
- ✅ Feature engineering creates 52+ features
- ✅ Meta-stacking completes training
- ✅ Submission file is generated in correct format

The final `submission.csv` will be ready for competition upload! 