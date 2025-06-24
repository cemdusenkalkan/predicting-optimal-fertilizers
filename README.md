# 🏆 Fertilizer Competition - AutoGluon + Hill Climbing Pipeline

An ultra-competitive machine learning pipeline for fertilizer recommendation using AutoGluon ensemble methods and hill climbing optimization. This implementation incorporates ALL proven techniques from Kaggle forum intelligence to achieve maximum competitive performance.

## 🎯 Target Performance

- **Baseline**: 0.33 MAP@3
- **Target**: 0.38+ MAP@3 (beat champion score of 0.383)
- **Strategy**: Exploit synthetic data patterns with proven forum techniques

## 🔥 Key Competitive Techniques Implemented

### ✅ Forum-Proven Techniques (+0.011 total improvement)
1. **Categorical Treatment** (+0.006): Convert ALL numerical features to categorical bins
2. **Constant Feature** (+0.005): Add `const=1` column (universally improves CV)
3. **NPK Ratios** (Hidden Signal): N/P, N/K, P/K ratios encode fertilizer chemistry
4. **Environmental Features**: `env_max`, `temp_suitability`, climate indices
5. **Data Expansion**: 3x training data + 2x original dataset multiplication
6. **Target Encoding**: CV-based encoding for Crop-Soil combinations

### 🧗 Hill Climbing Optimization
- **Iterative Feature Generation**: 3 iterations of domain-specific features
- **Performance-Driven Selection**: Only keep features that improve CV score
- **Advanced Agricultural Features**: Polynomial NPK interactions, fertilizer effectiveness

### 🚀 AutoGluon Ensemble Power
- **Multi-Algorithm**: XGBoost, LightGBM, CatBoost, Neural Networks, Random Forest
- **Automated Stacking**: 2-level stacking with 5-fold bagging
- **Hyperparameter Optimization**: Advanced tuning across all algorithms

## 📁 Project Structure

```
fertilizer/
├── src/
│   ├── config.py              # Central configuration with all proven techniques
│   ├── data_loader.py         # Data loading and expansion (3x + 2x original)
│   ├── feature_engineering.py # All 52 proven features + hill climbing
│   ├── hill_climbing.py       # Iterative optimization engine
│   ├── autogluon_trainer.py   # Multi-algorithm ensemble trainer
│   ├── predictor.py           # Main pipeline orchestrator
│   └── __init__.py
├── experiments/               # Hill climbing iteration results
│   ├── iteration_0/
│   ├── iteration_1/
│   ├── iteration_2/
│   └── best_experiment/
├── models/                   # Trained AutoGluon models
├── logs/                     # Execution logs
├── data/
│   ├── processed/           # Processed datasets
│   └── submissions/         # Generated submissions
├── run_competition.py       # Main execution script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone/download the project
cd fertilizer

# Install dependencies
pip install -r requirements.txt

# Make sure you have the competition data:
# - /kaggle/input/playground-series-s5e6/train.csv
# - /kaggle/input/playground-series-s5e6/test.csv
# - /kaggle/input/fertilizer-recommendation/Fertilizer_Prediction.csv (optional)
```

### Usage

#### 1. Full Competition Pipeline (Recommended)
```bash
# Run complete pipeline with hill climbing optimization
python run_competition.py

# Skip hill climbing and use base proven features only
python run_competition.py --skip-hill-climbing
```

#### 2. Quick Prediction (Using Existing Model)
```bash
# Fast prediction using cached model and features
python run_competition.py --quick
```

#### 3. Model Evaluation Only
```bash
# Evaluate model performance without generating submission
python run_competition.py --evaluate
```

#### 4. Custom Configuration
```bash
# Use custom configuration file
python run_competition.py --config configs/custom_config.yaml
```

## 🔧 Configuration

All competitive techniques are configurable in `src/config.py`:

```python
# Proven forum techniques
FeatureConfig.treat_all_as_categorical = True     # +0.006 improvement
FeatureConfig.add_constant_feature = True        # +0.005 improvement
FeatureConfig.add_npk_ratios = True              # Hidden signal
FeatureConfig.add_env_max = True                 # Environmental features

# Data expansion strategy
DataConfig.training_multiplier = 3               # 3x training data
DataConfig.original_multiplier = 2               # 2x original data

# Hill climbing optimization
HillClimbingConfig.max_iterations = 3            # Feature optimization rounds
HillClimbingConfig.improvement_threshold = 0.001 # Minimum improvement to accept
```

## 📊 Expected Performance

### Feature Engineering Impact
- **Base Features (8)**: Baseline performance
- **Proven Features (52)**: +0.011 MAP@3 improvement
- **Hill Climbing (+10-20)**: Additional +0.003-0.007 improvement
- **Final Features (60-70)**: Maximum competitive performance

### Model Performance Progression
1. **Baseline AutoGluon**: ~0.33 MAP@3
2. **+ Proven Features**: ~0.34 MAP@3
3. **+ Hill Climbing**: ~0.35-0.36 MAP@3
4. **+ Final Ensemble**: **0.37-0.39+ MAP@3** 🎯

## 🏗️ Architecture Details

### Data Pipeline
```python
# 1. Load competition + original data
data = DataLoader().load_all()

# 2. Feature engineering (52 proven features)
features = FeatureEngineer().create_base_features(data)

# 3. Hill climbing optimization (3 iterations)
optimized = HillClimbingOptimizer().optimize(features)

# 4. AutoGluon ensemble training
model = AutoGluonTrainer().train_final_model(optimized)

# 5. Submission generation
submission = model.predict_submission(test_data)
```

### Feature Engineering Arsenal
- **Categorical Binning**: Quantile-based bins for all numerical features
- **NPK Chemistry**: Fertilizer-specific ratio scoring (17-17-17, 28-28, DAP, etc.)
- **Environmental Zones**: Temperature/humidity/moisture categorization
- **Agricultural Compatibility**: Crop-soil strength mapping
- **Target Encoding**: CV-based encoding for high-cardinality features

### AutoGluon Configuration
```python
hyperparameters = {
    'GBM': [XGBoost variants, LightGBM DART/GOSS],
    'CAT': [CatBoost with optimal depth/iterations],
    'NN_TORCH': [Deep neural networks],
    'RF': [Random Forest ensemble],
    'FASTAI': [TabNet architecture]
}
```

## 🔍 Monitoring & Analysis

### Logging
- Comprehensive logging to `logs/competition_TIMESTAMP.log`
- Performance tracking across hill climbing iterations
- Feature importance analysis
- Model leaderboard comparison

### Experiment Tracking
- Each hill climbing iteration saved to `experiments/iteration_X/`
- Best experiment cached for quick re-runs
- Feature importance and model metadata preserved

### Performance Metrics
- Primary: MAP@3 (Mean Average Precision at 3)
- Cross-validation: Stratified K-fold with proper data handling
- Overfitting prevention: Separate validation on pure competition data

## ⚠️ Important Notes

### Data Handling
- **Original Dataset**: Only used for training, never validation (prevents leakage)
- **Data Expansion**: Applied after CV splits to prevent overfitting
- **Target Encoding**: Uses proper CV strategy to avoid circular reference

### Computational Requirements
- **RAM**: 8GB+ recommended for full pipeline
- **CPU**: Multi-core beneficial for AutoGluon ensemble
- **GPU**: Optional but speeds up individual model training
- **Time**: 30-60 minutes for complete pipeline

### Submission Format
- **Format**: Space-separated top-3 predictions per row
- **Example**: `"28-28 DAP 20-20"`
- **Validation**: Automatic format checking before save

## 🎯 Competition Strategy

This pipeline implements a comprehensive strategy to beat the 0.383 champion score:

1. **Exploit Synthetic Data Patterns**: Use quantile-based categorical binning to match data generation logic
2. **Hidden NPK Signal**: Leverage fertilizer chemistry knowledge for ratio-based features
3. **AutoGluon Ensemble Power**: Multi-algorithm ensemble for maximum robustness
4. **Hill Climbing Optimization**: Iterative improvement beyond proven baselines
5. **Proper Validation**: Prevent overfitting while maximizing performance

## 📈 Results

Expected leaderboard performance:
- **Public LB**: 0.37-0.39 MAP@3
- **Private LB**: 0.36-0.38 MAP@3 (accounting for variance)
- **Ranking**: Top 10% (target: Top 5%)

## 🔧 Troubleshooting

### Common Issues
1. **AutoGluon Installation**: Use `pip install autogluon.tabular` (not full autogluon)
2. **Memory Issues**: Reduce `training_multiplier` or `hill_climbing.max_iterations`
3. **GPU Issues**: AutoGluon will fallback to CPU automatically
4. **Data Paths**: Update paths in `config.py` for your environment

### Performance Debugging
```python
# Check feature quality
metrics = predictor.evaluate_model()
print(f"CV MAP@3: {metrics['cv_map3']:.6f}")

# Feature importance analysis
importance = predictor.get_feature_importance()
print(importance.head(10))

# Model comparison
leaderboard = predictor.get_model_leaderboard()
print(leaderboard)
```

---

**Built for maximum competitive performance** 🏆

**Target: Beat 0.383 champion score and achieve top leaderboard position** 