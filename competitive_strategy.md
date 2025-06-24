# Fertilizer Competition - Ultra-Competitive Strategy 

## Key Intelligence Gathered

### From Our Analysis (findings.md)
- **Crop Type** and **Soil Type** are dominant predictors
- NPK nutrients show moderate importance 
- Environmental factors (temp/humidity/moisture) are secondary
- Dataset is perfectly balanced (14.3% each class)
- Data appears synthetic with uniform distributions
- 750K training samples, 250K test samples

### From Kaggle Forum Intelligence

#### Critical Discoveries:
1. **Original Dataset Integration**: Forum confirms adding original dataset (100 samples) improves performance
2. **Data Expansion**: Duplicating competition training data multiple times enhances performance
3. **Categorical Treatment**: Treating ALL features as categorical dramatically improves scores (0.363 → 0.369)
4. **Constant Feature**: Adding a constant column "const"=1 universally improves CV by ~0.005
5. **N/P/K Ratios**: Hidden signal lies in nutrient ratios, not absolute values
6. **Target Distribution**: Public LB distribution analyzed - almost identical to training
7. **Original Data Has No Signal**: Original 100-sample dataset is pure noise

#### Proven Features That Work:
- `env_max`: max(Temperature, Humidity, Moisture) 
- `temp_suitability`: binary feature for crop-specific temperature ranges
- NPK ratios: N/P, N/K, P/K
- Categorical versions of all numerical features

#### Model Insights:
- XGBoost performs exceptionally well (forum consensus)
- CatBoost useful for speed/categorical handling
- More folds (5→20) can help but risk overfitting
- GPU training recommended

## Ultra-Competitive Implementation Strategy

### Phase 1: Advanced Data Engineering

#### Dataset Expansion Strategy:
```python
# Expand competition training data (proven technique)
X_train_expanded = pd.concat([X_train] * 3, ignore_index=True)  # 3x expansion
y_train_expanded = pd.concat([y_train] * 3, ignore_index=True)

# Add original dataset with multiplier
original_df = pd.read_csv('original_100_samples.csv')  # if available
X_orig = original_df.drop('Fertilizer Name', axis=1)
y_orig = original_df['Fertilizer Name']
X_train_final = pd.concat([X_train_expanded, X_orig * 2], ignore_index=True)
y_train_final = pd.concat([y_train_expanded, y_orig * 2], ignore_index=True)
```

#### Feature Engineering Arsenal:
```python
# 1. Categorical versions of ALL numerical features (PROVEN)
for col in ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']:
    df[f'{col}_cat'] = pd.cut(df[col], bins=20, labels=False)

# 2. Constant feature (PROVEN)  
df['const'] = 1

# 3. Environmental features (PROVEN)
df['env_max'] = df[['Temperature', 'Humidity', 'Moisture']].max(axis=1)
df['temp_humidity_index'] = df['Temperature'] * df['Humidity'] / 100
df['climate_comfort'] = (df['Temperature'] + df['Humidity'] + df['Moisture']) / 3

# 4. NPK Ratios (HIDDEN SIGNAL)
df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-8)
df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-8) 
df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-8)
df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
df['NPK_balance'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)

# 5. Temperature suitability (PROVEN)
crop_temp_map = {
    'Sugarcane': (26, 35), 'Maize': (25, 32), 'Wheat': (20, 30),
    'Paddy': (25, 35), 'Cotton': (25, 35), 'Tobacco': (20, 30),
    'Barley': (15, 25), 'Millets': (25, 35), 'Pulses': (20, 30),
    'Oil seeds': (20, 30), 'Ground Nuts': (25, 32)
}
df['temp_suitability'] = df.apply(lambda row: 1 if crop_temp_map.get(row['Crop Type'], (25, 32))[0] <= row['Temperature'] <= crop_temp_map.get(row['Crop Type'], (25, 32))[1] else 0, axis=1)

# 6. Crop-Soil interactions (HIGH IMPORTANCE FROM OUR ANALYSIS)
df['Crop_Soil_combo'] = df['Crop Type'] + '_' + df['Soil Type']

# 7. Target encoding for Crop-Soil combinations
from sklearn.model_selection import KFold
def target_encode(df, col, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = np.zeros(len(df))
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        encoded[val_idx] = df.iloc[val_idx][col].map(means)
    return encoded

df['Crop_Soil_target_enc'] = target_encode(df, 'Crop_Soil_combo', 'target')
```

### Phase 2: Multi-Model Ensemble with GPU Acceleration

#### Model Configuration:
```python
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  
from catboost import CatBoostClassifier

# XGBoost with GPU (PRIMARY MODEL)
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'tree_method': 'gpu_hist',  # GPU acceleration
    'gpu_id': 0,
    'enable_categorical': True,  # Handle categoricals natively
    'max_depth': 8,
    'learning_rate': 0.02,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'random_state': 42
}

# LightGBM with GPU  
lgb_params = {
    'objective': 'multiclass',
    'num_class': 7, 
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'boosting_type': 'gbdt',
    'max_depth': 7,
    'learning_rate': 0.02,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'random_state': 42
}

# CatBoost with GPU
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 8,
    'task_type': 'GPU',  # GPU acceleration
    'devices': '0',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False
}
```

#### Advanced Cross-Validation Strategy:
```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Multi-level CV with proper original data handling
def advanced_cv_strategy(X, y, X_orig=None, y_orig=None, n_splits=10):
    """
    Proper CV strategy:
    1. Use stratified K-fold only on competition data
    2. Add original data only to training folds 
    3. Keep validation folds pure
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds = np.zeros((len(X), 7))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Split competition data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Add original data ONLY to training (CRITICAL)
        if X_orig is not None:
            X_train_fold = pd.concat([X_train_fold, X_orig], ignore_index=True)
            y_train_fold = pd.concat([y_train_fold, y_orig], ignore_index=True)
        
        # Train models
        models = train_ensemble(X_train_fold, y_train_fold)
        
        # Predict on validation (pure competition data)
        val_preds = ensemble_predict(models, X_val_fold)
        oof_preds[val_idx] = val_preds
        
        # Calculate MAP@3
        fold_score = map_k(y_val_fold, val_preds, k=3)
        cv_scores.append(fold_score)
        print(f"Fold {fold+1} MAP@3: {fold_score:.6f}")
    
    return cv_scores, oof_preds
```

### Phase 3: Hyperparameter Optimization with Optuna

#### Smart Optimization Strategy:
```python
def objective(trial):
    # XGBoost hyperparameters
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    
    # Quick 3-fold CV for optimization speed
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx] 
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = XGBClassifier(**xgb_params, **base_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        
        preds = model.predict_proba(X_val)
        score = map_k_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, n_jobs=1)  # GPU limits parallelization
```

### Phase 4: Advanced Ensemble Strategy

#### Weighted Soft Voting with Optuna:
```python
def optimize_ensemble_weights(oof_predictions, y_true):
    """Find optimal ensemble weights using Optuna"""
    def objective(trial):
        w_xgb = trial.suggest_float('w_xgb', 0.1, 0.8)
        w_lgb = trial.suggest_float('w_lgb', 0.1, 0.8) 
        w_cat = trial.suggest_float('w_cat', 0.1, 0.8)
        
        # Normalize weights
        total = w_xgb + w_lgb + w_cat
        w_xgb, w_lgb, w_cat = w_xgb/total, w_lgb/total, w_cat/total
        
        # Weighted ensemble
        ensemble_pred = w_xgb * oof_predictions['xgb'] + w_lgb * oof_predictions['lgb'] + w_cat * oof_predictions['cat']
        
        return map_k_score(y_true, ensemble_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    return study.best_params
```

### Phase 5: Final Submission Strategy

#### Multi-Submission Approach:
1. **Main Ensemble**: XGB + LGB + CatBoost with optimized weights
2. **XGB Solo**: Best single XGBoost model (backup)
3. **Category-Heavy**: All features as categorical + constant feature
4. **Ratio-Heavy**: Focus on NPK ratios and environmental features

## Expected Performance Targets

Based on forum discussions and our analysis:
- **Baseline**: ~0.35 MAP@3 
- **With Categorical Features**: ~0.37 MAP@3
- **With All Techniques**: **>0.39 MAP@3**
- **Stretch Goal**: **>0.41 MAP@3**

## Implementation Priorities

1. **HIGH**: Categorical feature conversion + constant feature
2. **HIGH**: NPK ratios and environmental max
3. **HIGH**: XGBoost with GPU training  
4. **MEDIUM**: Original data integration (if available)
5. **MEDIUM**: Advanced ensemble weighting
6. **LOW**: 20-fold CV (high risk of overfitting)

## Notes to Self

- Forum strongly emphasizes **treating everything as categorical** - this alone gives +0.006 improvement
- **NPK ratios are the hidden signal** - don't rely on absolute values
- **Original dataset is noise** - use sparingly if at all
- **XGBoost dominance** - focus optimization efforts here
- **Constant feature trick** - easy +0.005 improvement
- **Temperature suitability** - domain knowledge feature that works
- **GPU training** - essential for speed with large expanded dataset
- MAP@3 metric rewards confidence in top predictions - optimize for this specifically 