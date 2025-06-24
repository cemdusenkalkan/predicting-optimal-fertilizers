#!/usr/bin/env python3
"""
Real Competitive Approach - 0.38+ Target
Based on structural insights from forum intelligence
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def create_structural_features(df):
    """
    Apply ONLY the proven structural insights from forum
    Focus on data generation patterns, not feature engineering
    """
    df = df.copy()
    
    # 1. CATEGORICAL TREATMENT - BIGGEST SINGLE IMPROVEMENT (+0.006)
    # The synthetic data was generated with categorical logic
    numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
    
    for col in numerical_cols:
        if col in df.columns:
            # Create categorical bins - this exploits the generation pattern
            df[f'{col}_cat'] = pd.cut(df[col], bins=20, labels=False, duplicates='drop')
            df[f'{col}_cat'] = df[f'{col}_cat'].astype('category')
    
    # 2. CONSTANT FEATURE TRICK (+0.005)
    # Exploits XGBoost's handling of constant features
    df['const'] = 1
    
    # 3. NPK RATIOS - HIDDEN SIGNAL IN SYNTHETIC DATA
    # The generation algorithm used ratio relationships
    epsilon = 1e-8
    if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
        
        # Clip extreme ratios
        for col in ['N_P_ratio', 'N_K_ratio', 'P_K_ratio']:
            df[col] = np.clip(df[col], 0, 100)
    
    # 4. ENVIRONMENTAL MAX (PROVEN PATTERN)
    env_cols = ['Temperature', 'Humidity', 'Moisture']
    if all(col in df.columns for col in env_cols):
        df['env_max'] = df[env_cols].max(axis=1)
    
    # 5. CROP-SOIL INTERACTION (STRUCTURAL IMPORTANCE)
    if 'Crop Type' in df.columns and 'Soil Type' in df.columns:
        df['Crop_Soil_combo'] = df['Crop Type'].astype(str) + '_' + df['Soil Type'].astype(str)
        df['Crop_Soil_combo'] = df['Crop_Soil_combo'].astype('category')
    
    # Ensure categorical columns are properly typed
    categorical_cols = ['Crop Type', 'Soil Type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

def map3_score_from_proba(y_true, y_pred_proba):
    """Calculate MAP@3 from probability predictions"""
    def apk(actual, predicted, k=3):
        if len(predicted) > k:
            predicted = predicted[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        if not actual:
            return 0.0
        
        return score / min(len(actual), k)
    
    top3_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]
    
    map3_scores = []
    for i, true_label in enumerate(y_true):
        predicted_labels = top3_indices[i]
        map3_scores.append(apk([true_label], predicted_labels, k=3))
    
    return np.mean(map3_scores)

def structural_cv(X, y, params, n_splits=10):
    """
    Proper CV focused on structural performance
    No data leakage tricks
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    oof_predictions = np.zeros((len(X), 7))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        # Predict
        y_pred_proba = model.predict_proba(X_val_fold)
        oof_predictions[val_idx] = y_pred_proba
        
        # Calculate MAP@3
        fold_score = map3_score_from_proba(y_val_fold, y_pred_proba)
        cv_scores.append(fold_score)
        
        print(f"Fold {fold+1} MAP@3: {fold_score:.6f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\n🏆 Structural CV Results:")
    print(f"Mean MAP@3: {mean_score:.6f} (+/- {std_score:.6f})")
    
    return cv_scores, oof_predictions, mean_score

def main():
    """Main execution function"""
    print("🎯 Real Competitive Approach - Structural Insights")
    print("Target: 0.38+ MAP@3 through understanding data generation patterns")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    
    # Fix column name typo
    if 'Temparature' in train_df.columns:
        train_df = train_df.rename(columns={'Temparature': 'Temperature'})
        test_df = test_df.rename(columns={'Temparature': 'Temperature'})
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare training data
    X_train = train_df.drop(['Fertilizer Name', 'id'], axis=1, errors='ignore')
    y_train = train_df['Fertilizer Name']
    
    # Apply structural feature engineering
    print("\nApplying structural insights...")
    X_train_structural = create_structural_features(X_train)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Structural features: {X_train_structural.shape[1]}")
    
    # Encode target
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Configure XGBoost with native categorical support
    print("\nConfiguring XGBoost with structural optimizations...")
    
    # Check GPU availability
    use_gpu = True
    try:
        test_model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, enable_categorical=True)
        print("✓ GPU available")
    except:
        use_gpu = False
        print("⚠ Using CPU")
    
    # Structural XGBoost configuration
    structural_params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'enable_categorical': True,  # CRITICAL for structural patterns
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'random_state': 42,
        'verbosity': 0,
        
        # Optimized for structural patterns
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'min_child_weight': 3
    }
    
    if use_gpu:
        structural_params['gpu_id'] = 0
    else:
        structural_params['n_jobs'] = -1
    
    # Run structural cross-validation
    print("\nRunning structural cross-validation...")
    cv_scores, oof_preds, mean_cv_score = structural_cv(
        X_train_structural, y_train_encoded, 
        structural_params, 
        n_splits=10
    )
    
    # Analyze results
    print(f"\n📊 Structural Performance Analysis:")
    print(f"Current CV MAP@3: {mean_cv_score:.6f}")
    
    if mean_cv_score >= 0.38:
        print("🏆 EXCELLENT! Above 0.38 target - structural insights working!")
    elif mean_cv_score >= 0.35:
        print("🥈 GOOD! Above 0.35 - competitive range")
    elif mean_cv_score >= 0.32:
        print("🥉 DECENT! Above 0.32 - room for improvement")
    else:
        print("📈 BASELINE - Need deeper structural analysis")
    
    print(f"\n🎯 Structural Techniques Applied:")
    print(f"  ✅ Categorical treatment (+0.006 expected)")
    print(f"  ✅ Constant feature (+0.005 expected)")
    print(f"  ✅ NPK ratios (synthetic pattern)")
    print(f"  ✅ XGBoost native categorical support")
    print(f"  ✅ Environmental max pattern")
    print(f"  ✅ Crop-Soil structural interaction")
    
    # Train final model and make predictions
    print("\nTraining final structural model...")
    final_model = xgb.XGBClassifier(**structural_params)
    final_model.fit(X_train_structural, y_train_encoded)
    
    # Prepare test data
    X_test = test_df.drop('id', axis=1, errors='ignore')
    X_test_structural = create_structural_features(X_test)
    
    # Make predictions
    test_probabilities = final_model.predict_proba(X_test_structural)
    top3_predictions = np.argsort(test_probabilities, axis=1)[:, ::-1][:, :3]
    
    # Convert to submission format
    top3_fertilizer_names = []
    for i in range(len(top3_predictions)):
        fertilizer_names = [label_encoder.inverse_transform([pred])[0] for pred in top3_predictions[i]]
        top3_fertilizer_names.append(' '.join(fertilizer_names))
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Fertilizer Name': top3_fertilizer_names
    })
    
    # Save submission
    submission.to_csv('structural_submission.csv', index=False)
    print(f"\n💾 Submission saved: structural_submission.csv")
    
    print(f"\n🎯 Final Structural Model Summary:")
    print(f"  • CV MAP@3: {mean_cv_score:.6f}")
    print(f"  • Expected LB: {mean_cv_score:.3f} - {mean_cv_score + 0.005:.3f}")
    print(f"  • Features: {X_train_structural.shape[1]}")
    print(f"  • Approach: Structural pattern exploitation")
    
    if mean_cv_score >= 0.38:
        print(f"  🏆 TARGET ACHIEVED! Structural insights successful!")
    else:
        print(f"  📈 Gap to 0.38: {0.38 - mean_cv_score:.6f}")
        print(f"  🔍 Need deeper analysis of synthetic data patterns")

if __name__ == "__main__":
    main() 