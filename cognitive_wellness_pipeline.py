"""
Cognitive Wellness ML Pipeline - Using Pre-processed Dataset
=============================================================
This version uses the final_dataset_25k.csv which is already processed
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
import xgboost as xgb
import shap
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

OUTPUT_DIR = Path("ml_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("COGNITIVE WELLNESS MONITORING ML PIPELINE")
print("="*80)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Output Directory: {OUTPUT_DIR}")
print("="*80)

# ============================================================================
# STEP 1: LOAD PRE-PROCESSED DATA
# ============================================================================
print("\nSTEP 1: LOADING PRE-PROCESSED DATA")
print("-"*80)

# Use the already processed dataset
data_path = r"c:\Users\shouryaa\Dementia\final_dataset\final_dataset_25k.csv"
print(f"Loading from: {data_path}")

df = pd.read_csv(data_path)
print(f"[OK] Loaded {len(df):,} rows, {len(df.columns)} columns")

# This dataset has normalized features already
# Columns: user_id, test_run_id, battery_id, specific_subtest_id, raw_score, time_of_day, grand_index

print("\nDataset columns:")
for col in df.columns:
    print(f"  - {col}")

print("\nBasic statistics:")
print(df.describe())

print("[OK] STEP 1 COMPLETE")

# ============================================================================
# STEP 2: CREATE SYNTHETIC DEMOGRAPHIC FEATURES
# ============================================================================
print("\nSTEP 2: CREATING SYNTHETIC DEMOGRAPHIC FEATURES")
print("-"*80)

# Since this dataset doesn't have demographics, we'll create synthetic ones
# based on the data patterns

# Get unique users
unique_users = df['user_id'].unique()
n_users = len(unique_users)
print(f"Found {n_users:,} unique users")

# Create synthetic demographics for each user
np.random.seed(RANDOM_SEED)
user_demographics = pd.DataFrame({
    'user_id': unique_users,
    'age': np.random.randint(18, 80, n_users),
    'gender': np.random.choice(['m', 'f'], n_users),
    'education_level': np.random.randint(1, 10, n_users)
})

# Merge with main data
df = df.merge(user_demographics, on='user_id', how='left')

print(f"[OK] Added synthetic demographics")
print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
print(f"Gender distribution:")
print(df['gender'].value_counts())

# Create bins
df['age_bin'] = pd.cut(df['age'], 
                       bins=[0, 30, 40, 50, 60, 100], 
                       labels=['18-30', '31-40', '41-50', '51-60', '60+'])

df['edu_bin'] = pd.cut(df['education_level'], 
                       bins=[0, 3, 5, 10, 100], 
                       labels=['Low', 'Medium', 'High', 'Very High'])

print("[OK] STEP 2 COMPLETE")

# ============================================================================
# STEP 3: AGGREGATE TO USER LEVEL & CREATE CHI
# ============================================================================
print("\nSTEP 3: AGGREGATING TO USER LEVEL & CREATING CHI")
print("-"*80)

# Aggregate by user
user_features = df.groupby(['user_id', 'age', 'gender', 'age_bin', 'edu_bin']).agg({
    'raw_score': 'mean',
    'grand_index': 'mean',
    'time_of_day': 'mean'
}).reset_index()

print(f"[OK] Aggregated to {len(user_features):,} users")

# Create CHI from grand_index (which is already a composite score)
# Scale to 0-100
user_features['CHI_raw'] = user_features['grand_index']
user_features['CHI'] = 50 + 50 * np.tanh(user_features['CHI_raw'] / 2)

print(f"\nCHI statistics:")
print(user_features['CHI'].describe())

# Plot CHI distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(user_features['CHI'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Cognitive Health Index (CHI)')
plt.ylabel('Frequency')
plt.title('Distribution of CHI Scores')
plt.axvline(user_features['CHI'].mean(), color='red', linestyle='--', 
            label=f'Mean: {user_features["CHI"].mean():.1f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(user_features['CHI'], vert=True)
plt.ylabel('CHI Score')
plt.title('CHI Score Distribution')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'chi_distribution.png', dpi=150, bbox_inches='tight')
print("[OK] Saved CHI distribution plot")
plt.close()

print("[OK] STEP 3 COMPLETE")

# ============================================================================
# STEP 4: PREPARE FEATURES FOR MODELING
# ============================================================================
print("\nSTEP 4: PREPARING FEATURES FOR MODELING")
print("-"*80)

feature_columns = ['age', 'raw_score', 'grand_index', 'time_of_day']

# Encode gender
user_features['gender_encoded'] = user_features['gender'].map({'m': 0, 'f': 1})
feature_columns.append('gender_encoded')

# Encode age_bin
age_bin_dummies = pd.get_dummies(user_features['age_bin'], prefix='age_bin')
user_features = pd.concat([user_features, age_bin_dummies], axis=1)
feature_columns.extend(age_bin_dummies.columns.tolist())

# Encode edu_bin
edu_bin_dummies = pd.get_dummies(user_features['edu_bin'], prefix='edu_bin')
user_features = pd.concat([user_features, edu_bin_dummies], axis=1)
feature_columns.extend(edu_bin_dummies.columns.tolist())

# Remove NaN
user_features_clean = user_features.dropna(subset=feature_columns + ['CHI'])

X = user_features_clean[feature_columns]
y = user_features_clean['CHI']

print(f"\nFeature matrix: {X.shape}")
print(f"Target vector: {y.shape}")
print(f"\nFeatures: {feature_columns}")

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED
)

print(f"\n[OK] Data split:")
print(f"  - Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  - Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

print("[OK] STEP 4 COMPLETE")

# ============================================================================
# STEP 5: TRAIN ML MODEL
# ============================================================================
print("\nSTEP 5: TRAINING XGBOOST MODEL")
print("-"*80)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
print("[OK] Model trained")

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)
y_test_pred = xgb_model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nMODEL PERFORMANCE:")
print(f"Training   - MAE: {train_mae:.3f}, R2: {train_r2:.3f}")
print(f"Validation - MAE: {val_mae:.3f}, R2: {val_r2:.3f}")
print(f"Test       - MAE: {test_mae:.3f}, R2: {test_r2:.3f}")

# Plot predictions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, s=10)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual CHI')
plt.ylabel('Predicted CHI')
plt.title(f'Training (R2={train_r2:.3f})')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_val, y_val_pred, alpha=0.5, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual CHI')
plt.ylabel('Predicted CHI')
plt.title(f'Validation (R2={val_r2:.3f})')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual CHI')
plt.ylabel('Predicted CHI')
plt.title(f'Test (R2={test_r2:.3f})')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_predictions.png', dpi=150, bbox_inches='tight')
print("[OK] Saved prediction plots")
plt.close()

print("[OK] STEP 5 COMPLETE")

# ============================================================================
# STEP 6: RISK CATEGORIZATION
# ============================================================================
print("\nSTEP 6: RISK CATEGORIZATION")
print("-"*80)

def categorize_risk(chi_score):
    if chi_score >= 60:
        return 'Green (Good)'
    elif chi_score >= 40:
        return 'Yellow (Monitor)'
    else:
        return 'Red (High Risk)'

y_test_category = y_test.apply(categorize_risk)
y_test_pred_category = pd.Series(y_test_pred).apply(categorize_risk)

print("\nActual risk distribution:")
print(y_test_category.value_counts().sort_index())

print("\nPredicted risk distribution:")
print(y_test_pred_category.value_counts().sort_index())

print("\nClassification Report:")
print(classification_report(y_test_category, y_test_pred_category))

# Confusion matrix
cm = confusion_matrix(y_test_category, y_test_pred_category, 
                     labels=['Red (High Risk)', 'Yellow (Monitor)', 'Green (Good)'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Red', 'Yellow', 'Green'],
            yticklabels=['Red', 'Yellow', 'Green'])
plt.xlabel('Predicted Risk Category')
plt.ylabel('Actual Risk Category')
plt.title('Risk Categorization Confusion Matrix')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'risk_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("[OK] Saved confusion matrix")
plt.close()

print("[OK] STEP 6 COMPLETE")

# ============================================================================
# STEP 7: EXPLAINABILITY
# ============================================================================
print("\nSTEP 7: EXPLAINABILITY (SHAP VALUES)")
print("-"*80)

print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
print("[OK] SHAP values computed")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
print("[OK] Saved feature importance plot")
plt.close()

# SHAP summary
print("Generating SHAP summary...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_columns, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
print("[OK] Saved SHAP summary")
plt.close()

print("[OK] STEP 7 COMPLETE")

# ============================================================================
# STEP 8: SAVE ARTIFACTS
# ============================================================================
print("\nSTEP 8: SAVING ARTIFACTS")
print("-"*80)

# Save model
joblib.dump(xgb_model, OUTPUT_DIR / 'xgb_chi_model.pkl')
print("[OK] Saved model")

# Save preprocessing info
preprocessing_info = {
    'feature_columns': feature_columns,
    'gender_encoding': {'m': 0, 'f': 1},
    'risk_bands': {
        'Green (Good)': '>=60',
        'Yellow (Monitor)': '40-59',
        'Red (High Risk)': '<40'
    }
}
joblib.dump(preprocessing_info, OUTPUT_DIR / 'preprocessing_info.pkl')
print("[OK] Saved preprocessing info")

# Save metrics
with open(OUTPUT_DIR / 'model_metrics.txt', 'w') as f:
    f.write("Model Performance Metrics\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: XGBoost Regressor\n")
    f.write(f"Random Seed: {RANDOM_SEED}\n\n")
    f.write(f"Training: {len(X_train):,} samples\n")
    f.write(f"  MAE: {train_mae:.3f}\n")
    f.write(f"  R2:  {train_r2:.3f}\n\n")
    f.write(f"Validation: {len(X_val):,} samples\n")
    f.write(f"  MAE: {val_mae:.3f}\n")
    f.write(f"  R2:  {val_r2:.3f}\n\n")
    f.write(f"Test: {len(X_test):,} samples\n")
    f.write(f"  MAE: {test_mae:.3f}\n")
    f.write(f"  R2:  {test_r2:.3f}\n\n")
    f.write("Top 10 Features:\n")
    f.write(feature_importance.head(10).to_string(index=False))

print("[OK] Saved metrics")

print("[OK] STEP 8 COMPLETE")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETE")
print("="*80)

print("\n[OK] All steps completed successfully!")
print(f"\nArtifacts saved to: {OUTPUT_DIR}/")
print("\nSummary:")
print(f"  - Dataset: {len(df):,} rows")
print(f"  - Users: {len(user_features):,}")
print(f"  - Features: {len(feature_columns)}")
print(f"  - Model: XGBoost Regressor")
print(f"  - Test MAE: {test_mae:.3f}")
print(f"  - Test R2: {test_r2:.3f}")

print("\nNext steps:")
print("  1. Review plots in ml_artifacts/")
print("  2. Examine model_metrics.txt")
print("  3. Use saved model for inference")

print("\n" + "="*80)
print("DISCLAIMER: Wellness screening tool, NOT clinical diagnostic system")
print("="*80)
