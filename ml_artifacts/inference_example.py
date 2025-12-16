"""
Inference Example: Using the Trained Model for New Predictions
===============================================================

This script demonstrates how to use the trained cognitive wellness model
to predict CHI (Cognitive Health Index) for new users.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load the trained model and preprocessing info
MODEL_DIR = Path("ml_artifacts")
model = joblib.load(MODEL_DIR / 'xgb_chi_model.pkl')
preprocessing_info = joblib.load(MODEL_DIR / 'preprocessing_info.pkl')

print("="*80)
print("COGNITIVE WELLNESS MODEL - INFERENCE EXAMPLE")
print("="*80)

print("\n[OK] Loaded trained model")
print(f"[OK] Model features: {len(preprocessing_info['feature_columns'])}")

# ============================================================================
# EXAMPLE 1: Single User Prediction
# ============================================================================
print("\n" + "-"*80)
print("EXAMPLE 1: SINGLE USER PREDICTION")
print("-"*80)

# Example new user data
new_user = {
    'age': 45,
    'gender': 'm',
    'raw_score': 0.5,
    'grand_index': 0.8,
    'time_of_day': 14.5
}

print(f"\nNew user data:")
for key, value in new_user.items():
    print(f"  {key}: {value}")

# Preprocess the new user data
def preprocess_user(user_data, preprocessing_info):
    """Preprocess user data for model inference"""
    
    # Create age bin
    age = user_data['age']
    if age <= 30:
        age_bin = '18-30'
    elif age <= 40:
        age_bin = '31-40'
    elif age <= 50:
        age_bin = '41-50'
    elif age <= 60:
        age_bin = '51-60'
    else:
        age_bin = '60+'
    
    # Encode gender
    gender_encoding = preprocessing_info['gender_encoding']
    gender_encoded = gender_encoding.get(user_data['gender'], 0)
    
    # Create feature vector
    features = {}
    features['age'] = user_data['age']
    features['raw_score'] = user_data['raw_score']
    features['grand_index'] = user_data['grand_index']
    features['time_of_day'] = user_data['time_of_day']
    features['gender_encoded'] = gender_encoded
    
    # One-hot encode age_bin
    age_bins = ['18-30', '31-40', '41-50', '51-60', '60+']
    for bin_name in age_bins:
        features[f'age_bin_{bin_name}'] = 1 if bin_name == age_bin else 0
    
    # One-hot encode edu_bin (assuming Medium as default for new users)
    edu_bins = ['Low', 'Medium', 'High', 'Very High']
    for bin_name in edu_bins:
        features[f'edu_bin_{bin_name}'] = 1 if bin_name == 'Medium' else 0
    
    # Create DataFrame with correct column order
    feature_columns = preprocessing_info['feature_columns']
    feature_df = pd.DataFrame([features])[feature_columns]
    
    return feature_df

# Preprocess and predict
X_new = preprocess_user(new_user, preprocessing_info)
chi_pred = model.predict(X_new)[0]

# Categorize risk
def categorize_risk(chi_score):
    """Categorize CHI into risk bands"""
    if chi_score >= 60:
        return 'Green (Good)'
    elif chi_score >= 40:
        return 'Yellow (Monitor)'
    else:
        return 'Red (High Risk)'

risk_category = categorize_risk(chi_pred)

print(f"\nPrediction Results:")
print(f"  Predicted CHI: {chi_pred:.2f}")
print(f"  Risk Category: {risk_category}")

# ============================================================================
# EXAMPLE 2: Batch Prediction for Multiple Users
# ============================================================================
print("\n" + "-"*80)
print("EXAMPLE 2: BATCH PREDICTION FOR MULTIPLE USERS")
print("-"*80)

# Create sample batch of users
batch_users = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'gender': ['m', 'f', 'm', 'f', 'm'],
    'raw_score': [0.8, 0.5, 0.3, 0.6, 0.4],
    'grand_index': [1.2, 0.7, 0.4, 0.8, 0.5],
    'time_of_day': [10, 14, 16, 11, 15]
})

print(f"\nBatch of {len(batch_users)} users:")
print(batch_users)

# Process batch
batch_predictions = []
for idx, user in batch_users.iterrows():
    user_dict = user.to_dict()
    X_user = preprocess_user(user_dict, preprocessing_info)
    chi = model.predict(X_user)[0]
    risk = categorize_risk(chi)
    batch_predictions.append({
        'user_id': idx + 1,
        'age': user['age'],
        'gender': user['gender'],
        'predicted_chi': chi,
        'risk_category': risk
    })

# Display results
results_df = pd.DataFrame(batch_predictions)
print(f"\nBatch Prediction Results:")
print(results_df.to_string(index=False))

# ============================================================================
# EXAMPLE 3: Understanding Risk Bands
# ============================================================================
print("\n" + "-"*80)
print("EXAMPLE 3: RISK BAND INTERPRETATION")
print("-"*80)

risk_bands = preprocessing_info['risk_bands']
print("\nRisk Band Definitions:")
for category, threshold in risk_bands.items():
    print(f"  {category}: CHI {threshold}")

print("\nInterpretation:")
print("  - Green (Good): Cognitive wellness appears good")
print("  - Yellow (Monitor): Consider monitoring cognitive wellness")
print("  - Red (High Risk): May benefit from wellness intervention")

print("\nIMPORTANT DISCLAIMER:")
print("  This is a wellness screening tool, NOT a clinical diagnostic system.")
print("  Consult healthcare professionals for clinical assessment.")

# ============================================================================
# EXAMPLE 4: Feature Importance
# ============================================================================
print("\n" + "-"*80)
print("EXAMPLE 4: FEATURE IMPORTANCE")
print("-"*80)

# Get feature importances from the model
feature_importance = pd.DataFrame({
    'Feature': preprocessing_info['feature_columns'],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head(5).to_string(index=False))

print("\nInterpretation:")
print("  Features with higher importance have more influence on CHI predictions.")
print("  The 'grand_index' is typically the most important feature as it")
print("  represents overall cognitive performance.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("INFERENCE EXAMPLES COMPLETE")
print("="*80)

print("\nKey Takeaways:")
print("  1. Load model and preprocessing info from ml_artifacts/")
print("  2. Preprocess new user data to match training format")
print("  3. Use model.predict() to get CHI scores")
print("  4. Categorize CHI into risk bands for interpretation")
print("  5. Always include disclaimer about wellness vs diagnosis")

print("\nFor more information, see README.md")
print("="*80)
