# Cognitive Wellness Monitoring ML Pipeline

## Overview

This project implements a machine learning pipeline for cognitive wellness monitoring based on NCPT (Neuropsychological Cognitive Performance Test) data. The system is designed for **wellness screening and monitoring**, NOT for clinical diagnosis.

## Purpose

- Monitor cognitive wellness trends over time
- Provide a Cognitive Health Index (CHI) score (0-100 scale)
- Categorize users into risk bands for wellness tracking
- Offer explainable predictions using SHAP values

## Pipeline Steps

### Step 1: Dataset Understanding
- Loads and explores the NCPT cognitive test dataset
- Examines demographic and performance metrics
- Handles missing values

### Step 2: Feature Engineering
- Creates demographic bins (age, education)
- Computes z-scores normalized by demographic groups
- Maps subtests to cognitive domains (memory, attention, processing speed, executive function)

### Step 3: Cognitive Health Index (CHI) Construction
- Defines CHI as a weighted combination of domain indices:
  - Memory: 30%
  - Attention: 25%
  - Processing Speed: 20%
  - Executive Function: 25%
- Scales CHI to 0-100 range using sigmoid transformation

### Step 4: Train/Validation/Test Split
- Splits data: 60% training, 20% validation, 20% test
- Uses stratified sampling to maintain CHI distribution

### Step 5: Baseline ML Model Training
- Trains XGBoost Regressor to predict CHI
- Evaluates using MAE and R² metrics
- Generates prediction visualizations

### Step 6: Risk Categorization
- Categorizes CHI into risk bands:
  - **Green (Good)**: CHI >= 60
  - **Yellow (Monitor)**: 40 <= CHI < 60
  - **Red (High Risk)**: CHI < 40
- Provides confusion matrix and classification report

### Step 7: Explainability
- Computes SHAP values for model interpretability
- Generates feature importance rankings
- Provides per-sample explanations

### Step 8: Save Artifacts
- Saves trained model (`xgb_chi_model.pkl`)
- Saves preprocessing information (`preprocessing_info.pkl`)
- Saves feature list and metrics
- Creates inference example script

## Model Performance

Based on the 25k sample dataset:

- **Training Set**: MAE = 0.088, R² = 1.000
- **Validation Set**: MAE = 0.102, R² = 1.000
- **Test Set**: MAE = 0.102, R² = 1.000

The model achieves excellent performance with very low error rates.

## Key Features

1. **Normalized Scoring**: Z-scores computed within demographic groups for fair comparison
2. **Explainable AI**: SHAP values provide transparency in predictions
3. **Risk Stratification**: Clear categorization for wellness monitoring
4. **Reproducible**: Fixed random seed (42) ensures consistent results
5. **Modular Design**: Each step is clearly separated and documented

## Output Artifacts

All artifacts are saved in the `ml_artifacts/` directory:

- `xgb_chi_model.pkl` - Trained XGBoost model
- `preprocessing_info.pkl` - Feature engineering parameters
- `model_metrics.txt` - Performance metrics
- `feature_list.txt` - List of features used
- `chi_distribution.png` - CHI score distribution
- `model_predictions.png` - Actual vs predicted CHI
- `risk_confusion_matrix.png` - Risk categorization performance
- `shap_summary.png` - SHAP feature importance
- `feature_importance.png` - XGBoost feature importance

## Usage

### Running the Pipeline

```bash
python cognitive_wellness_pipeline.py
```

### Using the Trained Model for Inference

See `inference_example.py` in the `ml_artifacts/` directory for a complete example of how to use the trained model to predict CHI for new users.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap
- joblib

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

## Important Disclaimers

⚠️ **This is a wellness screening tool, NOT a clinical diagnostic system.**

- Results should not be used for medical diagnosis
- Consult healthcare professionals for clinical assessment
- This tool is intended for monitoring cognitive wellness trends
- Not a substitute for professional medical advice

## Dataset

The pipeline uses NCPT cognitive test data with the following features:
- User demographics (age, gender, education)
- Cognitive performance metrics (raw scores, grand index)
- Test metadata (battery ID, subtest ID, time of day)

## Future Enhancements

Potential improvements for future versions:
1. Incorporate longitudinal data for trend analysis
2. Add more cognitive domains
3. Implement deep learning models
4. Include speech and MRI data
5. Develop mobile app interface
6. Add personalized recommendations

## License

This project is for research and educational purposes.

## Contact

For questions or contributions, please contact the Cognitive Wellness Team.

---

**Last Updated**: December 15, 2025
**Version**: 1.0
**Random Seed**: 42
