# Quick Start Guide - Cognitive Wellness ML Pipeline

## 🚀 Quick Start

### Run the Complete Pipeline
```bash
cd c:\Users\shouryaa\Dementia
python cognitive_wellness_pipeline.py
```

### Test Inference on New Data
```bash
python ml_artifacts/inference_example.py
```

---

## 📁 Project Structure

```
c:\Users\shouryaa\Dementia\
│
├── cognitive_wellness_pipeline.py    # Main pipeline script
├── README.md                          # Full documentation
├── EXECUTION_SUMMARY.md               # Execution summary
├── QUICK_START.md                     # This file
│
├── final_dataset/                     # Data directory
│   ├── final_dataset_25k.csv         # 25k sample dataset
│   └── cleaned_data.csv               # Full cleaned dataset
│
└── ml_artifacts/                      # Model outputs
    ├── xgb_chi_model.pkl             # Trained model
    ├── preprocessing_info.pkl         # Preprocessing params
    ├── model_metrics.txt              # Performance metrics
    ├── inference_example.py           # Inference script
    ├── chi_distribution.png           # CHI distribution plot
    ├── model_predictions.png          # Predictions plot
    ├── risk_confusion_matrix.png      # Confusion matrix
    ├── shap_summary.png               # SHAP summary
    └── feature_importance.png         # Feature importance
```

---

## 🎯 What the Pipeline Does

1. **Loads** NCPT cognitive test data
2. **Engineers** features (z-scores, demographic bins)
3. **Creates** Cognitive Health Index (CHI) score (0-100)
4. **Trains** XGBoost model to predict CHI
5. **Categorizes** users into risk bands (Green/Yellow/Red)
6. **Explains** predictions using SHAP values
7. **Saves** model and artifacts for future use

---

## 📊 Model Performance

- **Test MAE**: 0.102 (very low error)
- **Test R²**: 1.000 (excellent fit)
- **Test Samples**: 4,920 users

---

## 🔍 Understanding CHI Scores

| CHI Score | Risk Category | Interpretation |
|-----------|---------------|----------------|
| **60-100** | 🟢 Green (Good) | Cognitive wellness appears good |
| **40-59** | 🟡 Yellow (Monitor) | Consider monitoring |
| **0-39** | 🔴 Red (High Risk) | May benefit from intervention |

---

## 💻 Using the Model in Your Code

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('ml_artifacts/xgb_chi_model.pkl')
preprocessing_info = joblib.load('ml_artifacts/preprocessing_info.pkl')

# Prepare new user data
new_user = {
    'age': 45,
    'gender': 'm',
    'raw_score': 0.5,
    'grand_index': 0.8,
    'time_of_day': 14.5
}

# Preprocess (see inference_example.py for full code)
# X_new = preprocess_user(new_user, preprocessing_info)

# Predict
# chi_score = model.predict(X_new)[0]
# print(f"Predicted CHI: {chi_score:.2f}")
```

---

## 📈 Viewing Results

All visualizations are saved in `ml_artifacts/`:

1. **CHI Distribution** (`chi_distribution.png`)
   - Shows how CHI scores are distributed across users

2. **Model Predictions** (`model_predictions.png`)
   - Compares actual vs predicted CHI scores

3. **Risk Confusion Matrix** (`risk_confusion_matrix.png`)
   - Shows how well the model categorizes risk

4. **SHAP Summary** (`shap_summary.png`)
   - Explains which features influence predictions

5. **Feature Importance** (`feature_importance.png`)
   - Ranks features by importance

---

## ⚙️ Key Parameters

### CHI Formula Weights
- Memory: 30%
- Attention: 25%
- Processing Speed: 20%
- Executive Function: 25%

### Model Hyperparameters
- Algorithm: XGBoost Regressor
- Trees: 100
- Max Depth: 6
- Learning Rate: 0.1
- Random Seed: 42

---

## ⚠️ Important Notes

1. **Not for Diagnosis**: This is a wellness screening tool, NOT for clinical diagnosis
2. **Consult Professionals**: Always consult healthcare professionals for medical advice
3. **Wellness Monitoring**: Designed for tracking cognitive wellness trends
4. **Reproducibility**: Uses fixed random seed (42) for consistent results

---

## 🔧 Troubleshooting

### Issue: Module not found
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```

### Issue: File not found
- Ensure you're in the correct directory: `c:\Users\shouryaa\Dementia`
- Check that `final_dataset/final_dataset_25k.csv` exists

### Issue: Unicode errors
- The pipeline uses ASCII-safe characters only
- Should work on all Windows systems

---

## 📚 Documentation Files

- **README.md**: Full project documentation
- **EXECUTION_SUMMARY.md**: Detailed execution summary
- **QUICK_START.md**: This quick start guide
- **ml_artifacts/inference_example.py**: Inference examples

---

## 🎓 Next Steps

1. ✅ Review the visualizations in `ml_artifacts/`
2. ✅ Read `EXECUTION_SUMMARY.md` for detailed results
3. ✅ Try `inference_example.py` to see how to use the model
4. ✅ Experiment with different user data
5. ✅ Consider future enhancements (see README.md)

---

## 📞 Support

For questions or issues:
1. Check the README.md for detailed documentation
2. Review the EXECUTION_SUMMARY.md for pipeline details
3. Examine the inference_example.py for usage examples

---

**Last Updated**: December 15, 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready
