# Credit Card Fraud Detection

A machine learning project that builds a production-ready fraud detection model using feature engineering, class imbalance handling, and model comparison on the Kaggle Credit Card Fraud Detection dataset.

## Overview

This project demonstrates a complete fraud detection pipeline from data exploration to deployment. With only 0.172% of transactions being fraudulent, the challenge isn't accuracy—it's precision and recall. The solution uses intelligent feature engineering combined with gradient boosting to achieve **98.1% ROC-AUC with 0.025% false positive rate**.

## Key Results

| Metric | Logistic Regression | XGBoost | LightGBM |
|--------|---|---|---|
| ROC-AUC | 0.955 | **0.981** | 0.975 |
| Fraud Detection Rate | 89.8% | 83.7% | 81.6% |
| False Positive Rate | 2.60% | **0.025%** | 0.15% |
| Precision | 6% | **85%** | 60% |

**Winner: XGBoost** — Best balance of fraud detection (84%) with virtually zero false positives (0.025%)

## The Challenge

Standard ML approaches fail on this dataset because:
- **Extreme class imbalance**: 0.172% fraud vs 99.828% legitimate
- **Naive baseline is deceptive**: Predicting "no fraud" gives 99.8% accuracy while catching 0% of actual fraud
- **Production constraints**: False positives are expensive (customer friction, support costs, trust erosion)

## Solution Approach

### 1. Feature Engineering
Rather than relying solely on the pre-computed PCA features (V1-V28), I created new signals:

**Time-based Features:**
- Hour of day (fraud peaks at certain times)
- Time-of-day categories (night transactions are riskier)
- Transaction velocity (count and amount in 1-hour windows)
- Time since last transaction

**Amount-based Features:**
- Log transformation (handles skewed distribution)
- Deviation from user's historical average
- Large transaction flags (>95th percentile)
- Small amount + night time interactions

**Statistical Features:**
- User's average transaction amount
- Amount standard deviation
- Rolling window statistics (1-hour aggregations)

### 2. Class Imbalance Handling

**SMOTE (Synthetic Minority Oversampling)**: Generated synthetic fraud examples to balance training data to 30% minority class

**Weighted Loss**: Tree-based models use `scale_pos_weight` (580:1) to penalize fraud misclassification more heavily

### 3. Model Comparison

Trained three models to understand where complexity is needed:

- **Logistic Regression**: Interpretable baseline, captures linear patterns
- **XGBoost**: High-performance gradient boosting, handles non-linear fraud patterns
- **LightGBM**: Fast, efficient alternative with similar performance

### 4. Threshold Optimization

Rather than using the default 0.5 probability threshold, optimized for:
- F1-score maximization
- Precision-recall trade-offs
- Operational constraints (false positive rate tolerance)

## Project Structure

```
fraud-detection/
├── creditcard_fraud_detection.ipynb    # Full analysis notebook
├── README.md                            # This file
├── models/
│   ├── fraud_risk_model_xgboost.pkl    # Trained XGBoost model
│   ├── feature_scaler.pkl              # StandardScaler for preprocessing
│   └── feature_config.pkl              # Feature names & config
├── visualizations/
│   ├── model_comparison_curves.png     # ROC & PR curves
│   ├── feature_importance_comparison.png
│   ├── shap_importance.png             # SHAP feature importance
│   ├── threshold_optimization.png
│   ├── risk_score_analysis.png
│   └── fraud_distribution.png
└── data/
    └── creditcard.csv                  # Dataset (from Kaggle)
```

## Getting Started

### Prerequisites

```bash
python 3.9+
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
shap
imbalanced-learn
```

### Installation

```bash
# Clone repository
git clone https://github.com/nikhilm931/Fraud_Detection_CC.git
cd fraud-detection

# Create conda environment (recommended)
conda create -n fraud_detection python=3.9
conda activate fraud_detection

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap imbalanced-learn jupyter
```

### Running the Analysis

```bash
# Start Jupyter
jupyter notebook

# Open creditcard_fraud_detection.ipynb and run all cells
```

The notebook is organized into phases:
1. **Data Exploration** — Understand the dataset and class imbalance
2. **Feature Engineering** — Create meaningful signals from raw data
3. **Data Preparation** — Split, scale, handle imbalance
4. **Model Building** — Train three models with different approaches
5. **Evaluation** — Compare models with ROC/PR curves and metrics
6. **Interpretability** — SHAP values and feature importance
7. **Business Impact** — Risk scoring and financial impact
8. **Deployment** — Save and load models for inference

## Key Insights

### Feature Importance

The model identifies these as top fraud signals:
- **Amount patterns** — Unusual transaction amounts (high or low)
- **Time patterns** — Transactions at unusual hours (night/early morning)
- **Velocity** — Multiple transactions in short time windows
- **Deviations** — Transactions far from user's typical behavior

**Key Finding**: The engineered features (time, amount statistics) matter more than the raw PCA features, suggesting that understanding *what* to measure is more important than algorithmic complexity.

### Model Insights

- **Logistic Regression**: Catches more fraud (90%) but with 2.6% false positives
- **XGBoost**: Catches 84% of fraud with 0.025% false positives (14 false alarms on 56,850 legitimate transactions)
- **Trade-off**: In production, precision matters more than recall—false positives have real costs

### Threshold Optimization

Default threshold (0.5) is suboptimal for imbalanced data. By optimizing for business constraints:
- At 0.5 threshold: 84% fraud caught, 0.025% FP rate
- At 0.3 threshold: 90% fraud caught, 0.15% FP rate
- **Recommendation**: Use 0.5 threshold (default) for operational efficiency

## Usage: Scoring New Transactions

```python
import pickle
import pandas as pd

# Load model and config
with open('models/fraud_risk_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_config.pkl', 'rb') as f:
    config = pickle.load(f)

# Prepare new transactions (with same features as training)
new_df = pd.read_csv('new_transactions.csv')

# Scale features
X_new = scaler.transform(new_df[config['all_features']])

# Get fraud probabilities
fraud_probability = model.predict_proba(X_new)[:, 1]

# Risk score (0-100)
risk_scores = (fraud_probability * 100).round(0).astype(int)

# Flag transactions
fraud_flags = risk_scores >= 50  # Adjust threshold as needed

# Results
results = pd.DataFrame({
    'fraud_probability': fraud_probability,
    'risk_score': risk_scores,
    'fraud_flag': fraud_flags
})
```

## Model Performance Details

### XGBoost (Final Model)

**Hyperparameters:**
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: 580 (class imbalance ratio)

**Evaluation on Test Set (56,962 transactions):**
- True Negatives: 56,850 (legitimate correctly identified)
- False Positives: 14 (legitimate flagged as fraud)
- False Negatives: 16 (fraud missed)
- True Positives: 82 (fraud caught)

**Metrics:**
- Accuracy: 99.95% (but misleading due to class imbalance)
- Precision: 85% (85% of flagged transactions are actually fraud)
- Recall: 84% (catches 84% of all fraud)
- F1-Score: 0.845
- ROC-AUC: 0.981 (excellent discrimination)
- PR-AUC: 0.879 (strong precision-recall trade-off)

## Visualizations

The project includes several key visualizations:

1. **ROC & PR Curves** — Model comparison and discrimination ability
2. **Feature Importance** — Which features drive fraud signals
3. **SHAP Values** — How each feature impacts individual predictions
4. **Threshold Optimization** — Fraud detection vs false positive rate
5. **Risk Score Distribution** — Separation between fraud and legitimate
6. **Confusion Matrix** — True/false positives and negatives

## Technical Decisions

### Why XGBoost over Logistic Regression?

While logistic regression is interpretable and catches more fraud (90%), it produces 2.6% false positives. XGBoost's 0.025% false positive rate with 84% fraud detection is operationally superior because:
- False positives create customer friction and support costs
- 14 false alarms per 56,850 legitimate transactions is manageable
- 82 frauds caught per 98 total frauds is strong coverage

### Why Feature Engineering First?

The top predictive signals came from engineered features (time patterns, amount deviations), not raw PCA features. This suggests:
- Domain knowledge (understanding fraud patterns) matters
- Feature quality > algorithmic sophistication
- Starting with smart features is more efficient than chasing model complexity

### Why SMOTE?

With 0.172% fraud rate, standard train/test split leaves ~27 fraud cases in test set. SMOTE on training data balances to 30% minority, allowing the model to learn fraud patterns better while preserving test set distribution for honest evaluation.

## Limitations & Future Work

### Current Limitations
- Dataset is from 2013; fraud patterns may have evolved
- Features are PCA-transformed (anonymized), limiting interpretability of individual features
- Geographic data not available (location-based fraud detection is powerful)
- Real-time scoring performance not benchmarked

### Future Improvements
- **Online learning**: Retrain model regularly as fraud patterns evolve
- **Feature store**: Build systematic pipeline for feature engineering at scale
- **Ensemble methods**: Combine multiple models for robustness
- **Real-time deployment**: API endpoint for transaction scoring
- **Explainability**: Deeper SHAP analysis per fraud type
- **A/B testing**: Validate threshold in production before full deployment

## Results Summary

✅ **98.1% ROC-AUC** — Excellent discrimination between fraud and legitimate  
✅ **84% Fraud Detection** — Catches most actual fraud  
✅ **0.025% False Positive Rate** — Minimal customer impact  
✅ **85% Precision** — High confidence when flagging fraud  
✅ **Production-ready** — Saved model, scaler, and inference pipeline  

## Contact & Attribution

This project uses the **Kaggle Credit Card Fraud Detection Dataset** (available at [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

For questions or improvements, feel free to open an issue or contact me on [LinkedIn](https://linkedin.com/in/nikhil-muthuvenkatesh).

## License

This project is open source and available under the MIT License.

---

**Built with:** Python, scikit-learn, XGBoost, SHAP  
**Dataset:** 284,807 transactions | 0.172% fraud rate  
**Status:** Complete & production-ready
