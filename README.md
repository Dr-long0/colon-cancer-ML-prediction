# colon-cancer-ML-prediction
Code and model for Predicting Prevalent Colon Cancer Using Explainable Machine Learning: A Cross-Sectional Study Based on NHANES 2005–2018

- **Model type**: Regularized Logistic Regression (scikit-learn)
- **Algorithm**: `LogisticRegression` from `sklearn.linear_model`
- **Hyperparameters**:
  - `C = 1.0` (L2 regularization)
  - `penalty = 'l2'`
  - `solver = 'lbfgs'`
  - `max_iter = 100`
  - `n_jobs = 10`
  - `fit_intercept = True`

- Source: **NHANES 2005–2016**
- Target: Self-reported diagnosis of **colon cancer**
- Features: 29 variables (demographics, lab data, physical metrics, survey items)
- Preprocessing:
  - Missing data imputed using `missForest` (in R)
  - Feature standardization: Z-score normalization (mean = 0, std = 1)
  - SMOTE used for class balancing during training
  - All transformations were performed **only on training data**


1. Clean and preprocess NHANES data
2. Define binary classification labels (colon cancer vs. control)
3. Standardize continuous features using training fold statistics
4. Apply SMOTE to balance the training set
5. Train LR model with 10-fold cross-validation
6. Evaluate performance on internal validation (n = 9282)
7. Export model using `joblib`

---

| Metric         | Internal Validation |
|----------------|---------------------|
| AUC            | 0.908               |
| Accuracy       | 0.842               |
| Sensitivity    | 0.867               |
| Specificity    | 0.816               |
| PPV            | 0.825               |
| NPV            | 0.860               |
| Brier Score    | 0.158               |

---

## 🚀 How to Use the Model


```bash
pip install -r requirements.txt
import joblib
import pandas as pd

# Load model
model = joblib.load("Logistic Regression.pkl")

# Example data: dataframe with 20 features after preprocessing
X_new = pd.read_csv("your_processed_data.csv")

# Predict probability
y_pred_proba = model.predict_proba(X_new)[:, 1]

# Predict label
y_pred = model.predict(X_new)
