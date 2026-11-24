
# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project focuses on identifying fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.17% of all transactions.

The project includes exploratory data analysis (EDA), data cleaning, model development, performance evaluation, and threshold analysis to understand how different classification approaches perform when detecting rare fraudulent transactions.

**Objective:** Compare machine learning models and evaluate their ability to identify fraudulent transactions while balancing precision and recall.

## Dataset

The project uses the Credit Card Fraud Detection dataset available on Kaggle.

- **Total transactions:** 284,807
- **Legitimate transactions:** 284,315
- **Fraudulent transactions:** 492
- **Fraud rate:** Approximately 0.17%
- **Features:** 30 input features and 1 target variable (`Class`)

The dataset contains anonymized features (`V1`–`V28`), along with transaction `Time` and `Amount`.

The target variable:
- `0` – Legitimate transaction
- `1` – Fraudulent transaction

**Note:** The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside the `data/` directory.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Examined dataset structure, data types, and summary statistics.
- Analysed the distribution of fraudulent and legitimate transactions.
- Investigated transaction amount and time distributions.
- Examined feature distributions and relationships.
- Identified the severe class imbalance in the dataset.

### 2. Data Cleaning and Preprocessing

- Checked for missing values.
- Identified and removed duplicate records.
- Separated independent variables and the target variable.
- Applied feature scaling to `Time` and `Amount` for Logistic Regression.
- Used stratified train-test splitting to preserve the fraud-class distribution.

**Cleaning results:**
- Original records: 284,807
- Records after duplicate removal: 283,726
- Duplicate records removed: 1,081
- Missing values: 0

### 3. Model Development

Two classification models were developed and evaluated:

**Logistic Regression**
- Used `StandardScaler` for selected numerical features.
- Applied class weighting to address class imbalance.
- Used a maximum of 1,000 iterations.

**Random Forest Classifier**
- Used 100 decision trees.
- Set maximum tree depth to 12.
- Applied `balanced_subsample` class weighting.
- Used minimum leaf size of 2.

### 4. Model Evaluation

The models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

Accuracy was not treated as the primary evaluation metric because the dataset is highly imbalanced.

## Model Performance

The dataset was divided into training and testing sets using an 80:20 stratified split.

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 0.06 | 0.89 |
| Recall | 0.87 | 0.74 |
| F1-score | 0.11 | 0.80 |
| ROC-AUC | 0.9658 | 0.9708 |
| PR-AUC | 0.6719 | 0.8069 |

**Key observations:**

- Logistic Regression achieved higher fraud recall, identifying a larger proportion of fraudulent transactions, but generated more false positives.
- Random Forest achieved higher precision and F1-score, providing a better balance between detecting fraud and limiting false alarms.
- Random Forest also achieved a higher ROC-AUC and PR-AUC in this experiment.

Based on these evaluation metrics, Random Forest performed better overall at the default classification threshold.

## Threshold Analysis

The Random Forest model was further evaluated at a lower classification threshold to investigate the trade-off between fraud detection and false positives.

At the selected threshold of approximately **0.012**:

- Fraudulent transactions detected: 86 out of 95
- Recall: Approximately 91%
- Precision: Approximately 2%
- Fraudulent transactions missed: 9
- False positives: 4,686

Lowering the threshold increased fraud detection but also substantially increased false positives.

**Important limitation:** The threshold was selected using test-set probabilities. This means the test set was used during threshold selection and is not an untouched final evaluation set. A separate validation set and an untouched test set should be used in a future iteration for a more reliable performance estimate.

## Feature Importance

The Random Forest feature importance analysis identified the following features among the most influential for model predictions:

1. V14
2. V10
3. V12
4. V4
5. V17

These features contributed strongly to the model's predictions in this experiment. Feature importance indicates predictive contribution, not a causal relationship with fraudulent activity.

## Visualizations

The repository includes the following visualizations:

- `feature_importance.png` – Random Forest feature importance.
- `model_comparison.png` – Comparison of model evaluation metrics.
- `threshold_analysis.png` – Analysis of precision and recall at different thresholds.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── (Dataset not included)
│
├── notebooks/
│   └── Exploratory_Data_Analysis.ipynb
│
├── reports/
│
├── src/
│
├── visualizations/
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── threshold_analysis.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Key Learnings

- Understanding and analysing highly imbalanced datasets.
- Applying data cleaning and preprocessing techniques.
- Comparing linear and ensemble classification models.
- Evaluating classification performance beyond accuracy.
- Understanding the precision-recall trade-off.
- Analysing feature importance and classification thresholds.

## Future Improvements

- Use a separate validation set for threshold selection and retain an untouched test set for final evaluation.
- Experiment with SMOTE and other imbalance-handling techniques.
- Explore additional models such as XGBoost and Isolation Forest.
- Perform hyperparameter tuning.
- Evaluate model performance using cross-validation.
- Explore explainability techniques such as SHAP.

## Acknowledgements

Dataset source: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
