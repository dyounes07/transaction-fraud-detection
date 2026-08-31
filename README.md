## Project Overview

This project implements an end-to-end credit card fraud detection system using
two machine learning models:

- Logistic Regression as a baseline model
- XGBoost as the main fraud detection model

The project also includes a Flask API that serves predictions from the trained
XGBoost model.

The workflow includes:

- Dataset exploration
- Visualization of transaction and class distributions
- Model training
- Model evaluation
- Saving trained model artifacts
- Real-time fraud predictions through a REST API
- Automated API testing

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Flask
- Requests
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook


## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
(ULB Machine Learning Group) — 284,807 anonymized European credit card 
transactions from September 2013, with 492 labeled as fraud (**0.17%** of 
all transactions).


## Model Performance

Using a stratified 80/20 train-test split and balanced logistic regression:

- **Fraud recall:** 89.80%
- **Average Precision (PR-AUC):** 0.7293
- **ROC-AUC:** 0.9713
- **Fraud precision:** 5.41%
- **Fraud F1-score:** 10.21%

The model prioritizes detecting fraudulent transactions, achieving high fraud recall. However, its relatively low fraud precision means that many flagged transactions are false positives. Because fraud is highly imbalanced in this dataset, PR-AUC and fraud-class recall are more meaningful than accuracy alone.

### XGBoost Results

The XGBoost fraud detector achieved the following test-set results:

- Accuracy: 99.96%
- Fraud precision: 92.13%
- Fraud recall: 83.67%
- Fraud F1-score: 87.70%

The model correctly identified approximately 84% of fraudulent transactions while maintaining a fraud-alert precision of approximately 92%. Because the dataset is highly imbalanced, fraud recall, precision, F1-score, PR-AUC, and ROC-AUC are more informative than accuracy alone.

