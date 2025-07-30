# Predicting Heart Disease Presence and Severity with Machine Learning

This project explores the use of machine learning models to predict both the presence and severity of heart disease using clinical patient data.

We evaluated three machine learning models:
- Logistic Regression
- Random Forest
- XGBoost

on two classification tasks:
1. Binary classification – Does the patient have heart disease?
2. Multiclass classification – What is the severity of disease (0–4)?

<br>

**Dataset**

We used the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from the UCI Machine Learning Repository, specifically the Cleveland subset.

The dataset contains:
- 303 patient records
- 13 clinical features (e.g., age, sex, chest pain type, cholesterol, thalassemia)
- A target variable `num` representing disease severity on a scale from 0 (no disease) to 4 (most severe)

Data is loaded directly in the notebook using:
```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
