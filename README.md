# Mobile Price Range Classification

## Problem Statement

This project predicts the price range of a mobile phone based on its technical specifications using machine learning classification models.

## Dataset Description

The dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

It contains:

- 2,000 instances
- 20 input features
- 1 target column: `price_range`
- Four classes:
  - 0: Low Cost
  - 1: Medium Cost
  - 2: High Cost
  - 3: Very High Cost

The dataset is balanced, with 500 records in each class.

## Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient

The final comparison table and model observations will be added after completing model evaluation.

## Streamlit Application

The Streamlit application allows users to:

- Upload test data in CSV format
- Select one of the five trained models
- View evaluation metrics
- View a confusion matrix
- View a classification report

## Project Structure

```text
app.py
requirements.txt
README.md
test_data.csv
model/
notebooks/# kaggle-mobile-price-classification
