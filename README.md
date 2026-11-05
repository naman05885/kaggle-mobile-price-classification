# Mobile Price Range Classification

An end-to-end multi-class classification project for predicting the price range of a mobile phone from its technical specifications. This project was completed for the M.Tech AIML Machine Learning Assignment 2.

## Project Links

- **GitHub repository:** https://github.com/naman05885/kaggle-mobile-price-classification
- **Live Streamlit application:** https://naman-mobile-price-classification-x.streamlit.app/
- **Kaggle dataset:** https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

## Problem Statement

The objective is to classify a mobile phone into one of four price-range categories using its hardware and technical specifications. This is a supervised multi-class classification problem. The target is not the exact price; `price_range` indicates the category.

| Value | Meaning |
|---:|---|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

## Dataset Description

The project uses the public Kaggle **Mobile Price Classification** dataset.

- **Instances:** 2,000
- **Input features:** 20
- **Target column:** `price_range`
- **Number of classes:** 4
- **Class distribution:** 500 instances per class
- **Feature type:** Numerical and binary indicator features

### Input Features

`battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, and `wifi`.

The dataset satisfies the assignment minimum requirements of at least 12 input features and at least 500 instances.

## Data Preparation

1. The target column `price_range` was separated from the input features.
2. The data was divided into training and testing sets using an 80:20 stratified split.
3. The training set contains 1,600 rows and the test set contains 400 rows.
4. Stratification preserves the class balance; each test class contains 100 instances.
5. Numerical features were standardized using `StandardScaler` where required.
6. A fixed random seed of `42` was used for reproducibility.

## Machine Learning Models

The five classification algorithms required by the assignment were implemented and evaluated on the same test set:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation Metrics

Each model was evaluated using Accuracy, one-vs-Rest ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). Confusion matrices and classification reports were also generated.

## Model Comparison Results

The following results were obtained on the held-out test set of 400 instances. Precision, recall, and F1 are weighted averages.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree Classifier | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| K-Nearest Neighbors | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Gaussian Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Support Vector Machine | 0.8950 | 0.9834 | 0.8969 | 0.8950 | 0.8956 | 0.8602 |

### Result Interpretation

- Logistic Regression achieved the best overall performance, with accuracy and weighted F1 of **0.9650**.
- Logistic Regression achieved the highest MCC, **0.9534**.
- SVM was the second-best model.
- KNN produced the lowest performance in this experiment.
- The neighbouring price categories are comparatively more difficult to distinguish than the extreme categories.

Therefore, Logistic Regression was selected as the best-performing model for this dataset and experimental setup.

## Streamlit Application

The deployed application allows the user to:

1. Upload a CSV test dataset.
2. Preview the uploaded data.
3. Select one of the five saved classification models.
4. View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
5. View the confusion matrix.
6. View the classification report.

The uploaded CSV must contain the 20 input features and the `price_range` target column. The application loads saved models and evaluates the uploaded data; it does not retrain the models during upload.

**Live application:** https://naman-mobile-price-classification-x.streamlit.app/

## Repository Structure

```text
kaggle-mobile-price-classification/
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── test_data.csv                  # Held-out test data
├── model/                         # Saved trained model artifacts
├── notebooks/                     # Jupyter notebooks
├── src/                           # Supporting source code
└── .gitignore                     # Ignored files
```

## Installation and Execution

The project was executed in the BITS Virtual Lab environment.

```bash
# Clone the repository
git clone https://github.com/naman05885/kaggle-mobile-price-classification.git

# Enter the project directory
cd kaggle-mobile-price-classification

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook

# Run the Streamlit application
streamlit run app.py
```

## Reproducibility

- Random state: `42`
- Dataset: Kaggle Mobile Price Classification
- Split: 80% training and 20% testing
- Test instances: 400
- The same held-out test data was used for all five models.

## Assignment Submission Checklist

- [x] Public Kaggle classification dataset used
- [x] At least 12 input features and 500 instances
- [x] Five required classification models implemented
- [x] Accuracy, AUC, Precision, Recall, F1, and MCC reported
- [x] Confusion matrix and classification report generated
- [x] GitHub repository link provided
- [x] Live Streamlit application link provided
- [x] BITS Virtual Lab execution screenshot included in the submitted PDF
- [x] README file included in the repository

## Author

Naman Sharma
