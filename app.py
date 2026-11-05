# Import Path to create reliable file paths
from pathlib import Path

# Import joblib to load saved machine learning models
import joblib

# Import pandas to read uploaded CSV files
import pandas as pd

# Import matplotlib to display the confusion matrix
import matplotlib.pyplot as plt

# Import seaborn to create a readable confusion matrix
import seaborn as sns

# Import Streamlit to build the interactive web application
import streamlit as st

# Import the evaluation metrics required by the assignment
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Define the project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Define the directory containing saved models
MODEL_DIR = PROJECT_DIR / "model"

# Define the default test dataset path
DEFAULT_TEST_PATH = PROJECT_DIR / "test_data.csv"

# Define the target column
TARGET_COLUMN = "price_range"

# Define all required feature columns
FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi"
]

# Define the available saved models
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# Configure the Streamlit page
st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# Display the main application title
st.title("📱 Mobile Price Range Classification")

# Display a short description of the application
st.write(
    "This application predicts the price range of mobile phones "
    "using five trained machine learning classification models."
)

# Display the target class definitions
st.info(
    "Price ranges: 0 = Low Cost, 1 = Medium Cost, "
    "2 = High Cost, 3 = Very High Cost"
)

# Load all saved models
@st.cache_resource
def load_models():
    # Create an empty dictionary for loaded models
    loaded_models = {}

    # Load each saved model pipeline
    for model_name, model_filename in MODEL_FILES.items():
        # Build the complete model file path
        model_path = MODEL_DIR / model_filename

        # Load the trained model pipeline
        loaded_models[model_name] = joblib.load(model_path)

    # Return all loaded models
    return loaded_models


# Load the models and handle missing model files
try:
    models = load_models()
except Exception as error:
    st.error(f"Unable to load saved models: {error}")
    st.stop()

# Create a CSV upload control
uploaded_file = st.file_uploader(
    "Upload test data CSV",
    type=["csv"]
)

# Use the uploaded file when available
if uploaded_file is not None:
    # Read the uploaded CSV file
    input_data = pd.read_csv(uploaded_file)

    # Display the selected data source
    st.success("Uploaded test dataset loaded successfully.")

# Otherwise use the saved test dataset
elif DEFAULT_TEST_PATH.exists():
    # Read the default test dataset
    input_data = pd.read_csv(DEFAULT_TEST_PATH)

    # Inform the user that the default test dataset is being used
    st.info("Using the saved test_data.csv file.")

# Stop the application if no dataset is available
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Check whether all required columns are present
required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

# Identify columns missing from the uploaded dataset
missing_columns = [
    column for column in required_columns
    if column not in input_data.columns
]

# Display an error if required columns are missing
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# Select only the required input features
X_input = input_data[FEATURE_COLUMNS]

# Store the actual target values
y_actual = input_data[TARGET_COLUMN]

# Display the uploaded data preview
st.subheader("Dataset Preview")
st.dataframe(input_data.head())

# Allow the user to select one of the five models
selected_model_name = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys())
)

# Retrieve the selected trained model
selected_model = models[selected_model_name]

# Generate predictions using the selected saved model
y_predicted = selected_model.predict(X_input)

# Generate probability predictions for AUC calculation
y_probability = selected_model.predict_proba(X_input)

# Calculate accuracy
accuracy = accuracy_score(y_actual, y_predicted)

# Calculate multiclass one-vs-rest weighted AUC
auc_score = roc_auc_score(
    y_actual,
    y_probability,
    multi_class="ovr",
    average="weighted"
)

# Calculate weighted precision
precision = precision_score(
    y_actual,
    y_predicted,
    average="weighted",
    zero_division=0
)

# Calculate weighted recall
recall = recall_score(
    y_actual,
    y_predicted,
    average="weighted",
    zero_division=0
)

# Calculate weighted F1 score
f1 = f1_score(
    y_actual,
    y_predicted,
    average="weighted",
    zero_division=0
)

# Calculate the multiclass Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_actual, y_predicted)

# Display the evaluation section
st.subheader(f"Evaluation Results: {selected_model_name}")

# Create six columns for the required metrics
metric_columns = st.columns(6)

# Display accuracy
metric_columns[0].metric("Accuracy", f"{accuracy:.4f}")

# Display AUC
metric_columns[1].metric("AUC", f"{auc_score:.4f}")

# Display precision
metric_columns[2].metric("Precision", f"{precision:.4f}")

# Display recall
metric_columns[3].metric("Recall", f"{recall:.4f}")

# Display F1 score
metric_columns[4].metric("F1 Score", f"{f1:.4f}")

# Display MCC
metric_columns[5].metric("MCC", f"{mcc:.4f}")

# Create two columns for the confusion matrix and classification report
left_column, right_column = st.columns(2)

# Display the confusion matrix in the left column
with left_column:
    # Add a section heading
    st.subheader("Confusion Matrix")

    # Calculate the confusion matrix
    matrix = confusion_matrix(y_actual, y_predicted)

    # Create a figure for the confusion matrix
    figure, axis = plt.subplots(figsize=(6, 5))

    # Draw the confusion matrix
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axis
    )

    # Label the chart axes
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("Actual Label")
    axis.set_title(selected_model_name)

    # Display the chart in Streamlit
    st.pyplot(figure)

# Display the classification report in the right column
with right_column:
    # Add a section heading
    st.subheader("Classification Report")

    # Generate the classification report as a dictionary
    report = classification_report(
        y_actual,
        y_predicted,
        output_dict=True,
        zero_division=0
    )

    # Convert the report into a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Display the report
    st.dataframe(report_df.round(4))