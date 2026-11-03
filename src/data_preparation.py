from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COLUMN = "price_range"

raw_path = Path("data/raw/train.csv")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path)

required_columns = {
    "battery_power", "blue", "clock_speed", "dual_sim", "fc",
    "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
    "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
    "talk_time", "three_g", "touch_screen", "wifi",
    TARGET_COLUMN
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if df.duplicated().sum() > 0:
    raise ValueError("Dataset contains duplicate rows.")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

train_data = X_train.copy()
train_data[TARGET_COLUMN] = y_train

test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test

train_data = train_data.sort_index()
test_data = test_data.sort_index()

train_data.to_csv(processed_dir / "train_data.csv", index=False)
test_data.to_csv(processed_dir / "test_data.csv", index=False)

# Required later for Streamlit upload
test_data.to_csv("test_data.csv", index=False)

print("Data preparation completed successfully.")
print(f"Original dataset shape: {df.shape}")
print(f"Training dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")
print("\nTraining target distribution:")
print(train_data[TARGET_COLUMN].value_counts().sort_index())
print("\nTest target distribution:")
print(test_data[TARGET_COLUMN].value_counts().sort_index())
