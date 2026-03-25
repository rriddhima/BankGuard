# ============================================================
# BankGuard - Fraud Detection Model
# Step 1: Train and save the ML model
# ============================================================

# --- IMPORTS ---
# pandas: for loading and manipulating our dataset (like Excel but in Python)
# numpy: for math operations
# sklearn: the ML library we use to train our model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # to save/load our trained model
import os

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================
# We use the famous "creditcard.csv" dataset from Kaggle.
# Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it inside the ml/data/ folder.

print("Loading dataset...")

DATA_PATH = "data/creditcard.csv"

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    print("\n Dataset not found!")
    print("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("And place creditcard.csv inside the ml/data/ folder\n")

    # For testing without the dataset, we generate fake data
    print("Generating sample data for testing...")
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame(np.random.randn(n, 28), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = np.random.exponential(scale=100, size=n)
    df["Time"] = np.arange(n)
    # Only 1% fraud (realistic)
    df["Class"] = np.random.choice([0, 1], size=n, p=[0.99, 0.01])
    print(f"Sample data created: {n} transactions\n")
else:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} transactions\n")

# ============================================================
# STEP 2: EXPLORE THE DATA
# ============================================================
# Always look at your data before training!

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Total transactions : {len(df)}")
print(f"Fraudulent         : {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
print(f"Legitimate         : {len(df) - df['Class'].sum()}")
print(f"Features           : {df.shape[1] - 1}")
print()

# ============================================================
# STEP 3: PREPARE FEATURES AND LABELS
# ============================================================
# X = input features (what the model learns from)
# y = output label (0 = not fraud, 1 = fraud)

# We drop "Time" because it's not useful for fraud detection
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Scale the "Amount" column
# Models work better when all numbers are in a similar range
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# ============================================================
# STEP 4: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================
# We train on 80% of data, and test on 20%
# random_state=42 means we get the same split every time (reproducible)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps same fraud ratio in both sets
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}\n")

# ============================================================
# STEP 5: TRAIN THE MODEL
# ============================================================
# Random Forest = many decision trees voting together
# n_estimators=100 means 100 trees
# class_weight='balanced' helps because fraud cases are rare

print("Training Random Forest model...")
print("(This may take 1-2 minutes...)\n")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",  # handles imbalanced data (very little fraud)
    random_state=42,
    n_jobs=-1  # use all CPU cores for speed
)

model.fit(X_train, y_train)
print("Model trained!\n")

# ============================================================
# STEP 6: EVALUATE THE MODEL
# ============================================================
# Check how well the model performs on data it has never seen

y_pred = model.predict(X_test)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# ============================================================
# STEP 7: PLOT CONFUSION MATRIX
# ============================================================
# Shows how many predictions were correct vs wrong

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"])
plt.title("Confusion Matrix - BankGuard Fraud Detector")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png\n")

# ============================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================
# Which features does the model rely on most?

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.plot(kind="bar", color="steelblue")
plt.title("Top 10 Features Used by the Model")
plt.ylabel("Importance Score")
plt.xlabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance chart saved as feature_importance.png\n")

# ============================================================
# STEP 9: SAVE THE MODEL
# ============================================================
# joblib saves the trained model to a file
# Our Flask API will load this file later in Step 2

os.makedirs("../api", exist_ok=True)
joblib.dump(model, "../api/model.pkl")
joblib.dump(scaler, "../api/scaler.pkl")

print("=" * 50)
print("Model saved to: api/model.pkl")
print("Scaler saved to: api/scaler.pkl")
print("=" * 50)
print()
print("Step 1 COMPLETE! Move on to Step 2 (Flask API)")