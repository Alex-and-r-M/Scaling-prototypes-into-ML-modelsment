# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Install gcsfs for reading from Google Cloud Storage
!pip install gcsfs

# Load data from Google Cloud Storage
gs_path = "gs://firstbucket44/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(gs_path)

print("Data loaded successfully! First 5 rows:")
print(df.head())

print("\nData Info:")
df.info()

# Convert 'TotalCharges' from object to numeric (handle NaNs)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Drop 'customerID' column
df.drop('customerID', axis=1, inplace=True)

# Replace specific service responses with 'No'
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in service_cols:
    df[col] = df[col].replace({'No internet service': 'No'})

df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

# Map 'Churn' to binary (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nData after preprocessing (first 5 rows):")
print(df.head())
print("\nShape after preprocessing:", df.shape)

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Initialize and train RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
print("\nRandomForestClassifier model trained.")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")
