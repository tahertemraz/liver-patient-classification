"""
Liver Patient Classification Project 

This project predicts liver disease using machine learning.
The dataset contains medical attributes such as Age, Gender, Bilirubin levels,
Alkaline Phosphotase, SGOT, and Total Protein. The goal is to classify whether
a person is a liver patient (Class 1) or not a liver patient (Class 2).

Project Steps:
1. Load liver patient dataset (.arff)
2. Perform Exploratory Data Analysis (EDA) with graphs
3. Preprocess the data (cleaning, encoding, scaling)
4. Train 3 ML classification models:
      - Logistic Regression
      - K-Nearest Neighbors (KNN)
      - Random Forest
5. Hyperparameter tuning with GridSearchCV
6. Compare final accuracies and show confusion matrix heatmaps
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("TkAgg")

import seaborn as sns

from scipy.io import arff
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Load Liver Patient Dataset
def load_arff_to_dataframe(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Convert byte strings into normal strings
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode("utf-8")

    return df, meta



# EDA – Understanding the Liver Patient Data

def perform_eda(df):
    print("===== FIRST 5 PATIENT RECORDS =====")
    print(df.head())

    print("\n===== DATASET INFO =====")
    print(df.info())

    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe(include="all"))

    print("\n===== CLASS DISTRIBUTION (1 = Liver Patient, 2 = Healthy) =====")
    print(df["Class"].value_counts())

    
    # Numerical Feature Histograms
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(figsize=(12, 10))
        plt.tight_layout()
        plt.suptitle("Distribution of Medical Features", y=1.02)
        plt.show()

    
    # Correlation Heatmap
    
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm")
        plt.title("Correlation Between Medical Measurements")
        plt.show()

    
    # Boxplots For Outlier Detection
    
    if len(numeric_cols) > 0:
        plt.figure(figsize=(12, 6))
        df[numeric_cols].boxplot()
        plt.title("Outliers in Liver Patient Features")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



# Data Preprocessing

def preprocess_data(df):
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Label encoding for categorical values (e.g., Gender)
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == object:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])

    # Encode target variable
    y = y.astype(str)
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)

    return X_processed, y_encoded

# 5. Define ML Models & Hyperparameters
def get_models_and_params():
    models = {}

    # Logistic Regression
    pipe_logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    params_logreg = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__solver": ["lbfgs", "liblinear"]
    }
    models["Logistic Regression"] = (pipe_logreg, params_logreg)

    # KNN
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    params_knn = {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    }
    models["KNN"] = (pipe_knn, params_knn)

    # Random Forest
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    params_rf = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10]
    }
    models["Random Forest"] = (pipe_rf, params_rf)

    return models



# Train & Evaluate Models

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models_and_params()
    results = []

    for model_name, (pipeline, param_grid) in models.items():
        print("\n=======================================")
        print(f" Training Model: {model_name}")
        print("=======================================")

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        # Predictions
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print("Best Parameters:", best_params)
        print(f"Validation Accuracy: {best_cv_score:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        results.append({
            "Model": model_name,
            "Best CV Accuracy": best_cv_score,
            "Test Accuracy": test_acc
        })

    # Summary Table
    results_df = pd.DataFrame(results)
    print("\n===== FINAL MODEL COMPARISON =====")
    print(results_df.sort_values(by="Test Accuracy", ascending=False))


#Main Function

def main():
    data_path = "phpOJxGL9.arff"  # liver dataset file

    print("Loading Liver Patient Dataset...")
    df, meta = load_arff_to_dataframe(data_path)

    print("\nDataset Loaded Successfully.")
    print("Rows:", df.shape[0], "Columns:", df.shape[1])
    print("Attributes:", list(df.columns))
    print("Target Column: Class (1 = Liver Patient, 2 = Healthy)")

    perform_eda(df)

    X, y = preprocess_data(df)

    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
