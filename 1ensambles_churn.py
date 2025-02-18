#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Churn Prediction using Decision Tree, Bagging, and AdaBoost

This script reads the 'CustomerData_Composite-5.csv' dataset, preprocesses the data,
trains three classifiers (Decision Tree, Bagging, AdaBoost), and evaluates their performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import dmba  # For classificationSummary

import warnings

warnings.filterwarnings("ignore")


# -------------------------
# Function: Load Dataset
# -------------------------
def load_data(file_path):
    """
    Loads the dataset from a CSV file.

    :param file_path: str - Path to the dataset CSV file.
    :return: pd.DataFrame - Loaded dataset.
    """
    df = pd.read_csv(file_path)
    print("Dataset Loaded. Shape:", df.shape)
    print(df.head())
    return df


# -------------------------
# Function: Preprocess Data
# -------------------------
def preprocess_data(
    df, selected_features, categorical_features, target_column="churn_value"
):
    """
    Prepares the dataset by selecting relevant features and encoding categorical values.

    :param df: pd.DataFrame - Raw dataset.
    :param selected_features: list - Features to keep in the model.
    :param categorical_features: list - Categorical features to encode.
    :param target_column: str - Target column for classification.
    :return: tuple (X, y) - Processed feature matrix and target array.
    """
    df = df.copy()

    # Convert categorical Yes/No values to 1/0
    for col in categorical_features:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Define features and target variable
    X = df[selected_features]
    y = df[target_column]

    return X, y


# -------------------------
# Function: Train and Evaluate Model
# -------------------------
def train_and_evaluate(model, X_train, y_train, X_valid, y_valid, model_name):
    """
    Trains a model and evaluates its performance.

    :param model: sklearn model instance - Classifier to train.
    :param X_train: pd.DataFrame - Training feature matrix.
    :param y_train: pd.Series - Training labels.
    :param X_valid: pd.DataFrame - Validation feature matrix.
    :param y_valid: pd.Series - Validation labels.
    :param model_name: str - Name of the model for output display.
    """
    model.fit(X_train, y_train)
    print(f"\n{model_name} Model")
    dmba.classificationSummary(y_valid, model.predict(X_valid))


# -------------------------
# Main Execution
# -------------------------
def main():
    # File path
    file_path = "data/CustomerData_Composite-5.csv"

    # Define selected features and categorical columns for encoding
    selected_features = [
        "age",
        "under_30",
        "senior_citizen",
        "partner",
        "dependents",
        "number_of_dependents",
        "married",
        "phone_service",
        "internet_service",
        "monthly_ charges",
        "tenure",
        "satisfaction_score",
    ]

    categorical_features = [
        "under_30",
        "senior_citizen",
        "partner",
        "dependents",
        "married",
        "phone_service",
        "internet_service",
    ]

    # Load and preprocess data
    df = load_data(file_path)
    X, y = preprocess_data(df, selected_features, categorical_features)

    # Split dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=3
    )

    # Define models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=1),
        "Ensemble Bagging": BaggingClassifier(
            DecisionTreeClassifier(random_state=3), n_estimators=120, random_state=3
        ),
        "Adaptive Boosting": AdaBoostClassifier(
            DecisionTreeClassifier(random_state=3), n_estimators=120, random_state=3
        ),
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate(model, X_train, y_train, X_valid, y_valid, model_name)


# Run the main function
if __name__ == "__main__":
    main()
