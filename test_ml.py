import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def test_model_uses_random_forest():
    """
    Test that train_model returns a RandomForestClassifier.
    Verifies the ML model uses the expected algorithm.
    """
    #Load data
    data = pd.read_csv("data/census.csv")

    #Define categorical features
    cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
    
    #small sample for fast testing
    sample = data.sample(n=100, random_state=42)

    #process data
    X_train, y_train, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    #train model
    model = train_model(X_train, y_train)

    #assert model is random forest'
    assert isinstance(model, RandomForestClassifier), \
        "Model should be a RandomForestClassifier"



def test_compute_model_metrics_returns_valid_range():
    """
    Tests that compute_model_metrics returns values between 0 and 1
    """
    #create sample predictions
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    #compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    #assert all metrics in valid range 0-1
    assert 0 <= precision <= 1, \
        f"Precision should be between 0 and 1, got {precision}"
    assert 0 <= recall <= 1, \
        f"Recall should be between 0 and 1, got {recall}"
    assert 0 <= fbeta <= 1, \
        f"F-beta should be between 0 and 1, got {fbeta}"


def test_data_split_has_expected_size():
    """
    Tests that train/split produces expected sizes (80/20)
    """
    #load data
    data = pd.read_csv("data/census.csv")

    #split data
    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data['salary']
    )

    #assert train and test are dataframes
    assert isinstance(train, pd.DataFrame), \
        "Train set should be a pandas DataFrame"
    assert isinstance(test, pd.DataFrame), \
        "Test set should be a pandas DataFrame"
    
    # Calculate expected sizes
    total_size = len(data)
    expected_test_size = int(total_size * 0.20)
    expected_train_size = total_size - expected_test_size
    
    # Assert sizes are approximately correct (within 1 row due to rounding)
    assert abs(len(train) - expected_train_size) <= 1, \
        f"Train set should be ~80% ({expected_train_size} rows), got {len(train)}"
    assert abs(len(test) - expected_test_size) <= 1, \
        f"Test set should be ~20% ({expected_test_size} rows), got {len(test)}"
