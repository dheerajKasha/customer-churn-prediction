import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_df(df):
    """Core preprocessing logic mirroring src/preprocessing.py"""
    df = df.copy()
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    return df


@pytest.fixture
def sample_df():
    """Synthetic Telco-like DataFrame with 20 rows."""
    np.random.seed(42)
    n = 20
    data = {
        'customerID': [f'CUST-{i:04d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n),
        'tenure': np.random.randint(0, 72, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n
        ),
        'MonthlyCharges': np.round(np.random.uniform(18.0, 118.0, n), 2),
        'TotalCharges': [
            str(np.round(np.random.uniform(18.0, 8500.0), 2)) if i != 5 else ' '
            for i in range(n)
        ],
        'Churn': np.random.choice(['Yes', 'No'], n),
    }
    return pd.DataFrame(data)


def test_customer_id_dropped(sample_df):
    """customerID must not appear in result columns after preprocessing."""
    result = preprocess_df(sample_df)
    assert 'customerID' not in result.columns


def test_churn_binary(sample_df):
    """All values in the Churn column must be 0 or 1 after preprocessing."""
    result = preprocess_df(sample_df)
    assert set(result['Churn'].unique()).issubset({0, 1})


def test_total_charges_numeric(sample_df):
    """TotalCharges must be float64 after preprocessing."""
    result = preprocess_df(sample_df)
    assert result['TotalCharges'].dtype == np.float64


def test_no_nulls(sample_df):
    """Preprocessed DataFrame must contain no null values."""
    result = preprocess_df(sample_df)
    assert result.isnull().sum().sum() == 0


def test_no_object_columns(sample_df):
    """No object-dtype columns should remain after get_dummies encoding."""
    result = preprocess_df(sample_df)
    object_cols = result.select_dtypes(include=['object']).columns.tolist()
    assert len(object_cols) == 0, f"Object columns remain: {object_cols}"


def test_churn_column_preserved(sample_df):
    """The Churn column must still exist in the preprocessed DataFrame."""
    result = preprocess_df(sample_df)
    assert 'Churn' in result.columns


def test_row_count_preserved(sample_df):
    """The number of rows must not change during preprocessing."""
    result = preprocess_df(sample_df)
    assert len(result) == len(sample_df)


def test_train_test_split_ratio(sample_df):
    """Train split should be ~80% and test split ~20%, within 5% tolerance."""
    result = preprocess_df(sample_df)
    X = result.drop(columns=['Churn'])
    y = result['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    total = len(result)
    train_ratio = len(X_train) / total
    test_ratio = len(X_test) / total
    assert abs(train_ratio - 0.8) <= 0.05, f"Train ratio {train_ratio:.2f} not within 5% of 0.80"
    assert abs(test_ratio - 0.2) <= 0.05, f"Test ratio {test_ratio:.2f} not within 5% of 0.20"
