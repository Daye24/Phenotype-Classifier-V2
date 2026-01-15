"""
Utility functions for data loading, cleaning, and preprocessing.
"""
import pandas as pd
import joblib


def load_data(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def clean_data(df):
    """Fill missing values with median for numeric columns."""
    return df.fillna(df.median(numeric_only=True))


def encode_labels(df, column='label'):
    """Encode categorical labels to numeric codes."""
    df = df.copy()
    df[column] = df[column].astype('category').cat.codes
    return df


def split_data(df, target='label', test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def save_model(model, path='models/model.pkl'):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path='models/model.pkl'):
    """Load trained model from disk."""
    return joblib.load(path)
