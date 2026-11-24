"""Train and test splitting for aligned feature, target, and metadata rows."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def test_train_80_20(X, y, test_size=0.2, random_seed=42):
    """Split aligned feature and target rows in their existing order.

    Args:
        X: Feature rows.
        y: Target rows aligned with X.
        test_size: Fraction or count of rows assigned to the test set.
        random_seed: NumPy seed to set when supplied. Rows are unshuffled.

    Returns:
        Training features, targets, and a None metadata placeholder,
        followed by test features, targets, and another None placeholder.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    _ = None
    return X_train, y_train, _, X_test, y_test, _


def stratified_train_test_split_80_20(X, y, meta_columns, test_size=0.2, random_seed=42):
    """Split aligned rows while stratifying by metadata values.

    Args:
        X: Feature rows.
        y: Target rows aligned with X.
        meta_columns: Aligned metadata. DataFrame rows are converted to
            strings joined with underscores to form strata.
        test_size: Fraction or count of rows assigned to the test set.
        random_seed: Random state passed to scikit-learn for shuffling.

    Returns:
        Training features, targets, and metadata, followed by test
        features, targets, and metadata.

    Raises:
        ValueError: The split size or stratum counts prevent stratification.
    """
    # Print the types of X, y, and meta_columns
    print(f"Type of X: {type(X)}")
    print(f"Type of y: {type(y)}")
    print(f"Type of meta_columns: {type(meta_columns)}")

    if isinstance(meta_columns, pd.DataFrame):
        stratify = meta_columns.astype(str).apply('_'.join, axis=1)
    else:
        stratify = meta_columns

    # Print the type of stratify
    print(f"Type of stratify: {type(stratify)}")

    # Print the unique values of the stratified column
    print("Unique values for stratification:", stratify.unique())

    # Perform the train-test split on X, y, and meta_columns
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta_columns, test_size=test_size, random_state=random_seed, stratify=stratify
    )

    return X_train, y_train, meta_train, X_test, y_test, meta_test

