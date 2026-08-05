import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

def preprocess(X_train, X_test=None):
    """
    Preprocess medical claims data with skewness correction, feature engineering,
    and scaling. Fits transformers on X_train only.
    """
    # Create copies to avoid modifying original data
    X_train_processed = X_train.copy()
    if X_test is not None:
        X_test_processed = X_test.copy()
    
    # Feature engineering: Create ratio features
    X_train_processed['CONTRIBUTION_CLAIM_RATIO'] = X_train_processed['ANNUALCONTRIBUTION'] / (X_train_processed['ANNUALCLAIMAMOUNT'] + 1)
    X_train_processed['UNITS_PER_CLAIM'] = X_train_processed['UNITSTOTAL'] / (X_train_processed['ANNUALCLAIMAMOUNT'] + 1)
    
    if X_test is not None:
        X_test_processed['CONTRIBUTION_CLAIM_RATIO'] = X_test_processed['ANNUALCONTRIBUTION'] / (X_test_processed['ANNUALCLAIMAMOUNT'] + 1)
        X_test_processed['UNITS_PER_CLAIM'] = X_test_processed['UNITSTOTAL'] / (X_test_processed['ANNUALCLAIMAMOUNT'] + 1)
    
    # Identify continuous features for transformation
    continuous_features = ['AGE', 'ANNUALCONTRIBUTION', 'ANNUALCLAIMAMOUNT', 'UNITSTOTAL', 
                          'CONTRIBUTION_CLAIM_RATIO', 'UNITS_PER_CLAIM']
    
    # Apply Yeo-Johnson transformation to handle skewness
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_processed[continuous_features] = power_transformer.fit_transform(X_train_processed[continuous_features])
    
    if X_test is not None:
        X_test_processed[continuous_features] = power_transformer.transform(X_test_processed[continuous_features])
    
    # Standardize all features (including binary and transformed continuous)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test_processed)
        return X_train_scaled, X_test_scaled
    else:
        return X_train_scaled
