def preprocess(X_train, X_test=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    def _ensure_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X).copy()

    X_train = _ensure_dataframe(X_train)
    X_test_df = _ensure_dataframe(X_test) if X_test is not None else None

    expected_cols = [
        'AGE',
        'ANNUALCONTRIBUTION',
        'ANNUALCLAIMAMOUNT',
        'UNITSTOTAL',
        'GENDER_M',
        'SCHEMETYPE_MEDIUM',
        'SCHEMETYPE_PREMIUM',
        'DIAGNOSIS_HYPERTENSION',
        'COVERTYPE_STANDARD',
        'COMORBIDITY_NO_COMORBIDITY',
        'COMPLICATIONDEVELOPMENT_NO_COMPLICATION'
    ]

    for col in expected_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if X_test_df is not None and col not in X_test_df.columns:
            X_test_df[col] = 0

    X_train = X_train[expected_cols]
    if X_test_df is not None:
        X_test_df = X_test_df[expected_cols]

    def _transform(df):
        df = df.copy()

        # Ensure numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Continuous features
        df['AGE'] = df['AGE'].astype(float)
        df['ANNUALCONTRIBUTION'] = df['ANNUALCONTRIBUTION'].astype(float)
        df['ANNUALCLAIMAMOUNT'] = df['ANNUALCLAIMAMOUNT'].astype(float)
        df['UNITSTOTAL'] = df['UNITSTOTAL'].astype(float)

        # Fix skewness
        # Moderate negative skew
        ac_max = df['ANNUALCONTRIBUTION'].max()
        df['ANNUALCONTRIBUTION_reflect_log1p'] = np.log1p(np.maximum(ac_max - df['ANNUALCONTRIBUTION'], 0))

        # Extreme positive skew
        df['ANNUALCLAIMAMOUNT_log1p'] = np.log1p(np.clip(df['ANNUALCLAIMAMOUNT'], a_min=0, a_max=None))
        df['UNITSTOTAL_log1p'] = np.log1p(np.clip(df['UNITSTOTAL'], a_min=0, a_max=None))

        # Feature engineering: ratios / efficiencies / burden measures
        eps = 1e-6
        df['CLAIM_TO_CONTRIBUTION_RATIO'] = df['ANNUALCLAIMAMOUNT'] / (df['ANNUALCONTRIBUTION'] + eps)
        df['CLAIM_TO_CONTRIBUTION_RATIO_log1p'] = np.log1p(np.clip(df['CLAIM_TO_CONTRIBUTION_RATIO'], a_min=0, a_max=None))

        df['CLAIM_PER_UNIT'] = df['ANNUALCLAIMAMOUNT'] / (df['UNITSTOTAL'] + 1.0)
        df['CLAIM_PER_UNIT_log1p'] = np.log1p(np.clip(df['CLAIM_PER_UNIT'], a_min=0, a_max=None))

        df['CONTRIBUTION_PER_UNIT'] = df['ANNUALCONTRIBUTION'] / (df['UNITSTOTAL'] + 1.0)
        df['CONTRIBUTION_PER_UNIT_log1p'] = np.log1p(np.clip(df['CONTRIBUTION_PER_UNIT'], a_min=0, a_max=None))

        # Utilization / burden interactions
        df['AGE_x_COMORBIDITY_NO_COMORBIDITY'] = df['AGE'] * df['COMORBIDITY_NO_COMORBIDITY']
        df['AGE_x_DIAGNOSIS_HYPERTENSION'] = df['AGE'] * df['DIAGNOSIS_HYPERTENSION']
        df['AGE_x_COMPLICATION_NO_COMPLICATION'] = df['AGE'] * df['COMPLICATIONDEVELOPMENT_NO_COMPLICATION']

        df['PREMIUM_x_CLAIMRATIO'] = df['SCHEMETYPE_PREMIUM'] * df['CLAIM_TO_CONTRIBUTION_RATIO_log1p']
        df['MEDIUM_x_CLAIMRATIO'] = df['SCHEMETYPE_MEDIUM'] * df['CLAIM_TO_CONTRIBUTION_RATIO_log1p']
        df['STANDARD_x_CLAIMRATIO'] = df['COVERTYPE_STANDARD'] * df['CLAIM_TO_CONTRIBUTION_RATIO_log1p']

        df['COMORBIDITY_x_CLAIMRATIO'] = df['COMORBIDITY_NO_COMORBIDITY'] * df['CLAIM_TO_CONTRIBUTION_RATIO_log1p']
        df['HYPERTENSION_x_CLAIMRATIO'] = df['DIAGNOSIS_HYPERTENSION'] * df['CLAIM_TO_CONTRIBUTION_RATIO_log1p']
        df['COMPLICATION_x_CLAIMRATIO'] = df['COMPLICATIONDEVELOPMENT_NO_COMPLICATION'] * df['CLAIM_TO_CONTRIBUTION_RATIO_log1p']

        # Binary combination interactions
        df['PREMIUM_x_COMORBIDITY'] = df['SCHEMETYPE_PREMIUM'] * df['COMORBIDITY_NO_COMORBIDITY']
        df['MEDIUM_x_COMORBIDITY'] = df['SCHEMETYPE_MEDIUM'] * df['COMORBIDITY_NO_COMORBIDITY']
        df['HYPERTENSION_x_COMORBIDITY'] = df['DIAGNOSIS_HYPERTENSION'] * df['COMORBIDITY_NO_COMORBIDITY']
        df['COMPLICATION_x_COMORBIDITY'] = df['COMPLICATIONDEVELOPMENT_NO_COMPLICATION'] * df['COMORBIDITY_NO_COMORBIDITY']

        # Optional mild nonlinear age effects
        df['AGE_SQ'] = df['AGE'] ** 2

        # Replace any inf/nan produced by edge cases
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)

        return df

    X_train_t = _transform(X_train)
    X_test_t = _transform(X_test_df) if X_test_df is not None else None

    # Scale continuous and engineered numeric features; keep original binary indicators unchanged
    original_binary_cols = [
        'GENDER_M',
        'SCHEMETYPE_MEDIUM',
        'SCHEMETYPE_PREMIUM',
        'DIAGNOSIS_HYPERTENSION',
        'COVERTYPE_STANDARD',
        'COMORBIDITY_NO_COMORBIDITY',
        'COMPLICATIONDEVELOPMENT_NO_COMPLICATION'
    ]

    scale_cols = [c for c in X_train_t.columns if c not in original_binary_cols]

    scaler = StandardScaler()
    X_train_t[scale_cols] = scaler.fit_transform(X_train_t[scale_cols])

    if X_test_t is not None:
        X_test_t[scale_cols] = scaler.transform(X_test_t[scale_cols])
        X_test_t = X_test_t[X_train_t.columns]
        return X_train_t, X_test_t

    return X_train_t
