def preprocess(X_train, X_test=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler

    # ── helpers ──────────────────────────────────────────────────────────────
    CONTINUOUS = ['AGE', 'ANNUALCONTRIBUTION', 'ANNUALCLAIMAMOUNT', 'UNITSTOTAL']
    BINARY     = ['GENDER_M', 'SCHEMETYPE_MEDIUM', 'SCHEMETYPE_PREMIUM',
                  'DIAGNOSIS_HYPERTENSION', 'COVERTYPE_STANDARD',
                  'COMORBIDITY_NO_COMORBIDITY', 'COMPLICATIONDEVELOPMENT_NO_COMPLICATION']

    def engineer(df, log_shift_claim, log_shift_units):
        df = df.copy()

        # ── 1. Skewness fixes (heavy right-skew on ANNUALCLAIMAMOUNT & UNITSTOTAL) ──
        # log1p-style with fitted shift to avoid log(0)
        df['ANNUALCLAIMAMOUNT_LOG'] = np.log1p(
            np.maximum(df['ANNUALCLAIMAMOUNT'] - log_shift_claim, 0))
        df['UNITSTOTAL_LOG']        = np.log1p(
            np.maximum(df['UNITSTOTAL']        - log_shift_units, 0))

        # Mild skew on ANNUALCONTRIBUTION → sqrt transform
        contrib_min = df['ANNUALCONTRIBUTION'].min()
        df['ANNUALCONTRIBUTION_SQRT'] = np.sqrt(
            np.maximum(df['ANNUALCONTRIBUTION'] - contrib_min + 1e-6, 0))

        # ── 2. Ratio / interaction features ──────────────────────────────────
        # Claim-to-contribution ratio (efficiency / loss ratio proxy)
        df['CLAIM_TO_CONTRIBUTION'] = (
            df['ANNUALCLAIMAMOUNT_LOG'] /
            (df['ANNUALCONTRIBUTION_SQRT'] + 1e-6))

        # Claim per unit (severity per service unit)
        df['CLAIM_PER_UNIT'] = (
            df['ANNUALCLAIMAMOUNT_LOG'] /
            (df['UNITSTOTAL_LOG'] + 1e-6))

        # Age × comorbidity interaction (top corr features)
        df['AGE_x_COMORBIDITY'] = df['AGE'] * df['COMORBIDITY_NO_COMORBIDITY']

        # Age × no-complication interaction
        df['AGE_x_NO_COMPLICATION'] = (
            df['AGE'] * df['COMPLICATIONDEVELOPMENT_NO_COMPLICATION'])

        # Premium scheme × age (risk segmentation)
        df['PREMIUM_x_AGE'] = df['SCHEMETYPE_PREMIUM'] * df['AGE']

        # Hypertension × comorbidity (clinical risk stack)
        df['HYPERTENSION_x_COMORBIDITY'] = (
            df['DIAGNOSIS_HYPERTENSION'] * df['COMORBIDITY_NO_COMORBIDITY'])

        # Units × comorbidity (utilisation risk)
        df['UNITS_x_COMORBIDITY'] = (
            df['UNITSTOTAL_LOG'] * df['COMORBIDITY_NO_COMORBIDITY'])

        # Age squared (non-linear age effect)
        df['AGE_SQ'] = df['AGE'] ** 2

        return df

    # ── fit-time statistics (on X_train only) ────────────────────────────────
    log_shift_claim = X_train['ANNUALCLAIMAMOUNT'].min()
    log_shift_units = X_train['UNITSTOTAL'].min()

    # ── engineer features ────────────────────────────────────────────────────
    X_train_eng = engineer(X_train, log_shift_claim, log_shift_units)

    # ── columns to scale (continuous originals + engineered continuous) ───────
    scale_cols = [
        'AGE', 'ANNUALCONTRIBUTION', 'ANNUALCLAIMAMOUNT', 'UNITSTOTAL',
        'ANNUALCLAIMAMOUNT_LOG', 'UNITSTOTAL_LOG', 'ANNUALCONTRIBUTION_SQRT',
        'CLAIM_TO_CONTRIBUTION', 'CLAIM_PER_UNIT',
        'AGE_x_COMORBIDITY', 'AGE_x_NO_COMPLICATION',
        'PREMIUM_x_AGE', 'UNITS_x_COMORBIDITY', 'AGE_SQ'
    ]

    # ── fit RobustScaler on X_train (robust to outliers in claim/units) ───────
    scaler = RobustScaler()
    X_train_eng[scale_cols] = scaler.fit_transform(X_train_eng[scale_cols])

    if X_test is None:
        return X_train_eng

    # ── apply same transforms to X_test ──────────────────────────────────────
    X_test_eng = engineer(X_test, log_shift_claim, log_shift_units)
    X_test_eng[scale_cols] = scaler.transform(X_test_eng[scale_cols])

    return X_train_eng, X_test_eng
