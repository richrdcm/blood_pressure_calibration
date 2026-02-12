import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class SubjectCalibrator:
    @staticmethod
    def calibrate_patient(df_features, patient_id):
        """
        Calculates unique elasticity coefficients (Alpha and Baseline)
        for a specific patient based on their PTT vs. BP relationship.
        """
        # 1. Filter data for the specific patient and high-quality beats
        patient_data = df_features[
            (df_features['patient_id'] == patient_id) &
            (df_features['sqi_score'] > 80)
            ].copy()

        if len(patient_data) < 20:  # Need enough beats for a valid trend
            return None

        # 2. Prepare variables for Log-Linear Regression
        # ln(PTT) = Beta1 * Pressure + Beta0
        X = patient_data[['label_sbp']].values  # We use Systolic for PTT calibration
        y = np.log(patient_data['ptt_actual'].values)

        # 3. Fit the model
        model = LinearRegression()
        model.fit(X, y)

        # 4. Extract Biophysical Coefficients
        beta_1 = model.coef_[0]
        beta_0 = model.intercept_

        # alpha = -2 * beta_1 (derived from Moens-Korteweg)
        alpha = -2 * beta_1
        r_squared = model.score(X, y)  # Measure of "Belief" certainty

        return {
            'patient_id': patient_id,
            'alpha_stiffness': round(alpha, 4),
            'baseline_log_ptt': round(beta_0, 4),
            'model_certainty': round(r_squared, 3),
            'mean_ptt': round(patient_data['ptt_actual'].mean(), 2)
        }