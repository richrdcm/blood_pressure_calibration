import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class SubjectCalibrator:
    @staticmethod
    def calibrate_patient(df_features, patient_id):
        """
        Calculates Alpha (Stiffness) and Beta0 (Baseline) for a patient.
        Maps ln(PTT) = Beta1 * SBP + Beta0
        """
        # Filter for high-quality beats (SQI > 80)
        patient_data = df_features[
            (df_features['patient_id'] == patient_id) &
            (df_features['sqi_score'] > 80)
            ].copy()

        if len(patient_data) < 15:
            return None

        # Moens-Korteweg Linearized Form: ln(PTT)
        X = patient_data[['label_sbp']].values
        y = np.log(patient_data['ptt_actual'].values)

        model = LinearRegression()
        model.fit(X, y)

        # alpha = -2 * slope (Based on MK derivation)
        alpha = -2 * model.coef_[0]

        return {
            'patient_id': patient_id,
            'alpha': round(alpha, 5),
            'intercept': round(model.intercept_, 4),
            'r_squared': round(model.score(X, y), 3)
        }