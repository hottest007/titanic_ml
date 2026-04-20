import joblib
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))

from src.features.build_features import engineer_features

model = joblib.load("model/titanic_logreg.pkl")

def predict(data: dict):
    df = pd.DataFrame([data])
    
    df = engineer_features(df)  # reuse same logic
    
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }