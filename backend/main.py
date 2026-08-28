from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Stellar Classifier API")

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "stellar_classifier_model.pkl")
le = joblib.load(BASE_DIR / "label_encoder.pkl")


class StellarInput(BaseModel):
    alpha: float
    delta: float
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: float
    spectral_type: str
    galaxy_population: str


@app.get("/")
def root():
    return {"status": "Stellar Classifier API is running"}


@app.post("/predict")
def predict(input: StellarInput):
    input_df = pd.DataFrame([input.dict()])

    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    response = {"predicted_class": pred_label}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        response["probabilities"] = {
            cls: float(p) for cls, p in zip(le.classes_, proba)
        }

    return response