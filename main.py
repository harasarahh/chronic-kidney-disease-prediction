# load the libraries
import json
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from enum import Enum
from typing import Dict

# load the data (the final model and the scaler configuration required to normalize the input data)
model = joblib.load("lr_model.pkl")

with open("scaler_config.json", "r") as f:
    config = json.load(f)
    SCALER_MEANS = config["means"]
    SCALER_STDS = config["stds"]

# Exact order required by the model
MODEL_COLUMNS = [
    'age', 'bp', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 
    'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 
    'sg_1.005', 'sg_1.01', 'sg_1.015', 'sg_1.02', 'sg_1.025', 
    'al_0.0', 'al_1.0', 'al_2.0', 'al_3.0', 'al_4.0', 'al_5.0', 
    'su_0.0', 'su_1.0', 'su_2.0', 'su_3.0', 'su_4.0', 'su_5.0'
]

# initialize API
app = FastAPI()

# Binary yes/no values
class YesNo(str, Enum):
    yes, no = "yes", "no"

# Normal vs abnormal values
class NormalAbnormal(str, Enum):
    normal, abnormal = "normal", "abnormal"

# Present vs not present values
class PresentNotPresent(str, Enum):
    present, notpresent = "present", "notpresent"

# Appetite values
class AppetiteEnum(str, Enum):
    good, poor = "good", "poor"

# Specific gravity values
class SGEnum(float, Enum):
    v1005, v1010, v1015, v1020, v1025 = 1.005, 1.01, 1.015, 1.02, 1.025

# Albumin / sugar levels (0–5)
class LevelEnum(float, Enum):
    v0, v1, v2, v3, v4, v5 = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0

# Define the structure of the API prediction response
class PredictionResponse(BaseModel):
    result: int
    status: str
    data_sent: Dict[str, float] # data normalized that the model receives 


# Use of decorators to define the routes of the endpoints
# We need to connect first backend and frontend
@app.get("/", response_class=HTMLResponse, summary="Root Endpoint", description="Serves the main HTML interface for the CKD Diagnostic Service.")
def read_root():
    """
    Returns the main HTML frontend for the application.
    """
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/Predict/", response_model=PredictionResponse, summary="Predict Chronic Kidney Disease", description="This endpoint uses a Logistic Regression model with manual Z-score normalization to predict the presence of CKD based on clinical features.")
def predict(
    age: float = Form(..., description="**age**: The age of the patient, numeric", examples=[36.0]),
    blood_pressure: float = Form(..., description="**bp**: The blood pressure of the patient, numeric", examples=[80.0]),
    specific_gravity: SGEnum = Form(..., description="**sg**: Urinary specific gravity (maps to internal sg_1.005 to sg_1.025)"),
    albumin: LevelEnum = Form(..., description="**al**: Urine albumin level (maps to internal al_0 to al_5)"),
    sugar: LevelEnum = Form(..., description="**su**: Sugar level (maps to internal su_0 to su_5)"),
    rbc: NormalAbnormal = Form(..., description="**rbc**: Red blood cell status (1: rbc = normal, otherwise 0)"),
    pc: NormalAbnormal = Form(..., description="**pc**: Pus cell level (1: pc = normal, otherwise 0)"),
    pcc: PresentNotPresent = Form(..., description="**pcc**: Pus cell clumps status (1: pcc = present, otherwise 0)"),
    ba: PresentNotPresent = Form(..., description="**ba**: Bacteria status (1: ba = present, otherwise 0)"),
    blood_glucose_random: float = Form(..., description="**bgr**: Blood glucose random level, numeric", examples=[95.0]),
    blood_urea: float = Form(..., description="**bu**: Blood urea level, numeric", examples=[22.0]),
    serum_creatinine: float = Form(..., description="**sc**: Serum creatinine level, numeric", examples=[0.9]),
    sodium: float = Form(..., description="**sod**: Sodium level, numeric", examples=[138.0]),
    potassium: float = Form(..., description="**pot**: Potassium level, numeric", examples=[4.2]),
    hemoglobin: float = Form(..., description="**hemo**: Hemoglobin level, numeric", examples=[14.8]),
    packed_cell_volume: float = Form(..., description="**pcv**: Packed cell volume, numeric", examples=[42.0]),
    white_blood_cell_count: float = Form(..., description="**wbcc**: White blood cell count, numeric", examples=[7800.0]),
    red_blood_cell_count: float = Form(..., description="**rbcc**: Red blood cell count, numeric", examples=[4.9]),
    hypertension: YesNo = Form(..., description="**htn**: Hypertension status (1: htn = yes, otherwise 0)"),
    diabetes_mellitus: YesNo = Form(..., description="**dm**: Diabetes mellitus status (1: dm = yes, otherwise 0)"),
    coronary_artery_disease: YesNo = Form(..., description="**cad**: Coronary artery disease status (1: cad = yes, otherwise 0)"),
    appetite: AppetiteEnum = Form(..., description="**appet**: Appetite status (1: appet = good, otherwise 0)"),
    pedal_edema: YesNo = Form(..., description="**pe**: Pedal edema status (1: pe = yes, otherwise 0)"),
    anemia: YesNo = Form(..., description="**ane**: Anemia status (1: ane = yes, otherwise 0)")
):
    """
    Receives patient clinical data, applies preprocessing and normalization,
    and returns a CKD prediction.
    """
    try:
        # Initialize dictionary with all 38 required features set to 0.0
        raw_data = {col: 0.0 for col in MODEL_COLUMNS}

        # Now we convert the categorical inputs into numerical data
        raw_data.update({
            "age": age, "bp": blood_pressure, "bgr": blood_glucose_random, "bu": blood_urea, "sc": serum_creatinine,
            "sod": sodium, "pot": potassium, "hemo": hemoglobin, "pcv": packed_cell_volume, "wbcc": white_blood_cell_count, "rbcc": red_blood_cell_count,
            "htn": 1 if hypertension.value == "yes" else 0,
            "dm": 1 if diabetes_mellitus.value == "yes" else 0,
            "cad": 1 if coronary_artery_disease.value == "yes" else 0,
            "pe": 1 if pedal_edema.value == "yes" else 0,
            "ane": 1 if anemia.value == "yes" else 0,
            "rbc": 1 if rbc.value == "normal" else 0,
            "pc": 1 if pc.value == "normal" else 0,
            "pcc": 1 if pcc.value == "present" else 0,
            "ba": 1 if ba.value == "present" else 0,
            "appet": 1 if appetite.value == "good" else 0
        })

        # Specific encode for the specific gravity
        raw_data[f"sg_{specific_gravity.value}"] = 1

        # Specific encode for albumin and sugar levels
        raw_data[f"al_{albumin.value}"] = 1
        raw_data[f"su_{sugar.value}"] = 1

        # Normalization using Z-score (z = (x - μ) / σ)
        scaled_input = {}
        for feature in MODEL_COLUMNS:
            val = float(raw_data.get(feature, 0))
            if feature in SCALER_MEANS:
                scaled_input[feature] = (val - SCALER_MEANS[feature]) / SCALER_STDS[feature]
            else:
                scaled_input[feature] = val


        # Save normalized input data into a dataframe (using the exact order required)
        input_df = pd.DataFrame([scaled_input])[MODEL_COLUMNS]

        # Prediction
        prediction = int(model.predict(input_df)[0])

        # PREDICTION OUTPUT
        return {
            "result": prediction, # binary (0/1) classification result
            "status": "CKD" if prediction == 1 else "No CKD", # diagnostic label of the prediction
            "data_sent": scaled_input # normalized values used as input for the model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the server (only when this file is executed directly)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)