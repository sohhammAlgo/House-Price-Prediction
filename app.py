from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn

app = FastAPI(
    title="California House Price Prediction API",
    version="1.0.0"
)

model = joblib.load("model/model.pkl")

class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API Running"
    }

@app.get("/health")
def health():
    return {
        "message": "Healthy"
    }

@app.post("/predict")
def predict(data: HouseData):
    rooms_per_bedroom = (data.AveRooms / data.AveBedrms)
    income_per_occupant = (data.MedInc / data.AveOccup)
    population_density = (data.Population / data.AveOccup)
    population_log = np.log1p(data.Population)
    aveoccup_log = np.log1p(data.AveOccup)

    features = pd.DataFrame(
        [[
            data.MedInc,
            data.HouseAge,
            data.AveRooms,
            data.AveBedrms,
            data.Population,
            data.AveOccup,
            data.Latitude,
            data.Longitude,
            rooms_per_bedroom,
            income_per_occupant,
            population_density,
            population_log,
            aveoccup_log
        ]],
        columns=[
            'MedInc',
            'HouseAge',
            'AveRooms',
            'AveBedrms',
            'Population',
            'AveOccup',
            'Latitude',
            'Longitude',
            'RoomsPerBedroom',
            'IncomePerOccupant',
            'PopulationDensity',
            'Population_log',
            'AveOccup_log'
        ]
    )

    prediction = model.predict(features)[0]
    return {
        "predicted_house_price": round(
            float(prediction),
            4
        )
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )