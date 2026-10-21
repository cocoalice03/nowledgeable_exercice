from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model = joblib.load("regression.joblib")

class HouseFeatures(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.post("/predict")
def predict(features: HouseFeatures):
    # Create input DataFrame
    input_data = pd.DataFrame([[features.size, features.nb_rooms, features.garden]], 
                               columns=['size', 'nb_rooms', 'garden'])
    
    # Predict
    prediction = model.predict(input_data)
    
    return {"y_pred": float(prediction[0])}
