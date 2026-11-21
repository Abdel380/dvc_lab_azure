from pathlib import Path

import skops.io as sio
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Create the FastAPI application
app = FastAPI(title="Random Forest Prediction API")

# 2. Load the model downloaded from DagsHub
MODEL_PATH = Path(__file__).resolve().parent / "model"
model = sio.load(
    MODEL_PATH / "model.skops",
    trusted=["sklearn.tree._tree.Tree"]
)

# 3. Define the input data structure
class PredictionInput(BaseModel):
    pixels: list[float] = Field(
        min_length=64,
        max_length=64
    )


# 4. Create the prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    # Predict the handwritten digit
    prediction = model.predict([data.pixels])

    return {
        "prediction": int(prediction[0])
    }