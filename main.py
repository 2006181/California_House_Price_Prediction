import io
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(gt=0, description="Median house age in block group")
    AveRooms: float = Field(gt=0, description="Average number of rooms")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(gt=0, description="Population in block group")
    AveOccup: float = Field(gt=0, description="Average occupancy")
    Latitude: float = Field(gt=0, description="Block group latitude")
    Longitude: float = Field(lt=0, description="Block group longitude")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API",
        "status": "running",
        "endpoint": "Send a POST request to /predict"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }


# Single house prediction
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([house.model_dump()])

        predicted = model.predict(input_data)[0]

        price_usd = predicted * 1_000_000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": (
                f"${predicted * 10:,.2f} hundred thousand"
            ),
            "confidence_range": (
                f"${price_usd - 39000:,.0f} "
                f"to ${price_usd + 39000:,.0f}"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# CSV file prediction
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a CSV file."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )

        predictions = model.predict(df[required_columns])

        # Convert target values to USD
        df["PredictedPrice"] = predictions * 1_000_000

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; filename=prediction.csv"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

