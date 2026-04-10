from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
from model import predict_performance

# Create FastAPI app
app = FastAPI(
    title="Intern Performance Prediction API",
    description="Predicts intern performance as High, Medium or Low",
    version="1.0.0"
)

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Input data structure
class InternData(BaseModel):
    completion_time  : float
    feedback_rating  : float
    attendance       : float

# ── Endpoints ──────────────────────────────

# Home → HTML Dashboard
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Health Check
@app.get("/health")
async def health():
    return {
        "status" : "API is running!",
        "version": "1.0.0"
    }

# Model Info
@app.get("/model-info")
async def model_info():
    return {
        "model"     : "Random Forest Classifier",
        "accuracy"  : "94%",
        "features"  : 7,
        "trained_on": "1000 records",
        "labels"    : ["High", "Medium", "Low"]
    }

# Single Prediction
@app.post("/predict")
async def predict(data: InternData):
    result = predict_performance(
        data.completion_time,
        data.feedback_rating,
        data.attendance
    )
    return result

# Batch Prediction
@app.post("/predict-batch")
async def predict_batch(data: List[InternData]):
    results = []
    for intern in data:
        result = predict_performance(
            intern.completion_time,
            intern.feedback_rating,
            intern.attendance
        )
        results.append(result)
    return {"total": len(results), "predictions": results}