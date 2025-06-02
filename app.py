# app.py
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
import joblib
import pandas as pd # type: ignore
import numpy as np
from typing import List, Dict
import logging
from prometheus_client import Counter, Histogram, generate_latest # type: ignore
from fastapi.responses import PlainTextResponse # type: ignore
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('model_prediction_duration_seconds', 'Prediction latency')

app = FastAPI(title="Sentiment Classification API", version="1.0.0")

# Load model at startup
model = None
vectorizer = None

@app.on_event("startup")
async def load_model():
    global model, vectorizer
    try:
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

class BatchTextInput(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest())

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """Single prediction endpoint"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess text
        text_vector = vectorizer.transform([input_data.text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get class names
        classes = model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        confidence = float(max(probabilities))
        
        # Record metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=prob_dict
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """Batch prediction endpoint"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(input_data.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    start_time = time.time()
    
    try:
        # Preprocess texts
        text_vectors = vectorizer.transform(input_data.texts)
        
        # Make predictions
        predictions = model.predict(text_vectors)
        probabilities = model.predict_proba(text_vectors)
        
        # Get class names
        classes = model.classes_
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            confidence = float(max(probs))
            
            results.append(PredictionResponse(
                prediction=pred,
                confidence=confidence,
                probabilities=prob_dict
            ))
        
        # Record metrics
        PREDICTION_COUNTER.inc(len(input_data.texts))
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return BatchPredictionResponse(predictions=results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sentiment Classification API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)