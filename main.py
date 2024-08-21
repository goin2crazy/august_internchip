# fast api tools and requirements 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# math tools
import numpy as np 
from sentence_transformers import util

from texts_writer import Config as vacancy_write_cfg

# local models 
from feature_extractor import FeatureExtractor
from texts_writer import VacancyWriterModel

# Gemini API models 
from remote_api_models.gemini_inference import GeminiInference
from remote_api_models.gemini_feature_extractor import GeminiForFeatureExtraction
from remote_api_models.gemini_vacancy_writer import GeminiForVacancyGeneration

# For Sentance comparing 
from sorter_model import SortingModel

from typing import Dict, List, Union, Any

app = FastAPI()

# Functional by Gemini API 
gemini = GeminiInference() 
feat_gemini = GeminiForFeatureExtraction() 
model_gemini = GeminiForVacancyGeneration() 

# for extracting embedding-----------------
sort = SortingModel()

# Global variables to manage the state of models
feat = None
model = None

@app.post("/start_local_models")
async def start_local_models():
    global feat, model
    feat = FeatureExtractor()
    model = VacancyWriterModel()
    return {"message": "Local models loaded successfully."}

def check_models_loaded():
    if feat is None or model is None:
        raise HTTPException(status_code=400, detail="Models are not loaded. Please load the models using /start_local_models endpoint.")

class TextInput(BaseModel):
    text: str

# Endpoint to extract features from text
@app.post("/extract_features")
async def extract_features(input: TextInput):
    check_models_loaded()
    features = feat(input.text)
    return str(features)

# Endpoint to extract features from text using Gemini
@app.post("/extract_features_with_gemini")
async def extract_features_gemini(input: TextInput):
    features = feat_gemini(input.text)
    return str(features)

# Endpoint to call Gemini model
@app.post("/call_gemini")
async def gemini(input: TextInput):
    model_output = gemini(input.text)
    return model_output

class JobFeatures(BaseModel):
    title: str
    salary: str
    company: str
    experience: str
    mode: str
    skills: str

class InferenceRequest(BaseModel):
    job_features: JobFeatures
    input_text: str

class ForwardRequest(InferenceRequest):
    gen_config: Dict[str, Any] = vacancy_write_cfg.generation_config

@app.post("/call_local_model_write_vacancy/")
async def call_model(request: InferenceRequest):
    check_models_loaded()
    try:
        job_features = request.job_features.dict()
        input_text = request.input_text
        result = model(job_features, input_text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/call_gemini_write_vacancy/")
async def call_model_gemini(request: InferenceRequest):
    try:
        job_features = request.job_features.dict()
        result = model_gemini(**job_features)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forward_local_model_write_vacancy/")
async def forward_model(request: ForwardRequest):
    check_models_loaded()
    try:
        job_features = request.job_features.dict()
        input_text = request.input_text
        gen_config = request.gen_config
        result = model.forward(job_features, input_text, gen_config)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EmbeddingsInput(BaseModel):
    embedding1: List[float]
    embedding2: List[float]

@app.post("/extract_embeddings")
async def extract_embeddings(input: TextInput):
    embeddings = sort(input.text)
    return {"embeddings": embeddings.tolist()}

@app.post("/compare_embeddings")
async def compare_embeddings(input: EmbeddingsInput):
    cosine_scores = util.cos_sim(input.embedding1, input.embedding2)
    return cosine_scores.numpy().squeeze()

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
