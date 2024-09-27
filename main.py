# fast api tools and requirements 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# math tools
import numpy as np 
from sentence_transformers import util

from texts_writer import Config as vacancy_write_cfg

# Gemini API models 
from remote_api_models.gemini_inference import GeminiInference
from remote_api_models.gemini_feature_extractor import GeminiForFeatureExtraction
from remote_api_models.gemini_vacancy_writer import GeminiForVacancyGeneration
# There is located the class for discrimination detecting 
from remote_api_models.gemini_vacancy_filter import GeminiForVacancyFiltration

# For sentence comparing 
from sorter_model import SortingModel

from typing import Dict, List, Union, Any

app = FastAPI()

# Functional by Gemini API 
gemini = GeminiInference() 
feat_gemini = GeminiForFeatureExtraction() 
model_gemini = GeminiForVacancyGeneration()
vacancy_filter = GeminiForVacancyFiltration()  # Added for discrimination detection

# for extracting embedding-----------------
sort = SortingModel()


class TextInput(BaseModel):
    text: str

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

# Endpoint to extract features from text using Gemini
@app.post("/extract_features_with_gemini")
async def extract_features_gemini(input: TextInput):
    features = feat_gemini(input.text)
    return {"result": str(features)} 

# Endpoint to call Gemini model
@app.post("/call_gemini_directly")
async def gemini(input: TextInput):
    model_output = gemini(input.text)
    return {"result": model_output}


@app.post("/call_gemini_write_vacancy/")
async def call_model_gemini(request: InferenceRequest):
    try:
        job_features = request.job_features.dict()
        result = model_gemini(**job_features)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for detecting discrimination in vacancy text
@app.post("/detect_discrimination_with_gemini/")
async def detect_discrimination(input: TextInput):
    try:
        # Use the GeminiForVacancyFiltration class to detect discrimination
        generated_text = vacancy_filter.prompt(input.text)
        # Extract reasoning and answer using the extract_target_answer method
        result = vacancy_filter.extract_target_answer(generated_text)
        return result
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
