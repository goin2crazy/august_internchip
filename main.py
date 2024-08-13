# fast api tools and requirements 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# math tools
import numpy as np 
from sentence_transformers import util

# configs 
import config as cfg
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

# Generative model ------------------------

# Local trained 
feat = FeatureExtractor(model_path=cfg.model_feat_path, revision=cfg.model_feat_versions)
model = VacancyWriterModel()

# Functional by Gemini API 
gemini = GeminiInference() 
feat_gemini = GeminiForFeatureExtraction() 
model_gemini = GeminiForVacancyGeneration() 

# for extracting embedding-----------------
sort = SortingModel()


class TextInput(BaseModel):
    text: str

# Endpoint to extract features from text
@app.post("/extract_features")
async def extract_features(input: TextInput):
    features = feat(input.text)
    return features

# Endpoint to extract features from text
@app.post("/extract_features_with_gemini")
async def extract_features_gemini(input: TextInput):
    features = feat_gemini(input.text)
    return features

# ----------------------------------------------------------

# Endpoint to extract features from text
@app.post("/call_gemini")
async def gemini(input: TextInput):
    model_output = gemini(input.text)
    return model_output

# -----------------------------------------------------------------------------------------
# Initialize the model instance

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
    try:
        job_features = request.job_features.dict()
        input_text = request.input_text
        gen_config = request.gen_config
        result = model.forward(job_features, input_text, gen_config)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------------------------
# Endpoint to ex
# tract embeddings of features from text

class EmbeddingsInput(BaseModel):
    embedding1: List[float]
    embedding2: List[float]

@app.post("/extract_embeddings_of_features")
async def extract_embeddings_of_features(input: TextInput):
    _, embeddings = feat(input.text, sort)
    return {"embeddings": embeddings.tolist()}

# Endpoint to extract both features and embeddings from text
@app.post("/extract_features_and_embeddings")
async def extract_features_and_embeddings(input: TextInput):
    features, embeddings = feat(input.text, sort)
    return {"features": features, "embeddings": embeddings.tolist()}

# Endpoint to extract embeddings directly from text
@app.post("/extract_embeddings")
async def extract_embeddings(input: TextInput):
    embeddings = sort(input.text)
    return {"embeddings": embeddings.tolist()}

# Endpoint to compare two embeddings
@app.post("/compare_embeddings")
def compare_embeddings(self, embedding1, embedding2):
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.numpy().squeeze()


# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
