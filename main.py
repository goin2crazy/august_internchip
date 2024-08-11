# fast api tools and requirements 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# math tools
import numpy as np 
from sentence_transformers import util

# configs 
import config as cfg
from texts_writer import Config as vacancy_write_cfg

# models 
from feature_extractor import FeatureExtractor
from sorter_model import SortingModel
from texts_writer import VacancyWriterModel

from typing import Dict, List, Union, Any

app = FastAPI()

# Initialize the feature extractor and sorting model
feat = FeatureExtractor(model_path=cfg.model_feat_path, revision=cfg.model_feat_versions)
sort = SortingModel()


class TextInput(BaseModel):
    text: str

class EmbeddingsInput(BaseModel):
    embedding1: List[float]
    embedding2: List[float]

# Endpoint to extract features from text
@app.post("/extract_features")
async def extract_features(input: TextInput):
    features = feat(input.text)
    return features

# Endpoint to extract embeddings of features from text
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


# Initialize the model instance
model = VacancyWriterModel()

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

@app.post("/call/")
async def call_model(request: InferenceRequest):
    try:
        job_features = request.job_features.dict()
        input_text = request.input_text
        result = model(job_features, input_text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forward/")
async def forward_model(request: ForwardRequest):
    try:
        job_features = request.job_features.dict()
        input_text = request.input_text
        gen_config = request.gen_config
        result = model.forward(job_features, input_text, gen_config)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
