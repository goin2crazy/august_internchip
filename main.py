from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 

import config as cfg
from feature_extractor import FeatureExtractor
from sorter_model import SortingModel

from typing import Dict, List, Union

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

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
