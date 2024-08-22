import os
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from texts_writer.train_process import train as train_vacancy_writer
from feature_extractor.train_process import train as text2text_feature_extractor_train  # Import your training functions

app = FastAPI()

# Define request body model with default values for vacancy writer training
class TrainVacancyWriterRequest(BaseModel):
    dataframe: Optional[Dict[str, list]] = None
    dataframe_path: str = "hf://datasets/doublecringe123/parsed-vacancies-from-headhunter-tashkent/data/train-00000-of-00001.parquet"
    tokenizer_preset: str = "cometrain/neurotitle-rugpt3-small"
    model_preset: str = "doublecringe123/job-describtion-copilot-ru"
    model_revision: str = None
    validation_split: bool = True
    block_size: int = 256
    training_args_: dict = {
        "push_to_hub": True,
        "num_train_epochs": 1,
        "warmup_steps": 50,
        "torch_compile": True,
        "auto_find_batch_size": True
    }

# Define request body model with default values for text2text feature extractor training
class Text2TextFeatureExtractorTrainRequest(BaseModel):
    dataframe: Optional[Dict[str, list]] = None
    dataframe_path: str = "hf://datasets/doublecringe123/parsed-vacancies-from-headhunter-tashkent/data/train-00000-of-00001.parquet"
    tokenizer_preset: str = 'cointegrated/rut5-base-multitask'
    model_preset: str = 'cointegrated/rut5-base-multitask'
    model_revision: str = None
    validation_split: bool = True
    metrics: bool = True
    training_args_: dict = {
        "push_to_hub": True,
        "num_train_epochs": 1,
        "warmup_steps": 50,
        "torch_compile": True,
        "auto_find_batch_size": True
    }

# Define the training endpoint for vacancy writer
@app.post("/train-vacancy-writer")
def train_vacancy_writer_model(request: TrainVacancyWriterRequest):
    try:
        if request.dataframe is not None:
            df = pd.DataFrame(request.dataframe)
            temp_csv_path = "/tmp/temp_dataframe.csv"
            df.to_csv(temp_csv_path, index=False)
            dataframe_path = temp_csv_path
        else:
            dataframe_path = request.dataframe_path
        
        model = train_vacancy_writer(
            dataframe_path=dataframe_path,
            tokenizer_preset=request.tokenizer_preset,
            model_preset=request.model_preset,
            model_revision=request.model_revision,
            validation_split=request.validation_split,
            block_size=request.block_size,
            **request.training_args_,
        )

        try:
            if request.dataframe is not None:
                os.remove(temp_csv_path)
        except Exception as e:
            print(e)

        return {"message": "Vacancy Writer Model training started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the training endpoint for text2text feature extractor
@app.post("/train-text2text-feature-extractor")
def train_text2text_feature_extractor_model(request: Text2TextFeatureExtractorTrainRequest):
    try:
        if request.dataframe is not None:
            df = pd.DataFrame(request.dataframe)
            temp_csv_path = "/tmp/temp_dataframe.csv"
            df.to_csv(temp_csv_path, index=False)
            dataframe_path = temp_csv_path
        else:
            dataframe_path = request.dataframe_path
        
        model = text2text_feature_extractor_train(
            dataframe_path=dataframe_path,
            tokenizer_preset=request.tokenizer_preset,
            model_preset=request.model_preset,
            model_revision=request.model_revision,
            validation_split=request.validation_split,
            metrics=request.metrics,
            **request.training_args_,
        )

        try:
            if request.dataframe is not None:
                os.remove(temp_csv_path)
        except Exception as e:
            print(e)

        return {"message": "Text2Text Feature Extractor Model training started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
