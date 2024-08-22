# train_api

**There is some important enviroment variables u need to set**

```Linux Ubuntu Terminal 
export "GEMINI_API_KEY"="YOUR GEMINI API KEY"
export "HF_TOKEN"="YOU HUGGING FACE AUTHORIZATED ACCOUNT API TOKEN" 
```

```Windows cmd 
set "GEMINI_API_KEY"="YOUR GEMINI API KEY"
set "HF_TOKEN"="YOU HUGGING FACE AUTHORIZATED ACCOUNT API TOKEN" 
```
Here's an example of how you could write the instructions in the README file for your colleagues, guiding them on how to use the FastAPI service and explaining the recent changes related to resolving the Pydantic warning:

## Getting Started

Ensure you have the following installed:

- Python 3.10 or above
- Pip (Python package installer)
- Git

### Installation

1. Clone the repository:

    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the FastAPI Application

To start the FastAPI server, run:

```bash
uvicorn main:app --reload
```

This will start the server locally on `http://127.0.0.1:8000`.

## API Endpoints

### 1. Train Vacancy Writer Model

**Endpoint:** `/train-vacancy-writer`

**Method:** `POST`

This endpoint is used to initiate the training of the vacancy writer model. Below is the expected request body:

**Request Body:**

```json
{
    "dataframe": {
        "column1": ["value1", "value2"],
        "column2": ["value3", "value4"]
    },
    "dataframe_path": "path/to/data.parquet",
    "tokenizer_preset": "cometrain/neurotitle-rugpt3-small",
    "model_preset": "doublecringe123/job-describtion-copilot-ru",
    "model_revision": null,
    "validation_split": true,
    "block_size": 256,
    "training_args_": {
        "push_to_hub": true,
        "num_train_epochs": 1,
        "warmup_steps": 50,
        "torch_compile": true,
        "auto_find_batch_size": true
    }
}
```

**Response:**

- **Success:** `200 OK` with a message indicating the training has started.
- **Failure:** `500 Internal Server Error` with a detailed error message.

### 2. Train Text2Text Feature Extractor Model

**Endpoint:** `/train-text2text-feature-extractor`

**Method:** `POST`

This endpoint is used to initiate the training of the text-to-text feature extractor model. Below is the expected request body:

**Request Body:**

```json
{
    "dataframe": {
        "column1": ["value1", "value2"],
        "column2": ["value3", "value4"]
    },
    "dataframe_path": "path/to/data.parquet",
    "tokenizer_preset": "cointegrated/rut5-base-multitask",
    "model_preset": "cointegrated/rut5-base-multitask",
    "model_revision": null,
    "validation_split": true,
    "metrics": true,
    "training_args_": {
        "push_to_hub": true,
        "num_train_epochs": 1,
        "warmup_steps": 50,
        "torch_compile": true,
        "auto_find_batch_size": true
    }
}
```

**Response:**

- **Success:** `200 OK` with a message indicating the training has started.
- **Failure:** `500 Internal Server Error` with a detailed error message.

### Handling Protected Namespace Conflicts

We encountered warnings related to the conflict between certain field names (`model_preset`, `model_revision`) and Pydantic's protected namespace `model_`. To resolve this, the `protected_namespaces` configuration has been explicitly disabled in the Pydantic model configurations. This change is included in the `Config` class of each `BaseModel`:

```python
class Config:
    protected_namespaces = ()  # Disable protected namespaces
```

This ensures that the warnings are suppressed, allowing us to continue using field names like `model_preset` and `model_revision` without issues.


**To train feature extractor model (Example)** 

```
! python /kaggle/working/august_internchip/feature_ext_model_train_argparser.py \
--dataframe_path "/kaggle/working/august_internchip/data.csv" --tokenizer_preset 'cointegrated/rut5-base-multitask' \
--model_preset 'cointegrated/rut5-base-multitask'\
--training_args \
num_train_epochs=1 \
warmup_steps=50 \
torch_compile=True \
auto_find_batch_size=True
```

**To train vacancy writer model**

```
check the file vacancy_writer_train_argparser.py
```


# Gemini API for features extraction and vacancy writing 

Using all my skill in prompting and other stuff i added the Gemini API Usage into folder remote_api_models. There also is the python-baseline.ipynb file where i had tested and runned this code. 

There are new functions in main.py the 

```
@app.post("/call_gemini")
async def gemini(input: TextInput)

@app.post("/extract_features_with_gemini")
async def extract_features_gemini(input: TextInput) 

@app.post("/call_gemini_write_vacancy/")
async def call_model_gemini(request: InferenceRequest)
```

- call_gemini - just request the text input to gemini API 


- extract_features_with_gemini - the same as local model for vacancy extracting 


- call_gemini_write_vacancy - the same as Local Vacancy writer model but Main Difference is that **it writes full text of vacancy instead of just giving hints**

## Import Gemini API to enviroment

[Generate own Gemini API key](https://aistudio.google.com/app/apikey)
*If you use free version of Gemini you should probably update the Key sometimes*

**Please set the 'GEMINI_API_KEY' in enviroment with command**

```Linux Ubuntu Terminal 
export "GEMINI_API_KEY"="YOUR GEMINI API KEY"
```

```Windows cmd 
set "GEMINI_API_KEY"="YOUR GEMINI API KEY"
```

# Local Vacancy writer model 

There i added the inference to model i trained for helping HRs writing vacancy text by other vacancy features like 'title', 'salary' and etc. 

## Data

Dataset for train this model was parsed from website hh.taskent.uz, and for saving confidency i replaced company names in this dataset to ''

All Dataset Process i did in google colab, there is saved notebooks of this process: 

- file "hh_web_scrapping.ipynb" - Web scrabbing process 
- file "replacing the compamy names.ipynb" - Using NER model for recognize and then remove company names from dataset describtion column values process

[Dataset is loaded to hugging face hub](https://huggingface.co/datasets/doublecringe123/parsed-vacancies-from-headhunter-tashkent)

## New Model

- file "jobs_writer_usage.ipynb" There u can try model, play with generation config and evaluate ;) 

I trained model on my own parsed dataset, finetuning the 'cometrain/neurotitle-rugpt3-small' as base model, in my [kaggle notebook](https://www.kaggle.com/code/yannchikk/solution-for-my-summer-internship?scriptVersionId=191968661)

Then when i got bad results, i trained model again, but augmented my data, making it size 4 times bigger, until i was [satisfied with results](https://www.kaggle.com/code/yannchikk/solution-for-my-summer-internship?scriptVersionId=192142602)


all model versions saved to my [hugging face hub models](https://huggingface.co/doublecringe123/job-describtion-copilot-ru/)

### Software

    - num epochs = 30, 
    - lr rate = 5e-4, 
    - seed = 42, 
    - fp16 = True, 
    - torch_compile = True, 

### Hardware 
Model trained with transformers Trainer on Kaggle GPU 2x4T

- First train runtime took 53 minutes
- Second train runtime (with augmentation) took 3 hours, 11 minutes

But Model extra ligh, so it easilly runs on CPU (i tried on colab CPU)

#

(RECOMENDED: Change the max_length = 150, n_sent = 1 to  max_length = 768, n_sent = 5 for more generation accuracy) 
(RECOMENDED): Look or try models example usage in colab notebook in file "Train TextGen model to exract resumes: Example Usage.ipynb" 

# Model Version 

in config.py please change feature_extraction (parameter name: model_feat_versions) model version u like 

```python
# f2df1d0b06bce3ac1a7ecf7d9408737efb416118 - mini model, no prompt (Default)
# 00e8801ca3d4314a1d2cb0be101440ce738dd129 - mini model with prompt 
# f67bc736ef24991e646c96b6993c4d44f2d12078 - base model with prompt
# dfcb75554b01fce5ab2360ed9c5d3e6cf49ca12a - base model, no prompt
```

and u can change also SentenceComparing model preset (parameter name: sentence_sorting_model_preset)to other model from huggingface.hub 

```
sentence-transformers/all-MiniLM-L6-v2'
```
(Default)

# Installation 

### Torch (LIght version)

#### GPU (Recommended)

(Depends on cuda version)
CUDA 11.8 
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

CUDA 12.1 
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

CUDA 12.4
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

#### CPU 

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### install other libs 
```
pip install -q requirements.txt
```
