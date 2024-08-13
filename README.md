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
