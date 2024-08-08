### (RECOMENDED: Change the max_length = 150, n_sent = 1 to  max_length = 768, n_sent = 5 for more generation accuracy) 
### (RECOMENDED): Look or try models example usage in colab notebook in file "Train TextGen model to exract resumes: Example Usage.ipynb" 

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
