import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

import ast
import json

class Config(): 
    model_path = 'doublecringe123/job-features-extractor'
    model_revision = None # latest
    tokenizer_path = 'doublecringe123/job-features-extractor'

    max_length = 768

    default_train_args = dict(
        push_to_hub=True, 
        # strategies 
        save_strategy="epoch", 
        eval_strategy="epoch", 
        
        # batch size 
        auto_find_batch_size=True, 
        num_train_epochs=1,
        
        # optimizer 
        weight_decay=1e-3, 
        learning_rate=5e-4, 
        warmup_steps=100, 
        
        save_total_limit=1, 
        seed=42, 
    )

class FeatureExtractor():
    def __init__(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.tokenizer_path,
            revision=Config.model_revision
        )

        # Check if CUDA (GPU) is available and decide on the precision
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.int8

        # Load the model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.model_path,
            revision=Config.model_revision,
            torch_dtype=torch_dtype,
            device_map='auto'
        ).to(device)

        self.device = device

    def __call__(self, inputs): 
        # Tokenize the inputs
        inputs = self.tokenizer(
            inputs, 
            max_length=Config.max_length, 
            truncation=True, 
            padding='longest', 
            return_tensors="pt"
        ).to(self.device)

        # Generate features using the model
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode the generated tokens to text
        features = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return features

# Example usage:
# extractor = FeatureExtractor()
# features = extractor("Your input text here")
# print(features)
