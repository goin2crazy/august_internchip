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

    default_generation_config = dict(max_length=768, 
                        do_sample=True,
                        top_p = 0.97,
                        top_k = 5,
                        num_beams=2,
                        # bad_words_ids = [tokenizer('\n').input_ids],
                        temperature = 0.4,
                        repetition_penalty=1.2,

                        no_repeat_ngram_size=2,)

    
    default_train_args = dict(
        output_dir = model_path.split('/')[-1], 
        push_to_hub=False, 
        # strategies 
        save_strategy="epoch", 
        eval_strategy="epoch", 
        
        # batch size 
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_strategy = 'epoch', 
        
        num_train_epochs=1,
        
        # optimizer 
        weight_decay=1e-3, 
        learning_rate=5e-4, 
        warmup_steps=100, 
        
        save_total_limit=1,
        load_best_model_at_end=True, 
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
        torch_type = torch.int8 if torch.cuda.is_available() else torch.float16

        # Load the model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.model_path,
            revision=Config.model_revision,
            torch_dtype=torch_type,
            device_map='auto'
        ).to(device)

        self.device = device

    def __call__(self, inputs, generation_config=None): 
        # Tokenize the inputs
        inputs = self.tokenizer(
            inputs, 
            max_length=Config.max_length, 
            truncation=True, 
            padding='longest', 
            return_tensors="pt"
        ).to(self.device)

        # Use default or provided generation config
        gen_config = generation_config if generation_config else Config.default_generation_config

        # Generate features using the model
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)

        # Decode the generated tokens to text
        features = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return features

# Example usage:
# extractor = FeatureExtractor()
# features = extractor("Your input text here")
# print(features)
