# CHECK THE ENVIROMENT VARIABLES 
import os 

assert "HF_TOKEN" in os.environ, "Please set HF_TOKEN in os enviroments variables"

# IMPORT LIBS 
import torch 

import datasets 

# load model 
from transformers import AutoTokenizer, AutoModelForCausalLM
# train model 
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# train test split
from sklearn.model_selection import train_test_split

import ast
import json 

import pandas as pd 
import numpy as np

# IMPORT CONFIG 

from .config import Config as cfg
from .train_augmentations import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize(examples, tokenizer):
    return tokenizer([x for x in examples["text"]]) 

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def build_dataset(df, extract_text = None): 
    
    if extract_text == None: 
        extract_text = extract_text_v1
    
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.map(lambda i: extract_text(i))
    
    return ds

def build_dataset_all(df, tokenizer, block_size): 
    
    ds =  datasets.concatenate_datasets([
        build_dataset(df), 
        build_dataset(df, extract_text_v0), 
        build_dataset(df, extract_text_v2), 
        build_dataset(df, extract_text_v3), 
    ])

    ds = ds.map(lambda i: tokenize(i, tokenizer), remove_columns=ds.column_names, batched=True) 
    ds = ds.map(lambda i: group_texts(i, block_size), batched=True) 
    return ds 


def build_data(df, *args, **kwargs): 
    train_df, val_df = train_test_split(df, test_size=0.015, random_state=42)
    
    print('[train]')
    train_ds  = build_dataset_all(train_df, *args, **kwargs)
    print('[val]') 
    val_ds = build_dataset_all(val_df, *args, **kwargs)
    return train_ds, val_ds
    
def run_train(model:torch.nn.Module, 
              tokenizer: AutoTokenizer, 
              ds_train: datasets.Dataset, 
              ds_val: datasets.Dataset = None , 
              **kwargs) -> torch.nn.Module: 
    """

    Arguments: 
            model - Transformers model you wanna train 
            tokenizer - Aokenizer of this model whic needed for build the data collator
            ds_train - Train Dataset for train model on 
            ds_val - Validation Dataset for validate model on 
            **kwargs - Arguments for training

    In kwargs you can put whatever params you want [Look params and explanation from huggingface.docs about TrainingArguments and Train]

    There is example [And arguments i used to train model in first time]: 
        
            params = dict(
                push_to_hub = True, 
                # strategies 
                save_strategy = "epoch", 
                eval_strategy = "epoch", 
                
                # batch size 
                auto_find_batch_size = True, 
                num_train_epochs = 50,
                
                # optimizer 
                weight_decay = 1e-3, 
                learning_rate = 5e-4, 
                warmup_steps = 2000, 
                
                save_total_limit = 1, 
                torch_compile = True, 

                seed=42, 
            )
    """
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir = cfg.model_path, 
        **kwargs, 
        fp16 = torch.cuda.is_available(), 
    )

    trainer = Trainer(
        model, 
        args= training_args, 
        train_dataset = ds_train, 
        eval_dataset = ds_val, 
        data_collator = collator, 
    #     compute_metrics=compute_metrics, 
        ) 

    trainer.train()
    return model

# MAIN TRAINING FUNCTIONS 
def train(
    dataframe_path:str = None, 
    tokenizer_preset:str = None, 
    model_preset:str = None, 
    model_revision: str = None, 
    validation_split=  True, 
    block_size = 256, 
    **training_args_, 
) -> torch.nn.Module: 
    """
    Arguments: 
        dataframe_path:str - Local or Remote path to csv dataset which including columns ['title', 'salary', 'company', 'experience', 'mode', 'skills', 'description'], 
        tokenizer_preset:str - Local or Remote path to AutoTokenizer for model you wanna train , 
        model_preset:str - Local or Remote path to model you wanna train, 
        model_revision: str - if you use the remote path u also can set the version of model with this argument, 
        validation_split - If you also want to use the validation set to validation model, 
        **training_args - arguments you have to set for training, 
    """
    # Load data
    if dataframe_path is None: 
        dataframe_path = cfg.default_dataframe_path
    print(f"Dataframe path: {dataframe_path}")

    print("Loading data...")
    if dataframe_path.endswith("parquet"): 
        df = pd.read_parquet(dataframe_path)
        print("Data loaded from a Parquet file.")
    elif dataframe_path.endswith("csv"): 
        df = pd.read_csv(dataframe_path)
        print("Data loaded from a CSV file.")
    else:
        print("Unrecognized file type.")
        return None
        
    print("WARNING! Dataframe must include the columns ['title', 'salary', 'company', 'experience', 'mode', 'skills', 'description']")
    df = df[['title', 'salary', 'company', 'experience', 'mode', 'skills', 'description']]
    print("Dataframe columns selected.")
    print(df.head())


    # Load tokenizer 
    if tokenizer_preset is None: 
        tokenizer_preset = cfg.tokenizer_path
    print(f"Tokenizer preset: {tokenizer_preset}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_preset)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded and pad_token set.")

    # Build datasets 
    if validation_split: 
        ds_train, ds_val = build_data(df, tokenizer, block_size)
        print("Datasets for training and validation created.")
    else: 
        _, ds_val = build_data(df, tokenizer, block_size)
        ds_train = build_dataset_all(df, tokenizer, block_size)
        print("Dataset for training created.")

    # Load model 
    if model_preset is None: 
        model_preset = cfg.model_path
    if model_revision is None: 
        model_revision = cfg.model_revision
    print(f"Model preset: {model_preset}, Model revision: {model_revision}")

    model = AutoModelForCausalLM.from_pretrained(model_preset, revision=model_revision)
    print("Model loaded.")

    training_args = cfg.default_train_args
    print("Default training arguments loaded.")

    for k, v in training_args_.items(): 
        training_args[k] = v
    print(f"Training arguments updated with custom settings: {training_args_}")

    model = run_train(
        model=model, 
        tokenizer=tokenizer, 
        ds_train=ds_train, 
        ds_val=ds_val, 
        **training_args, 
    )
    print("Training complete.")


    # Save model in hf hub 
    print("Start saving...")
    tokenizer.push_to_hub(cfg.model_path, commit_message=f'tokenizer of {model_preset}')
    model.push_to_hub(cfg.model_path, commit_message=f'trained {model_preset}({tokenizer_preset}) on dataset {dataframe_path}')

    return model
