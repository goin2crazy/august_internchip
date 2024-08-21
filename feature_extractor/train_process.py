# CHECK THE ENVIROMENT VARIABLES 
import os 

assert "HF_TOKEN" in os.environ, "Please set HF_TOKEN in os enviroments variables"

# IMPORT LIBS 
import torch 

import datasets 

# load model 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# train model 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# train test split
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np

# import nltk
# import evaluate 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# there load evaluate rouge metric
# IMPORT CONFIG 

from .feature_extractor import Config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# nltk.download("punkt", quiet=True)
# metric = evaluate.load("rouge")

# def get_metrics(tokenizer): 
#     def compute_metrics(eval_preds):
#         preds, labels = eval_preds
#         # decode preds and labels
#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         # rougeLSum expects newline after each sentence
#         decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#         decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
#         result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
#         return result
#     return compute_metrics

def tokenize(item, tokenizer): 
    return {
        **tokenizer(item['description'], truncation=True, max_length=768,), 
        'labels': tokenizer(item['features'], truncation=True,max_length=768).input_ids, 
    }

def build_dataset(df, tokenizer): 

    ds = datasets.Dataset.from_pandas(df)
    ds = ds.map(lambda i: tokenize(i, tokenizer), remove_columns=ds.column_names) 
    return ds 

def build_data(df, split_size=0.15, *args, **kwargs): 
    train_df, val_df = train_test_split(df, test_size=split_size, random_state=42)
    
    print('[train]')
    train_ds  = build_dataset(train_df, *args, **kwargs)
    print('[val]') 
    val_ds = build_dataset(val_df, *args, **kwargs)
    return train_ds, val_ds


        
def run_train(model:torch.nn.Module, 
              tokenizer: AutoTokenizer, 
              ds_train: datasets.Dataset, 
              ds_val: datasets.Dataset, 
              metrics = True, 
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
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        **kwargs, 
        fp16 = torch.cuda.is_available(), 
    )

        
    trainer = Seq2SeqTrainer(
        model, 
        args= training_args, 
        train_dataset = ds_train, 
        eval_dataset = ds_val, 
        data_collator = collator, 
        # compute_metrics=get_metrics(tokenizer) if metrics else None, 
    )

    trainer.train()
    return model

# MAIN TRAINING FUNCTIONS 
def train(
    dataframe_path:str, 
    tokenizer_preset:str = None, 
    model_preset:str = None, 
    model_revision: str = None, 
    validation_split=  True, 
    metrics = True, 
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

    In training_args you can put whatever params you want [Look params and explanation from huggingface.docs about TrainingArguments and Train]
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
    # Load data
    print("Loading data...")
    if dataframe_path.endswith("parquet"): 
        df = pd.read_parquet(dataframe_path)
    elif dataframe_path.endswith("csv"): 
        df = pd.read_csv(dataframe_path)
    else:
        print("Unrecognized file type")
        return None
    
    print("Data loaded successfully.")
    
    assert (("features" in df.columns) and ("description" in df.columns)), "Need DataFrame included the 'features' (label) and 'description' (inputs) columns"
    df = df[['features', 'description']]
    
    # Load tokenizer 
    print("Loading tokenizer...")
    if tokenizer_preset == None: 
        tokenizer_preset = cfg.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_preset)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")
    
    # Build datasets 
    print("Building datasets...")
    if validation_split: 
        ds_train, ds_val = build_data(df, tokenizer=tokenizer)
        print("Training and validation datasets built.")
    else: 
        ds_train, _ = build_data(df, split_size = 0.0, tokenizer=tokenizer)
        ds_val = None 
        print("Training dataset built without validation split.")
    
    # Load model 
    print("Loading model...")
    if model_preset == None: 
        model_preset = cfg.model_path
    if model_revision == None: 
        model_revision = cfg.model_revision 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_preset, revision = model_revision)
    print("Model loaded successfully.")
    
    # Update training arguments
    print("Updating training arguments...")
    training_args = cfg.default_train_args
    for k, v in training_args_.items(): 
        training_args[k] = v
    print("Training arguments updated.")
    
    # Train the model
    print("Starting training...")
    model = run_train(
        model = model, 
        tokenizer = tokenizer, 
        ds_train=ds_train, 
        ds_val = ds_val, 
        metrics=metrics, 
        **training_args, 
    )

    print("Training complete.")
    
    # Save model in hf hub 
    print("Start saving...")
    tokenizer.push_to_hub(cfg.model_path, commit_message=f'tokenizer of {model_preset}')
    model.push_to_hub(cfg.model_path, commit_message=f'trained {model_preset}({tokenizer_preset}) on dataset {dataframe_path}')

    return model
