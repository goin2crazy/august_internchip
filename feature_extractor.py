# prompt: write code to find folder in all dirs in '/content/' which starts from 'checkpoint'

import os
import config as cfg

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

import ast
import json

class FeatureExtractor():

  @staticmethod
  def collect_outputs(items_):
    items_ = [i if i.startswith('{') else '{'+i  for i in items_]
    items_ = [i if i.endswith('}') else i+'}'  for i in items_]


    items = list()
    for item in items_:

      try:
        v = ast.literal_eval(item)
        v = items.append(v)
      except Exception as e:
        print (e)

    return {k: [item[k] for item in items if item[k] != ['N/A']][:2] if len([item[k] for item in items if item[k] != ['N/A']]) else "None" for k in items[0].keys()}

  def __init__(self, model_path = 'doublecringe123/lora_job_features_extractor_flant5_v2', revision = '00e8801ca3d4314a1d2cb0be101440ce738dd129'):


    config = PeftConfig.from_pretrained(model_path, revision=revision)
    
    self.config = config

    self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    self.lora_model = PeftModel.from_pretrained(
        self.base_model,
        model_path,
        revision=revision,
        torch_dtype=torch.float16, 
        is_trainable=False,
        device_map='auto',
    )
    self.text_gen_config = dict(
        max_length =cfg.max_length,
          do_sample=True,
          top_k=80,
          num_return_sequences=5,
          temperature=0.8,
          eos_token_id=self.tokenizer.eos_token_id,)

  def set_gen_config(self, key_name, value):
    self.text_gen_config[key_name] = value

  def call_single(self, text):
    with torch.inference_mode():
      inputs = self.tokenizer(
          "summarize: " + text,
          return_tensors='pt',
      )
      outputs = self.lora_model.generate(
          input_ids=inputs['input_ids'].to(self.lora_model.device),
          attention_mask=inputs['attention_mask'].to(self.lora_model.device),
          **self.text_gen_config,
          )
      outputs =  self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
      return self.collect_outputs(outputs)

  def call_batch(self, batch):

    with torch.inference_mode():
      inputs = self.tokenizer(
          batch,
          return_tensors='pt',
      )
      outputs = self.lora_model.generate(
          input_ids=inputs['input_ids'].to(self.lora_model.device),
          attention_mask=inputs['attention_mask'].to(self.lora_model.device),
          **self.text_gen_config,
          )
      return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

  def __call__(self, text_or_batch, sorting_embeddings_model = None):
    outputs = self.call_single(text_or_batch) if isinstance(text_or_batch, str) else self.call_batch(text_or_batch)

    if sorting_embeddings_model != None:

      outputs_values = str(outputs).replace('[', '').replace(']', '').replace('"', '')
      output_embeddings = sorting_embeddings_model(outputs_values)
      return outputs, output_embeddings
    else:
      return outputs
