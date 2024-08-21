
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import Config

class VacancyWriterModel(): 
    def __init__(self): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AutoModelForCausalLM.from_pretrained(Config.model_path,
                                            #  trust_remote_code=True,
                                            #  low_cpu_mem_usage=True,
                                            #  torch_dtype=torch.float16,
                                             device_map='auto',
                                             
                                             revision=Config.model_revision
                                             ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)

    @staticmethod
    def prompt_to_model(job_features, input_text): 
        features = "\n".join([f"{key}: {value}" for key, value in job_features.items()])
        prompt = f"{features}\n\n{input_text}"
        return prompt



    def forward(self, job_features, input_text, gen_config): 
        prompt = self.prompt_to_model(job_features, input_text)
    
        with torch.no_grad(): 
            tokenz = self.tokenizer(prompt, return_tensors='pt')
            tokenz = {k: v.to(self.device) for k, v in tokenz.items()} 

            output = self.model.generate(
                        # inputs 
                        **tokenz,
                        # tokens generation settings 
                        **gen_config, 
                        # fix tokenizer
                        pad_token_id=self.tokenizer.eos_token_id
                        )

            return " ".join(self.tokenizer.batch_decode(output, skip_special_tokens=True))
                    

    def __call__(self, job_features: dict, input_text: str) -> Any:
        """
        Arguments: 
            job_features - have to be dict of job features like title, calary, and etc. 

            
        Example final prompt to model will look like
            prompt = '''
                title: python Разработчик, 
                salary: 800-1600$, 
                company: "stile-ex OOO", 
                experince: 1-3 года, 
                mode: "Офисс, полный день", 
                skills: [Python, Django, Numpy, Enblish B1]

                Мы ищем талантливого массажера простаты в нашу команду! Обязательно знать'''

                tokenz = tokenizer(prompt, return_tensors='pt')

        Return: 
            models answer in str format 
        """
        return self.forward(job_features, input_text, gen_config = Config.generation_config)