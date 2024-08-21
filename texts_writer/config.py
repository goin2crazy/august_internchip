class Config:
  required_info_keys = ['title', 'salary', 'company', 'experience', 'mode', 'skills']

  model_path = 'doublecringe123/job-describtion-copilot-ru'
  tokenizer_path = 'cometrain/neurotitle-rugpt3-small'

  model_revision = None

  generation_config = dict(max_new_tokens=5,
                        do_sample=True,
                        top_p = 0.97,
                        top_k = 5,
                        num_beams=2,
                        # bad_words_ids = [tokenizer('\n').input_ids],
                        temperature = 0.4,
                        repetition_penalty=1.2,

                        no_repeat_ngram_size=2,)
  
  # TRAIN 
  default_dataframe_path = "hf://datasets/doublecringe123/parsed-vacancies-from-headhunter-tashkent/data/train-00000-of-00001.parquet"
  block_size = 512

  default_train_args = dict(
                push_to_hub = True, 
                # strategies 
                save_strategy = "epoch", 
                eval_strategy = "epoch", 
                
                # batch size 
                auto_find_batch_size = True, 
                num_train_epochs = 1,
                
                # optimizer 
                weight_decay = 1e-3, 
                learning_rate = 5e-4, 
                warmup_steps = 100, 
                
                save_total_limit = 1, 
                seed=42, 
            )