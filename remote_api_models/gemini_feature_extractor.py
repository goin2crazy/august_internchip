import json
from .gemini_inference import GeminiInference

class GeminiForFeatureExtraction(GeminiInference): 
  def __init__(self, ):
    super().__init__()

  def extract_target_answer(self, text):
        # Remove the Markdown formatting
    cleaned_text = text.strip("```json\n").strip("```")
    
    # Convert the cleaned JSON string to a Python dictionary
    try:
        data_dict = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    
    return data_dict

  def prompt(self, text):
    prompt = (lambda text: '''You are not only language model, also talented mind created for help me to solve my tasks. 
    You are the my helper for speeding up my HR tasks.
    I will show you text, you task is extract information in json format to like: 

    {"Core Responsibilities": ["N/A"],
    "Required Skills": ["N/A"],
    "Educational Requirements": ["N/A"],
    "Experience Level": ["N/A"],
    "Preferred Qualifications": ["N/A"],
    "Compensation and Benefits": ["N/A"]\}
    ''' + f'''
    PLEASE GIVE ME ONLY THIS AS ANSWER. 
    from vacancy or resume text
    My Text: 
    "{text}"''')

    return prompt(text)