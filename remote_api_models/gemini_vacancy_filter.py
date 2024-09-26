from .gemini_inference import GeminiInference
import re

class GeminiForVacancyFiltration(GeminiInference): 
    def __init__(self):
        super().__init__()

    def prompt(self, vacancy_info_text: str):
        template = (lambda i: f'''Today You are have to help old HR to write vacancy. He is so old so he still can be racist and sexist because he was raised like this. 
So your task is to return a warning if he is writing discriminating text with any kind of discrimination: 

The text: 
"{i}" 

Please return your answer in format like: 

<reasoning> [there you can take a deep breath and think about the answer, reasoning here has to be longer in summarized reasoning] </reasoning> 

<summarized reasoning> [There is in two small sentences your reason why the text is discriminating some people if it does so]  - [The part in text where it is discriminating] </summarized reasoning>

<answer> [Here you have to write only "<warning>" if discrimination is present, else "<fine>"]</answer>''')

        return template(vacancy_info_text)

    def extract_target_answer(self, text):
        """Extract summarized reasoning and answer from the provided text."""
        try:
            # Extract reasoning, summarized reasoning, and answer using regex
            reasoning = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL).group(1).strip()
            summarized_reasoning = re.search(r"<summarized reasoning>(.*?)</summarized reasoning>", text, re.DOTALL).group(1).strip()
            answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL).group(1).strip()

            # Return results in dictionary format
            return {
                "reasoning": reasoning,
                "summarized_reasoning": summarized_reasoning,
                "answer": answer, 
                "warning": True if "warning" in answer else False 
            }

        except AttributeError:
            # Handle case where any part of the response is missing
            return {
                "error": "Failed to extract one or more fields from the response."
            }
