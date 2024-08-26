from transformers import pipeline
from langdetect import detect

ner = pipeline("ner", model="Davlan/distilbert-base-multilingual-cased-ner-hrl")

def replace_compamy_names(text, ranges) -> str:
    # Replace text within each range with '#'
    for start, end in ranges:
        text = text[:start] + '#' * (end - start) + text[end:]

    # Replace all '#' symbols with an empty string
    text = text.replace('#', '')

    return text

def fix_descrition(desc):  
    return replace_compamy_names(desc, [(i['start'], i['end']) for i in ner(desc)])

def filter_by_language(text):
    try:
        language = detect(text)

        if language in ['ru', 'en']: 
            return True
        else: 
            return None
    except:
        return None
