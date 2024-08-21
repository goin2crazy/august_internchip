def extract_text_v0(i): 
    text = f''''''
    
    text = {k: v for k, v in i.items() if k != 'url' and k != 'description'}
    text = str(text) + f'\n\n{i["description"]}'
    return {'text': text}

def extract_text_v1(i): 
    text = f''''
        title: {i["title"]}, 
        salary: {i["salary"]}, 
        company: {i['company']}, 
        experience: {['experience']}, 
        mode: {i['mode']}, 
        skills: {i['skills']}
        
    '''
    
    text = {k: v for k, v in i.items() if k != 'url' and k != 'description'}
    text = str(text) + f'\n\n{i["description"]}'
    return {'text': text}

def extract_text_v2(i): 
    text = f''''
        {i["title"]}, 
        {i["salary"]}, 
        {i['company']}, 
        {['experience']}, 
        {i['mode']}, 
        {i['skills']}
        
    '''
    
    text = {k: v for k, v in i.items() if k != 'url' and k != 'description'}
    text = str(text) + f'\n\n{i["description"]}'
    return {'text': text}

def extract_text_v3(i): 
    text = f''''
        {i["title"]}, 
        {i['company']},
        
    '''
    
    text = {k: v for k, v in i.items() if k != 'url' and k != 'description'}
    text = str(text) + f'\n\n{i["description"]}'
    return {'text': text}