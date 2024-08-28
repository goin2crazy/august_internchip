from bs4 import BeautifulSoup
import requests
import json
import os
import time
from datetime import datetime
import IPython.display
from tqdm.notebook import tqdm
import pandas as pd 
import datasets
import argparse

from .filters import filter_by_language, fix_descrition

assert "HF_TOKEN" in os.environ, "Please set your HF_TOKEN to environment variables"

try: 
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.headless = True
except: 
    print("Damn, install selenium bro")
          
driver = None

tqdm.pandas()

def get_html(url, mode='regular', wait_time=5):
    global driver
    if mode == 'selenium':
        driver.get(url)
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        page_html = driver.page_source
        return page_html

    elif mode =='regular': 
        try:
            headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
            response = requests.get(url, headers=headers)
            return response.content
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return None

class scrape_pages():
    base_url = "https://tashkent.hh.uz/"

    def correct_link(self, link_to_correct): 
        if not link_to_correct.startswith("http"):
            link_to_correct = self.base_url + link_to_correct.lstrip('/')
        return link_to_correct

    def collect_links(self, url, mode, selenium_waittime): 
        print(f"Collecting links from: {url}")

        content = get_html(url, mode, wait_time=selenium_waittime)
        if content is None:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        all_links = [self.correct_link(link.get('href')) for link in soup.find_all('a') if type(link.get('href')) == str]
        print(f"Collected {len(all_links)} links")

        vacancies_links = [i for i in all_links if (self.base_url in i) and ('/vacancy/' in i)]
        print(f"Collected {len(vacancies_links)} vacancy links")
        return vacancies_links
    
    def collect_daily(self, url='https://tashkent.hh.uz/vacancies/za_poslednie_tri_dnya', mode='regular', selenium_waittime=5): 
        global driver
        driver = webdriver.Chrome(options=chrome_options)

        print(f"Collecting daily vacancies from: {url}")

        content = get_html(url, mode, wait_time=selenium_waittime)
        if content is None:
            return []
        
        soup = BeautifulSoup(content, 'html.parser')
        all_links = [self.correct_link(link.get('href')) for link in soup.find_all('a') if type(link.get('href')) == str]

        pages = [i for i in all_links if '/vacancies/za_poslednie_tri_dnya?page' in i]
        pages = list(set(pages))
        print(f"Collected {len(pages)} pages")
        
        if len(pages): 
            all_links = list() 
            for p in pages: 
                vacancies = self.collect_links(p, mode, selenium_waittime=selenium_waittime)
                all_links.extend(vacancies)
                
            self.urls = list(set(all_links))
            print(f"Collected total {len(self.urls)} vacancy links across multiple pages")
            return self.urls
        else: 
            all_links = self.collect_links(url, mode, selenium_waittime=selenium_waittime)
            self.urls = list(set(all_links))
            
        if len(self.urls) < 100: 
            self.collect_daily(url, mode, selenium_waittime=selenium_waittime)

        driver.quit() 
        

    def __init__(self, urls=[], verbose=0, mode='regular'):
        self.verbose = verbose
        self.urls = urls
        self.mode = mode
        print(f"Initialized with {len(urls)} URLs")

    def __getitem__(self, idx):
        url = self.urls[idx]
        print(f"Scraping data from: {url}")

        content = get_html(url, self.mode)
        if content is None:
            return None

        soup = BeautifulSoup(content, 'html.parser')

        try:
            vacancy_title = soup.find('h1', {'data-qa':'vacancy-title'}).text.strip()
        except:
            print(f"Could not find vacancy title at {url}")
            return

        try:
            vacancy_salary = soup.find('div', {'data-qa': 'vacancy-salary'}).text.strip()
        except:
            vacancy_salary = 'None'

        vacancy_description = soup.find('div', {'data-qa': 'vacancy-description'}).text.strip()
        if filter_by_language(vacancy_description) is None: 
            return 

        vacancy_company_name = soup.find('span', {'class': 'vacancy-company-name'}).text.strip()
        vacancy_required_experience = soup.find('span', {'data-qa': 'vacancy-experience'}).text.strip()
        employment_mode = soup.find('p', {"data-qa": "vacancy-view-employment-mode"}).text.strip()
        required_skils = ", ".join([skill.text.strip() for skill in soup.find_all('li', {'data-qa': 'skills-element'})])

        if self.verbose: 
            print("Title:", vacancy_title)
            print("Salary:", vacancy_salary)
            print("Company:", vacancy_company_name)
            print("Experience:", vacancy_required_experience)
            print("Model:", employment_mode)
            print("Description:", vacancy_description)
            print("Skills:", required_skils)
            print("-" * 30)

        time.sleep(1)  # Be kind to the server, avoid getting blocked
        IPython.display.clear_output(wait=True)  # clear output

        return {
            'title': vacancy_title,
            'salary': vacancy_salary,
            'company': vacancy_company_name,
            'experience': vacancy_required_experience,
            'mode': employment_mode,
            'skills': required_skils,
            'url': url,
            'description': vacancy_description,
        }
    
    def __len__(self):
        return len(self.urls)


def run(save_folder='', collector_verbose=0, hub_path='doublecringe123/parsed-hh-last-tree-days-collection', mode='regular', selenium_waittime=5): 

    p = scrape_pages(verbose=collector_verbose, mode=mode)
    p.collect_daily(mode=mode, selenium_waittime=selenium_waittime)

    collected_dict = {
        'title': list(),
        'salary': list(),
        'company': list(),
        'experience': list(),
        'mode': list(),
        'skills': list(),
        'url': list(),
        'description': list(),
    }
    
    for features in tqdm(p): 
        if features is None:
            continue
        for k, v in collected_dict.items(): 
            v.append(features[k])
    
    collected_dataframe = pd.DataFrame(collected_dict)
    
    print("Removing company names...")
    collected_dataframe['description'] = collected_dataframe['description'].progress_apply(fix_descrition)
    collected_dataframe['company'] = ''

    collected_dataset = datasets.Dataset.from_pandas(collected_dataframe)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    file_name = os.path.join(save_folder, f"Collected_daily_{current_date}.csv")
    collected_dataset.push_to_hub(hub_path, commit_message=f"Update {current_date}")
    collected_dataframe.to_csv(file_name)

    return file_name
