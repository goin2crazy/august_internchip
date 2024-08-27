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

from .filters import filter_by_language, fix_descrition

assert "HF_TOKEN" in os.environ, "Please set your HF_TOKEN to environment variables"

tqdm.pandas()

class scrape_pages():
    # Please also add logs to this class fns
    base_url = "https://tashkent.hh.uz/"

    def correct_link(self, link_to_correct): 
        # if there link not starts with http you add the self.base_url to start
        # You have to apply this fn to everywhere i use *.get('href')
        if not link_to_correct.startswith("http"):
            link_to_correct = self.base_url + link_to_correct.lstrip('/')
        # print(f"Corrected link: {link_to_correct}")
        return link_to_correct

    def collect_links(self, url): 
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        print(f"Collecting links from: {url}")

        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            print(f"Error while collecting links: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')

        all_links = [self.correct_link(link.get('href')) for link in soup.find_all('a') if type(link.get('href')) == str]
        print(f"Collected {len(all_links)} links")

        vacancies_links = [i for i in all_links if (self.base_url in i) and ('/vacancy/' in i)]
        print(f"Collected {len(vacancies_links)} vacancy links")
        return vacancies_links
    
    def collect_daily(self, url = 'https://tashkent.hh.uz/vacancies/za_poslednie_tri_dnya'): 
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        print(f"Collecting daily vacancies from: {url}")

        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            print(f"Error while collecting daily vacancies: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        all_links = [self.correct_link(link.get('href')) for link in soup.find_all('a') if type(link.get('href')) == str]

        pages = [i for i in all_links if '/vacancies/za_poslednie_tri_dnya?page' in i]
        pages = list(set(pages))
        print(f"Collected {len(pages)} pages")
        
        if len(pages): 
            all_links = list() 
            for p in pages: 
                vacancies = self.collect_links(p)
                all_links.extend(vacancies)
                
            self.urls = list(set(all_links))
            print(f"Collected total {len(self.urls)} vacancy links across multiple pages")
            return self.urls
        else: 
            all_links = self.collect_links(url)
            self.urls = list(set(all_links))
            
        if len(self.urls) < 100: 
          self.collect_daily(url)
        

    def __init__(self, urls = [], verbose=0):
        self.verbose= verbose
        self.urls = urls
        print(f"Initialized with {len(urls)} URLs")

    def __getitem__(self, idx):
        url = self.urls[idx]
        print(f"Scraping data from: {url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error while scraping {url}: {e}")
            return

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
        if filter_by_language(vacancy_description) == None: 
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

        return {'title': vacancy_title,
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


def run(save_folder='', 
        collector_verbose=0, 
        hub_path='doublecringe123/parsed-hh-last-tree-days-collection'): 

    # Initialize the scraper with the specified verbosity level
    p = scrape_pages(verbose=collector_verbose)
    
    # Collect daily vacancy links
    p.collect_daily() 

    # Initialize a dictionary to store collected data
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
    
    # Loop through each URL and scrape the data
    for features in tqdm(p): 
        if features is None:
            continue
        for k, v in collected_dict.items(): 
            collected_dict[k].append(features[k])
    
    # Convert the collected data into a DataFrame
    collected_dataframe = pd.DataFrame(collected_dict)
    
    print("Removing company names...")
    collected_dataframe['description'] = collected_dataframe['description'].progress_apply(fix_descrition)
    collected_dataframe['company'] = ''

    # Convert the DataFrame to a Hugging Face dataset
    collected_dataset = datasets.Dataset.from_pandas(collected_dataframe)
    
    # Get the current date in the format YYYY-MM-DD
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    file_name = os.path.join(save_folder, f"Collected_daily_{current_date}.csv")
    # Push the dataset to the Hugging Face hub with a commit message
    collected_dataset.push_to_hub(hub_path, commit_message=f"Update {current_date}")
    collected_dataframe.to_csv(file_name)

    return file_name
