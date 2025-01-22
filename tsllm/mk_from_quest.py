import numpy as np 
import json
import os 
import random
import re 
import csv 

TASK='blank_rorder'

def mk_CoT():

    for game in ['nba' , 'nfl']:
        
        file_path = f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{game}_+timestamp.json'
        file_path_= f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{game}_+timestamp_CoT.json'
        
        with open(file_path , 'r', encoding='utf-8') as f:
            data = json.load(f) 

        for game_name in data:
            data[game_name]['input'] = data[game_name]['input'][:-177] + 'Please reason step by step mainly based on win probabilities. Return your answer in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
            
        with open(file_path_, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def mk_none():
    for game in ['nba' , 'nfl']:
        
        file_path = f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{game}_base.json'
        file_path_= f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{game}_none.json'
        
        with open(file_path , 'r', encoding='utf-8') as f:
            data = json.load(f) 
            
        for game_name in data:
            text = data[game_name]['input']
            # pattern = r'(?<=\d\.).*?(?=\()|(?<=\d{2}\.).*?(?=\()'
            # pattern = r'(?<=0\.).*?(?=\()|(?<=11\.).*?(?=\()'
            pattern = r'(?<=0\.).*?(?=\(\d)|(?<=11\.).*?(?=\(\d)'
            cleaned_text = re.sub(pattern, ' ', text).strip()
                        
            print(data[game_name]['input'])
            print(cleaned_text)
            print("-"*50)
            data[game_name]['input'] = cleaned_text

        with open(file_path_, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
mk_none()