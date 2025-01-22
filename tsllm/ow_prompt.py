import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass
import random 
import re 
import pickle
import json 
from prompt_utils import * 
from prompt import  generate_blank_query 
import csv

SERIESLEN = 10 
TESTWITHTIME = False
KEEPNUM = False

def build_ow_prompts(config , datasets):
    global KEEPNUM
    '''
        datasets[nums_date] ={
            'nums' : round(close_numss[i] , 0) , 
            'events' : events
        }
    '''
    series_len = 10
    starts = [-1]
    start_index = -1
    chatgpt_sys_message , provided_info = get_sys_prompt(config) 
    domain = config.experiment.data_path.split("/")[-2]
    
    for _ in range(100):
      for opt in [ 'wo_num' , 'w_num' , ]:
        if 'wo' in opt : 
            KEEPNUM = False 
            note = '(Note that numbers in events are replaced with symbols)'
        else:
            KEEPNUM = True
            note = ''

        while start_index in starts:
            event_series , ground_truth , steps , start_index  =\
                pick_event_series(datasets , series_len)
                
        assert start_index != -1
        starts.append(start_index)
        
        options , ground_truth   = generate_options( ground_truth , \
                datasets  , series_len , start_index  )
        
        
        reasoning_prompt = f'''Below is provided {provided_info},
{event_series}
Please select the correct sequence of events (news) from the four options below,
{options[0]}{options[1]}{options[2]}{options[3]}{note}
'''     
        cot = False 
        if cot:
            reasoning_prompt += 'Please reason step by step mainly based on price series. Return your answer in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
        else : 
            reasoning_prompt += 'Without including any additional content, please return your answer directly in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
        
        # continue
        print(chatgpt_sys_message)
        print(reasoning_prompt)
        print('Label' , ground_truth)
        save_prompt_json(chatgpt_sys_message , reasoning_prompt ,ground_truth  , 'blank_rorder'  , f'{domain}_{opt}.json' ,  f"{start_index}"  , start_index)
        start_index = -1
        
def generate_options(ground_truth ,   datasets  , series_len , start_index ):
    options = []
    keys = list(datasets.keys())
    while len(options)!=4 : 
        # Put the ground-truth as the first one. 
        if len(options) == 0 : 
            event_series = evets_to_string(ground_truth)
            options.append(event_series)
            continue
            
        events_lts = []
        
        neg_start_index = start_index
        while abs(neg_start_index - start_index ) < 10 : 
            neg_start_index = random.randint(0 , len(datasets.keys())-series_len-3)  

        for i in range(2 + series_len):
            event =  get_one_from_events(datasets[keys[neg_start_index+i]]['events'])
            if TESTWITHTIME :
                events_lts.append(f"{i}. {keys[neg_start_index+i]} {event}")
            else:
                events_lts.append(f"{i}.{event}")

        event_series= evets_to_string(events_lts)
        options.append(event_series)

    original_options = [0 , 1 ,2 ,3 ]
    random.shuffle(original_options)
    
    ground_truth = None         
    final_options = []
    
    for i in range(4):
        opt_alpha =  ['a' , 'b' , 'c' , 'd'][i]
        label = original_options[i]
        
        if label == 0 :  
            ground_truth = opt_alpha 

        final_options.append(f"{opt_alpha}.\n"+options[label])

    return final_options , ground_truth  
    
def pick_event_series(datasets , series_len):
    start_index = random.randint(0 , len(datasets.keys())-series_len-3)  # Randomly select 
    event_series , ground_truth , steps  =\
            make_ow_events(start_index , datasets, series_len  )
    return event_series , ground_truth , ",".join(steps)  ,start_index 

def make_ow_events(start_index , datasets, series_len ):
    '''
        datasets[nums_date] ={
            'nums' : round(close_prices[i] , 0) , 
            'events' : events
        }
    '''
    event_series = ''
    ground_truth = []
    steps = []
    lag_ = 2 
    keys = list(datasets.keys())
    # 2 more events
    for i in range(lag_+series_len):
        date=keys[start_index+i]
        nums = datasets[date]['nums']
        event = get_one_from_events(datasets[date]['events'])
        # title = pick_title(datasets[date]['events'])
        timestampe = date
        if i >= 2:
            event_series+=f'{i}. {timestampe} {nums}\n'
        if TESTWITHTIME : 
            ground_truth.append(f"{str(i)}. {timestampe} {event}")
        else :
            ground_truth.append(f"{str(i)}.{event}")
            
        steps.append(str(i))
        
    return event_series , ground_truth , steps 

def get_one_from_events(events):
    global KEEPNUM
    assert len(events) > 0
    
    if len(events) > 1 : 
        event = random.choice(events)
    else : 
        event = events[0]
    if not KEEPNUM : 
        # print('1' , event )
        event = remove_nums(event)
        # print('2' , event )
        # print()
    return event

def get_sys_prompt(config):
    if 'crypto' in config.experiment.data_path:
        chatgpt_sys_message = f'You are an assistant for a cryptocurrency (Bitcoin) task. We will provide a series of consecutive timestamps along with the closing prices for those days. Additionally, we will present four potential event (news title) sequences that occurred during that period, as well as from the previous two days. Your task is to identify and select the correct sequence of events.'
        provided_info = 'date and prices ($)'
    elif 'trading' in config.experiment.data_path:
        chatgpt_sys_message = f'You are an assistant for an import-export trade task. We will provide a series of consecutive timestamps along with the values for imports and exports. Additionally, we will present four potential event (news) sequences that occurred during that period, as well as from the previous two days. Your task is to identify and select the correct sequence of events.'
        provided_info = 'date and values of import and export'
    elif 'health_US' in config.experiment.data_path:
        chatgpt_sys_message = f'You are an assistant for an Influenza Patients task. We will provide a series of consecutive timestamps along with the Influenza Patients Proportion. Additionally, we will present four potential event (news) sequences that occurred during that period, as well as from the previous two days. Your task is to identify and select the correct sequence of events.'
        provided_info = 'date and Patients Proportion (%)'
    elif 'energy' in config.experiment.data_path:
        chatgpt_sys_message = f'You are an assistant for an Energy (Gasoline) task. We will provide a series of consecutive timestamps along with the Price (Dollars per Gallon). Additionally, we will present four potential event (news) sequences that occurred during that period, as well as from the previous two days. Your task is to identify and select the correct sequence of events.'
        provided_info = 'date and Dollars per Gallon ($)'
    else : 
        raise FileExistsError
    return chatgpt_sys_message , provided_info