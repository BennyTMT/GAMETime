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

def build_avai_prompts(config , datasets):
    
    if 'nfl' in config.experiment.data_path:
        sport = 'nfl'
        items = ['base'  , '+entity', '+score' , '+timestamp',  '+event' , 'reorder' , '-timeseries']
    elif 'nba' in config.experiment.data_path:
        sport = 'nba'
        items = ['base'  , '+entity', '+score' , '+timestamp',  '+event' , 'reorder' , '-timeseries']
    else: 
        raise FileExistsError
        
    for file_name,datas in datasets.items():
        
        #============================ Main Process ============================
        
        init_options = select_options(datasets , datas , file_name ,  sport = sport)
        
        for opt in items:
            reorder = True if opt == 'reorder' else False
            
            conditions = get_condition(opt)

            chatgpt_sys_message = get_sys_message(sport , conditions )
            
            team1 , team2 , event_series , ground_truth, options , steps =\
                    select_from_real_fake(init_options , sport , conditions , reorder)
            
            is_score = f' scores ({team1}),' if conditions['score'] else ''
            is_time = ' timestamps,' if conditions['timestamp'] else ''
            is_event = ' starting and ending events,' if conditions['event'] else ''
            
            reasoning_prompt = f'''Below is provided{is_time}{is_event}{is_score} win probabilities ({team1}).
{event_series}
Please select the correct sequence of events for steps {steps} from the four options below,
{options[0]}{options[1]}{options[2]}{options[3]}
'''         
            reasoning_prompt += 'Without including any additional content, please return your answer directly in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
            print(opt)
            print(chatgpt_sys_message)
            print(reasoning_prompt)
            print('Label' , ground_truth)
            print('opt ' , opt , 'k' , file_name )
            # continue
            save_prompt_json(chatgpt_sys_message , reasoning_prompt ,ground_truth  , 'blank_rorder'  , f'{sport}_{opt}.json' ,  f"{file_name}"  , init_options)
        # exit()
     #============================ Main Process ============================

     
def select_options(datasets , datas , file_name ,  sport = '' ):
    
    def generate_options(datasets):
        options = []
        while len(options)!=3 : 
            game , neg_index  = pick_detialed_neg_opt(datasets)
            options.append((game,neg_index))
        return options   
        
    def pick_event_series(datas):
        while True:
            start_index = random.randint(2, len(datas)-2-SERIESLEN)  # Randomly select 
            _ , _ , _ , score_gap , _ =\
                        generate_blank_query(start_index , datas,  SERIESLEN, sport  )
            if score_gap is None : continue 
            if score_gap < 18 and sport == 'nba' : break
            elif sport == 'nfl': break 
            
        return start_index

    options = []
    start_index =  pick_event_series(datas)
    
    options.append((file_name,start_index))
    neg_options = generate_options(datasets)
    for neg in neg_options:
        options.append(neg)
        
    assert len(options) == 4 
    return options 
    
def pick_detialed_neg_opt(datasets):
    try:
        game = random.choice(list(datasets.keys()))
        datas = datasets[game]
        opt_index = random.randint(SERIESLEN+1, len(datas)-SERIESLEN-1)  
        # lts = [float(datas[opt_index+i-1][WP1]) for i in range(SERIESLEN+2)]
    except Exception as e:
        print(f"{game} , Error: {e} , WP:{WP1} , SERIESLEN:{SERIESLEN} ")
        lts = [(datas[opt_index+i-1][WP1]) for i in range(SERIESLEN+2)]
        print(lts , opt_index)
        exit()
    return game , opt_index 
    
def select_from_real_fake(init_options, sport , conditions , reorder):
    if conditions['entity']:
        if sport =='nfl':
            data_path = "./datasets/nfl_timeseries/real_name_games/"
        elif sport =='nba':
            data_path = "./datasets/nba_timeseries/24-25/chart/"
    else :
        if sport =='nfl':
            data_path = "./datasets/nfl_timeseries/fake_name_games/"
        elif sport =='nba':
            data_path = "./datasets/nba_timeseries/24-25-fakename/"
            
    event_series , ground_truth , steps , team1 , team2= select_with_start_index(init_options , data_path , conditions , sport)
    if not reorder:
        options , ground_truth_label   = generate_options(ground_truth ,  data_path , init_options   , sport)
    else:
        options , ground_truth_label   = generate_reorder_options(ground_truth)

    return team1 , team2 , event_series , ground_truth_label, options , steps

def select_with_start_index(init_options  , data_path , conditions , sport ):
    game , start_index = init_options[0]
        
    for fn in os.listdir(data_path):
        if fn == game:
            datas = read_game(data_path+fn)
    assert datas is not None
    
    event_series , ground_truth , steps , score_gap , trends_wp_A =\
                generate_blank_query(start_index , datas, SERIESLEN  , sport,  conditions)

    return event_series , ground_truth , ",".join(steps) , datas[0][2].split("_")[0] , datas[0][3].split("_")[0]


def generate_reorder_options(ground_truth):
    options = []
    
    for i in range(4):
        
        if i == 0 : 
            options.append(evets_to_string(ground_truth))
            continue
            
        random.shuffle(ground_truth)
        options.append(evets_to_string(ground_truth))

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


def generate_options( ground_truth ,  data_path , init_options   , sport ):
    options = []
    
    for i in range(4):

        if i == 0 : 
            event_series = evets_to_string(ground_truth)
            options.append(event_series)
            continue
            
        game , opt_index = init_options[i]
        datas = read_game(data_path+game)
        # lts = [float(datas[opt_index+i-1][WP1]) for i in range(SERIESLEN+2)]
        
        events_lts = []
        for i in range(SERIESLEN):
            event = datas[opt_index+i][EVENTIDX] 
            if sport =='nfl':
                event = drop_nfl_posi(event)
            events_lts.append(event)
            
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


def get_condition(opt):
    # OPT is Base or reorder
    is_timestamp = False
    is_score = False
    is_entity = False
    event = False 
    timeseries = True
    if opt == '+entity':
        is_entity = True
    elif opt == '+score':
        is_score = True
    elif opt == '+timestamp':
        is_timestamp = True 
    elif opt =='+event':
        event = True 
    elif opt =='-timeseries':
        timeseries = False
         
    conditions = {
        'timestamp':is_timestamp ,
        'score': is_score , 
        'entity' : is_entity , 
        'event' : event,  
        'timeseries' : timeseries
    }
    return conditions 

def get_sys_message(sport , conditions ):
    is_score = ', scores,' if conditions['score'] else ''
    is_timestamp = ' timestamps,' if conditions['timestamp'] else ''
    is_event = ' event,' if conditions['event'] else ''
    if sport == 'nfl':
        chatgpt_sys_message = f'You are an assistant for NFL football task. We will provide a series of consecutive{is_timestamp}{is_event}{is_score} win probabilities from a NFL football game, though some intermediate events will be missing. You will need to infer the likely events that occurred in the missing intervals.'
    elif sport == 'nba':
        chatgpt_sys_message = f'You are an assistant for NBA basketball task. We will provide a series of consecutive{is_timestamp}{is_event}{is_score} win probabilities from a basketball game, though some intermediate events will be missing. You will need to infer the likely events that occurred in the missing intervals.'

    return chatgpt_sys_message