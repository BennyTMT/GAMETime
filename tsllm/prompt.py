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
import time

TIMEIND  = 0 
EVENTIDX = 1
SCORE1= 2 
SCORE2= 3
WP1 = 4 
WP2 = 5
NFLGAP =3 

# DIFF = {
#     1 : (3,5),            # easiest
#     2 : (6,8),
#     3 : (9,11),
#     4 : (12,14),        
#     5 : (15,100)          # hardest
# }

DIFF = {
    1 : (0,0.4),              # easiest
    2 : (0.4,0.5),
    3 : (0.5,0.6),
    4 : (0.6,0.7),        
    5 : (0.7,0.8),             
    6 : (0.8,0.9),             # hardest
    7 : (0.9,1.0)             # hardest
}

def build_prompts(config , datasets):
    
    if 'nfl' in config.experiment.data_path:
        sport = 'nfl'
    elif 'nba' in config.experiment.data_path:
        sport = 'nba'
    else: 
        raise FileExistsError

    if 'blank_rorder' in config.experiment.task:
        build_num_events(datasets , task = 'blank_rorder', sport = sport )
        build_similarity(datasets , task = 'blank_rorder', sport = sport )
        # diff_testing(datasets , task = 'blank_rorder', sport = sport )

def build_similarity(datasets , task = 'blank_rorder', sport = '' ):
    
    condition = {
            'timestamp':True ,
            'score': False , 
            'entity' : False , 
            'event' : False,  
            'timeseries' : True
        }
    def generate_options(ground_truth , datasets  , series_len ,   
                         select_pos_trend = None , level=None , seriesA=None):
        options = []
        metrics = []
        while len(options)!=4 : 
            # Put the ground-truth as the first one. 
            if len(options) == 0 : 
                event_series = evets_to_string(ground_truth)
                options.append(event_series)
                continue
                
            count=0
            while True :
                count +=1 
                if count >5000 : 
                    print('超时!')
                    return None , None , None 

                datas , opt_index , score_gap , seriesB , select_neg_trend = pick_neg_opt(datasets , series_len , sport=sport)
                if datas is None : continue
                
                # diff = diff_metric(select_neg_trend , select_pos_trend )
                diff = similarity_metric(seriesB , seriesA )
                if not (DIFF[level][0]<=diff<=DIFF[level][1]):  continue
                print( 'Selected  :' ,  DIFF[level] , diff)
                
                metrics.append(diff)
                if score_gap >= 18 and sport == 'nba': 
                    continue 
                else:    break

            events_lts = []
            for i in range(series_len):
                event = datas[opt_index+i][EVENTIDX] 
                if sport=='nfl': 
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

        return final_options , ground_truth , metrics

    def pick_event_series(datas , series_len ):
        while True:
            start_index = random.randint(2, len(datas)-2-series_len)  # Randomly select 
            event_series , ground_truth , steps , score_gap , trends_wp_A =\
                    generate_blank_query(start_index , datas, series_len , sport , condition )
            if event_series is None : continue
            select_trend = transform_ts_trends(trends_wp_A)

            if score_gap < 18 and sport == 'nba' : break
            elif sport == 'nfl': break 

        return event_series , ground_truth , ",".join(steps) , datas[0][2].split("_")[0] , datas[0][3].split("_")[0] ,start_index , score_gap , select_trend , trends_wp_A
    
    series_len = 10 
    chatgpt_sys_message = get_sys_message(sport , condition )
    
    for file_name,datas in datasets.items():
        for level in [1, 2 ,3, 4 , 5 ,  6 ,7]:
            
            options = None 
            count = 0 
            while options is None : 
                count +=1 
                if count > 500 : 
                    print(f'level--{level} , {file_name}重选失败!')
                    exit()
                    
                event_series , ground_truth , steps , team1 , team2 , start_index , score_gap , \
                    select_trend , trends_wp_A = pick_event_series(datas , series_len)

                # print('Select_trend', select_trend)
                options , ground_truth , metrics  = generate_options( ground_truth , \
                        datasets  , series_len ,   select_pos_trend = select_trend  , level = level, seriesA = trends_wp_A )
            
            print(f'level:{level} , {metrics}')

            reasoning_prompt = f'''Below is provided timestamps, events, scores ({team1}-{team2}), and win probabilities ({team1}-{team2}).
{event_series}
Please select the correct sequence of events for steps {steps} from the four options below,
{options[0]}{options[1]}{options[2]}{options[3]}
'''         
            reasoning_prompt += 'Without including any additional content, please return your answer directly in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
            print(chatgpt_sys_message)
            print(reasoning_prompt)
            print('Label' , ground_truth)
            save_prompt_json(chatgpt_sys_message , reasoning_prompt ,ground_truth  ,task  , f'{sport}_sim_{level}.json' ,  f"{file_name}"  , start_index)
        # exit()

def build_num_events(datasets , task = 'blank_rorder', sport = 'nfl' ):
    print('sport' , sport)
    
    condition = {
            'timestamp':True ,
            'score': False , 
            'entity' : False , 
            'event' : False,  
            'timeseries' : True
        }

    def generate_options(ground_truth , datasets  , series_len ,   select_pos_trend = None , opt=None    ):
        options = []
        
        while len(options)!=4 : 
            # Put the ground-truth as the first one. 
            if len(options) == 0 : 
                event_series = evets_to_string(ground_truth)
                options.append(event_series)
                continue
            while True :
                datas , opt_index , score_gap , _ , select_neg_trend = pick_neg_opt(datasets , series_len)
                
                if datas is None : continue 
                
                if score_gap >= 18 and sport == 'nba': 
                    continue 
                else: 
                    break
            
            events_lts = []
            for i in range(series_len):
                event = datas[opt_index+i][EVENTIDX] 
                if sport=='nfl': 
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

    def pick_event_series(datas , series_len ):
        c=0 
        while True:
            start_index = random.randint(2, len(datas)-2-series_len)  # Randomly select 
            event_series , ground_truth , steps , score_gap , trends_wp_A =\
                    generate_blank_query(start_index , datas, series_len , sport , condition=condition )
            c +=1 
            if c > 60 : 
                print('err in' , datas)
                exit()

            if trends_wp_A is None or score_gap is None : 
                continue 
            select_trend = transform_ts_trends(trends_wp_A)

            if score_gap < 18 and sport == 'nba' : break
            elif sport == 'nfl': break 
            
        return event_series , ground_truth , ",".join(steps) , datas[0][2].split("_")[0] , datas[0][3].split("_")[0] ,start_index , score_gap , select_trend , trends_wp_A

    for file_name,datas in datasets.items():
        
        chatgpt_sys_message = get_sys_message(sport , condition )
        
        for series_len in [1 , 5 , 10 , 15 , 20]:
            event_series , ground_truth , steps , team1 , team2 , start_index , score_gap , select_trend , trends_wp_A =\
                pick_event_series(datas , series_len)

            # print('Select_trend', select_trend)
            # print(event_series)
            
            options , ground_truth   = generate_options( ground_truth , \
                    datasets  , series_len ,   select_pos_trend = select_trend  , opt = None)

            assert options is not None 
            
            reasoning_prompt = f'''Below is provided timestamps, and win probabilities ({team1}).
{event_series}
Please select the correct sequence of events for steps {steps} from the four options below,
{options[0]}{options[1]}{options[2]}{options[3]}
'''         
            opt = 'wo_cot'
            if opt == 'w_cot':
                reasoning_prompt += 'Please reason step by step mainly based on win probabilities. Return your answer in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
            else : 
                reasoning_prompt += 'Without including any additional content, please return your answer directly in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
            
            print(series_len)
            print(chatgpt_sys_message)
            print(reasoning_prompt)
            print('Label' , ground_truth)
            # print('opt ' , opt , 'k' , file_name )
            save_prompt_json(chatgpt_sys_message , reasoning_prompt ,ground_truth  ,
                             'test' , f'{sport}_{series_len}.json' ,  f"{file_name}"  , start_index)

def build_random_negative(datasets , task = 'blank_rorder', sport = 'nfl' ):
    print('sport' , sport)
    def generate_options(ground_truth , datasets  , series_len ,   select_trend = None , opt=None    ):
        options = []

        while len(options)!=4 : 
            # Put the ground-truth as the first one. 
            if len(options) == 0 : 
                event_series = evets_to_string(ground_truth)
                options.append(event_series)
                continue
                
            while True :
                try:
                    datas , opt_index , score_gap , _ , _ = pick_neg_opt(datasets , series_len)
                except: 
                    print(2)
                    continue 
                if score_gap >= 18 and sport == 'nba': continue 
                else: break
                
            events_lts = []
            for i in range(series_len):
                event = datas[opt_index+i][EVENTIDX] 
                if sport=='nfl': 
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

    def pick_event_series(datas , series_len ):
        while True:
            start_index = random.randint(2, len(datas)-2-series_len)  # Randomly select 
            try:
                event_series , ground_truth , steps , score_gap , trends_wp_A =\
                        generate_blank_query(start_index , datas, series_len , provide_length=2 , game=sport )
                if event_series is None : continue
            except: 
                print(1)
                continue
            if score_gap < 18 and sport == 'nba' : break
            elif sport == 'nfl': break 
        return event_series , ground_truth , ",".join(steps) , datas[0][2].split("_")[0] , datas[0][3].split("_")[0] ,start_index , score_gap , None , trends_wp_A

    for file_name,datas in datasets.items():
        if sport == 'nfl':
            chatgpt_sys_message = f'You are an assistant for NFL football task.' + ' We will provide a series of consecutive timestamps, events, scores, and win probabilities from a NFL football game, though some intermediate events will be missing. You will need to infer the likely events that occurred in the missing intervals.'
        elif sport == 'nba':
            chatgpt_sys_message = f'You are an assistant for NBA basketball task.' + ' We will provide a series of consecutive timestamps, events, scores, and win probabilities from a basketball game, though some intermediate events will be missing. You will need to infer the likely events that occurred in the missing intervals.'

        for series_len in [1]:
            event_series , ground_truth , steps , team1 , team2 , start_index , score_gap , select_trend , trends_wp_A =\
                pick_event_series(datas , series_len)

            print(event_series)
            
            options , ground_truth   = generate_options( ground_truth , \
                    datasets  , series_len ,   select_trend = select_trend  , opt = None)

            if options is None : continue 

            for opt in [ 'wo_cot' , 'w_cot']:
                reasoning_prompt = f'''Below is provided timestamps, events, scores ({team1}-{team2}), and win probabilities ({team1}-{team2}).
{event_series}
Please select the correct sequence of events for steps {steps} from the four options below,
{options[0]}{options[1]}{options[2]}{options[3]}
'''             
                if opt == 'w_cot':
                    reasoning_prompt += 'Please reason step by step mainly based on win probabilities. Return your answer in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
                else : 
                    reasoning_prompt += 'Without including any additional content, please return your answer directly in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.'
                print(chatgpt_sys_message)
                print(reasoning_prompt)
                print('Label' , ground_truth)
                print('opt ' , opt , 'k' , file_name )
                save_prompt_json(chatgpt_sys_message , reasoning_prompt ,ground_truth  ,task  , f'{opt}_{series_len}.json' ,  f"{file_name}"  , start_index)

def diff_testing(datasets , task = 'blank_rorder', sport = '' ):
    
    def test_opts(ground_truth , datasets  , series_len ,   select_pos_trend = None , opt=None , seriesA=None):
        options = []
        while len(options)!=4 : 
            while True :
                try:
                    datas , opt_index , score_gap , seriesB , select_neg_trend = pick_neg_opt(datasets , series_len)
                except: 
                    continue 
                # diff = diff_metric(select_neg_trend , select_pos_trend )
                # print(dis , select_neg_trend , select_pos_trend)
                diff = similarity_metric(seriesA , seriesB)
                if score_gap >= 18 and sport == 'nba': 
                    continue 
                else: 
                    break
            options.append(diff)
        return options
    
    def pick_event_series(datas , series_len ):
        c = 0 
        while True:
            start_index = random.randint(2, len(datas)-2-series_len)  # Randomly select 
            event_series , ground_truth , steps , score_gap , trends_wp_A =\
                        generate_blank_query(start_index , datas, series_len )
            c +=1 
            if c > 20 : return [None]*9
            
            if trends_wp_A is None : 
                continue 
            select_trend = transform_ts_trends(trends_wp_A)
                
            if score_gap < 18 and sport == 'nba' : break
            elif sport == 'nfl': break 

        return event_series , ground_truth , ",".join(steps) , datas[0][2].split("_")[0] , datas[0][3].split("_")[0] ,start_index , score_gap , select_trend , trends_wp_A
    
    diffs = []
    games = list(datasets.keys())
    print('len of games ' ,  len(games))
    while True:
        game = random.choice(games)
        datas = datasets[game]
        event_series , ground_truth , steps , team1 , team2 , start_index , score_gap , \
                    select_trend , trends_wp_A = pick_event_series(datas , 10)
        if event_series is None : continue 
        res = test_opts(ground_truth , datasets  , 10 ,   select_pos_trend = select_trend , opt=None , seriesA = trends_wp_A)
        diffs.extend(res)
        
        if len(diffs) % 1000 == 0 : print(len(diffs))
        if len(diffs) > 10000: break 
    
    diffs = np.array(diffs) / 7.0 
    print(np.sum(diffs < 0.4) / len(diffs))
    draw_distribution(diffs , sport )
    exit()
    
'''
NFL  0.0938624550179928
NBA  0.08656537385045981

90.6% 91.3%
def select_game_from_diff_game(diff_game , datasets , file_name):
    # Select Game 
    if diff_game:
        pick_game = file_name
        while pick_game == file_name:
            pick_game = random.choice(list(datasets.keys()))
        datas = datasets[pick_game]
    else :
        datas = datasets[file_name]
    return datas 

def build_difficuly_levels(datasets , task = 'blank_rorder'):

    def generate_options(ground_truth , datasets  , series_len ,   select_trend = None , opt=None    ):
        options = []
        selected_times = []

        while len(options)!=4 : 
            # Put the ground-truth as the first one. 
            if len(options) == 0 : 
                event_series = evets_to_string(ground_truth)
                options.append(event_series)
                continue
            count = 0
            while True :
                count +=1 
                if count == 500 : print(count)
                if count > 1000 : return None, None 
                
                pick_game = random.choice(list(datasets.keys()))
                datas = datasets[pick_game]
                opt_index = random.randint(series_len+1, len(datas)-series_len-1)  
                score_gap = abs(int(datas[opt_index][SCORE1]) - int(datas[opt_index][SCORE2]))
                if score_gap >= NFLGAP: continue 
                
                try: lts = [float(datas[opt_index+i-1][WP1]) for i in range(series_len+2)]
                except: continue
                
                next_trend = transform_ts_trends(lts)
                
                # if next_trend not in TRENDS[opt] : continue
                
                if next_trend == select_trend and 'hard' in opt : 
                    selected_times.append((next_trend ,  lts))
                    break
                elif next_trend != select_trend and 'easy'in opt :
                    selected_times.append((next_trend ,  lts))
                    break

            events_lts = [datas[opt_index+i][EVENTIDX] for i in range(series_len)]
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
        
        print('neg options:' , selected_times )
        return final_options , ground_truth  
    
    def pick_event_series(datas , series_len ):
        # start_index =  get_max_window(datas, window_size=20)
        # start_index =  get_min_window(datas, window_size=20)
        select_trend = None
        count = 0 
        while select_trend not in TRENDS[opt]:
            start_index = random.randint(2, len(datas)-2-series_len)  # Randomly select 
            try:
                event_series , ground_truth , steps , score_gap , trends_wp_A = generate_blank_query(start_index , datas, series_len , provide_length=2 )
            except:
                continue
            select_trend = transform_ts_trends(trends_wp_A)
            # if score_gap >= NBAGAP or select_trend not in TRENDS[opt] : continue
            # print(select_trend ,  TRENDS[opt] , opt)
            count+=1
            if count > 500 : break 
        return event_series , ground_truth , ",".join(steps) , datas[0][2].split("_")[0] , datas[0][3].split("_")[0] ,start_index , score_gap , select_trend , trends_wp_A
    
    count = 0 
    for file_name,datas in datasets.items():
        count +=1 
        if count > 125 : break 
        # chatgpt_sys_message = f'You are an assistant for NBA basketball task.' + ' We will provide a series of consecutive timestamps, events, scores, and win probabilities from a basketball game, though some intermediate events will be missing. You will need to infer the likely events that occurred in the missing intervals.'
        chatgpt_sys_message = f'You are an assistant for NFL football task.' + ' We will provide a series of consecutive timestamps, events, scores, and win probabilities from a NFL football game, though some intermediate events will be missing. You will need to infer the likely events that occurred in the missing intervals.'
        # s -> options from the same series
        for series_len in [ 1, 3 , 5]:
          for opt in ['easy' , 'easy1' , 'easy2' , 'hard1' , 'hard2' ,'hard3' ]:
            if opt == 'hard3':
                global TRENDS
                TRENDS['hard3'] = random.choice([['+'],['-']])

            event_series , ground_truth , steps , team1 , team2 , start_index , score_gap , select_trend , trends_wp_A =\
                pick_event_series(datas , series_len)
                
            if event_series is None : continue 

            print(opt , TRENDS[opt] , select_trend )
            print(event_series)
            
            options , ground_truth   = generate_options( ground_truth , \
                    datasets  , series_len ,   select_trend = select_trend  , opt = opt)

            if options is None : continue 
            
            reasoning_prompt = f'Below is provided timestamps, events, scores ({team1}-{team2}), and win probabilities ({team1}-{team2}).
{event_series}
Please select the correct sequence of events for steps {steps} from the four options below,
{options[0]}{options[1]}{options[2]}{options[3]}
Please reason step by step mainly based on win probabilities. Return your answer in the format **X**, where X only contains the chosen option, such as **a**, **b**, **c**, or **d**.
'      
            # print(team1 , team2)
            print(chatgpt_sys_message)
            print(reasoning_prompt)
            print('Label' , ground_truth)
            print('opt ' , opt , 'k' , file_name , start_index )
            save_prompt_json(chatgpt_sys_message , reasoning_prompt ,ground_truth  ,task  , f'{opt}_{series_len}.json' ,  f"{file_name}" )
            
'''