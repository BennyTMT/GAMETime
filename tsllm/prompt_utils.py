import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass
import random 
import re 
import json 
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
# import pickle
import csv

TIMEIND  = 0 
EVENTIDX = 1
SCORE1= 2 
SCORE2= 3
WP1 = 4 
WP2 = 5

MOVE  = 1 
CENSCORE = 2 
SIDE = 3
FEN = 4

def time2text(datas, preprocess):
    '''
        ts is a list, [0.331, 0.212 , ...] 
        out is a str : "33, 21, ..."
    '''
    ts = []
    for i in range(1,len(datas)):
        value = datas[i]
        ts.append(value[4])
    if preprocess == 'percent':
        rounded_values = [ str(int(round(float(value),2) * 100)) for value in ts]
        # for i in range(len(ts)):
            # print(ts[i] , rounded_values[i])
        return ", ".join(rounded_values)

def max_change_window(arr, window_size=20):
    changes = np.abs(np.diff(arr))
    max_change = -1
    max_start_index = 0
    for i in range(len(changes) - window_size + 1):
        window_change_sum = np.sum(changes[i:i + window_size - 1])  #
        
        if window_change_sum > max_change:
            max_change = window_change_sum
            max_start_index = i
    return max_start_index

def pick_events(datas , win_size=10 , pick_method='random'):
    '''
        For matching evaluation
    '''
    event = []
    scores = []
    loc = -1 
    if pick_method == 'random':
        # randomly pick a period 
        loc = random.randint(1, len(datas)-20)  
        for i in range(win_size):
            event.append(datas[loc+i][1])
            scores.append("{}-{}".format(datas[loc+i][2],datas[loc+i][3]))
    return loc-1 , event , scores


def get_max_window(datas, window_size=20):
    wps = [float(datas[i][WP1]) for i in range(1,len(datas))]
    changes = np.abs(np.diff(wps))
    max_change = -1
    max_start_index = 0
    for i in range(len(changes) - window_size + 1):
        window_change_sum = np.sum(changes[i:i + window_size - 1])  #
        
        if window_change_sum > max_change:
            max_change = window_change_sum
            max_start_index = i
    return max_start_index + 1 

def evets_to_string(lts):
    event_series=''
    assert len(lts) >= 1 
    for item in lts:
        #Ground-truth : (step,event)
        if len(item) == 2 : 
            event = item[1]
        else :
            event = item
        event_series += f"{event.strip()}\n"
    return event_series


def get_score_diff(datas , opt_index  , series_len):
    init_score_1 = int(datas[opt_index-1][SCORE1])
    init_score_2 = int(datas[opt_index-1][SCORE2])
    
    score_diff1 = int(datas[opt_index-1+series_len][SCORE1]) - init_score_1
    score_diff2 = int(datas[opt_index-1+series_len][SCORE2]) - init_score_2
    
    return max(score_diff1 , score_diff2)
    
def save_prompt_json(chatgpt_sys_message , forecast_prompt , ground_truth, task , \
                        file_name , dsname , start_index , nums_ques=200 ):
    question ={}
    question['sys_prompt'] = chatgpt_sys_message 
    question['input'] = forecast_prompt 
    question['label'] = ground_truth
    question['start_index'] = start_index
    
    if not os.path.exists(f'./prompts/{task}'):
        os.makedirs(f'./prompts/{task}')
        
    file_path = f'./prompts/{task}/{file_name}'
    data={}
    if os.path.exists(file_path):
        # 添加到已有类别
        with open(file_path , 'r', encoding='utf-8') as f:
            data = json.load(f) 
        if dsname in data: 
            return
        else:
            data[dsname] = question
    else:
        # 新建问题
        data[dsname] = question

    if len(data) <= nums_ques : 
        with open(f'./prompts/{task}/{file_name}', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(file_path , 'len:' , len(data))
        
def abstract_event(event):
    categories = {
        'makes three point': ['makes three point','makes 26-foot', 'makes 23-foot','makes 24-foot' , 'makes 27-foot'],
        'makes two point': [ 'makes 15-foot' , 'makes 1-foot', 'makes two point', 'makes layup', 'makes driving layup','makes 2-foot','makes 9-foot', 'makes 19-foot', 'makes 11-foot'],
        'makes free throw': ['makes free throw' , 'makes technical free throw'],
        'foul': ['foul'],
        'rebound': ['rebound'],
        'misses': ['misses', 'traveling', 'blocks' , 'charge'],
        'lost ball': ['steals','lost ball'],
        'enters the game': ['enters the game'],
        'timeout': ['timeout'],
        'turnover': ['turnover'] , 
        'CHALLENGE' : ['CHALLENGE']
    }
    
    for category, keywords in categories.items():
        if 'makes' in event : 
            pattern = r'\b(\d+)-foot\b'
            matches = re.findall(pattern, event) 
            if len(matches) == 1 : 
                result = int(matches[0])
                if result<= 22 : return 'makes two point'
                else : return 'makes three point'
            if 'two point' in event or 'dunk' in event or '': return 'makes two point'
            elif 'three point' in event : return 'makes three point'
            elif 'free throw' in event : return 'makes free throw'
        if any(keyword in event for keyword in keywords):
            return category
    return 'others'


def read_game(file_path):
    game = []
    with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)
            if 'nfl' in file_path and 'real' in file_path:
                column_names.append(column_names.pop(2))  
            game.append(column_names)
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            values_ = list(row.values()) 
            assert None not in values_ 
            if 'nfl' in file_path and 'real' in file_path:
                values_.append(values_.pop(2))  
            game.append(values_)
    return game 

def drop_nfl_posi(text):
    modified_text = re.sub(r'\([^()]*\)$', '', text)
    return modified_text


#################
#################
# Tools 
def generate_blank_query(start_index , datas, series_len , condition=None):

    if condition is None :
        condition = {
        'timestamp':False ,
        'score': False , 
        'entity' : False , 
        'posi' : False , 
        'init' :True 
    }
    
    provide_length=2
    event_series = ''
    ground_truth = []
    steps = []
    trends_wp_A = []
    last_wp1 = None ; last_wp2 = None
    
    for i in range(provide_length+series_len):
        try:
            wp1 = float(datas[start_index+i][WP1])
            wp2 = float(datas[start_index+i][WP2])
        except Exception as e:
            wp1 = last_wp1  ;  wp2 = last_wp2
            print(f"Error: {e} in function 'generate_blank_query'")
            if last_wp1 is None : exit()
            
        last_wp1=wp1 ; last_wp2=wp2
        trends_wp_A.append(wp1)
        
        wp1p = str(round(wp1*100,1))
        wp2p = str(round(wp2*100,1))
        
        s1 = datas[start_index+i][SCORE1]
        s2 = datas[start_index+i][SCORE2]
        event = datas[start_index+i][EVENTIDX]
        
        if not condition['posi']:
            event = drop_nfl_posi(event)
        
        timestampe =  datas[start_index+i][TIMEIND]
        
        if i == series_len:
            score_gap = abs(int(s1) - int(s2))
            
        if 1<=i<=series_len : 
            if condition['timestamp']:
                if condition['score']:
                    event_series+=f'{i}. {timestampe} ({s1}-{s2}) ({wp1p}%-{wp2p}%)\n'
                else:
                    event_series+=f'{i}. {timestampe} ({wp1p}%-{wp2p}%)\n'
            else:
                if condition['score']:
                    event_series+=f'{i}. ({s1}-{s2}) ({wp1p}%-{wp2p}%)\n'
                else:
                    event_series+=f'{i}. ({wp1p}%-{wp2p}%)\n'
                        
            ground_truth.append((i,event))
            steps.append(str(i))
        else :
            if not condition['init']:
                event = ''
            if condition['timestamp']:
                if condition['score']:
                    event_series+=f'{i}. {timestampe} {event} ({s1}-{s2}) ({wp1p}%-{wp2p}%)\n'
                else:
                    event_series+=f'{i}. {timestampe} {event} ({wp1p}%-{wp2p}%)\n'
            else:
                if condition['score']:
                    event_series+=f'{i}. {event} ({s1}-{s2}) ({wp1p}%-{wp2p}%)\n'
                else:
                    event_series+=f'{i}. {event} ({wp1p}%-{wp2p}%)\n'

    return event_series , ground_truth , steps , score_gap , trends_wp_A 

def transform_ts_trends(series):
    if len(series)>3 : 
        series = series[:-1]
    trend = []
    for i in range(1, len(series)):
        if series[i] > series[i - 1]:
            trend.append("+")
        elif series[i] < series[i - 1]:
            trend.append("-")
    summarized_trend = []
    for i in range(len(trend)):
        if i == 0 or trend[i] != trend[i - 1]:
            summarized_trend.append(trend[i])
    return "".join(summarized_trend)


def pick_neg_opt(datasets , series_len):
    
    # Pick a random Game
    game = random.choice(list(datasets.keys()))
    datas = datasets[game]
    
    # Pick random game point
    opt_index = random.randint(series_len+1, len(datas)-series_len-1)  
    
    score_gap = abs(int(datas[opt_index][SCORE1]) - int(datas[opt_index][SCORE2]))
    lts = [float(datas[opt_index+i-1][WP1]) for i in range(series_len+2)]
    select_trend = transform_ts_trends(lts)
    return datas , opt_index , score_gap , lts , select_trend
    

def similarity_metric(seriesA, seriesB):
    def z_score_normalize(series):
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std
    normalized_A = z_score_normalize(seriesA)
    normalized_B = z_score_normalize(seriesB)
    # print(seriesA , normalized_A )
    # return np.mean(np.abs(normalized_A - normalized_B))
    return np.linalg.norm(normalized_A - normalized_B)

def diff_metric(trends1, trends2, nums_of_events=10):
    similarity = 0 
    # 0-7
    activity = (len(trends1)+len(trends2)) // 2 
    if trends1 == trends2:
        similarity=10
    return activity + similarity

import matplotlib.pyplot as plt
def draw_distribution(data , game=''):
    print(len(data) , np.min(data) ,  np.mean(data),  np.max(data))
    
    data = np.array(data) 
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
    
    # 添加图表标签和标题
    plt.xlabel("Value (0-1)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of Values (0-1)", fontsize=16)

    # 显示图表
    plt.tight_layout()
    # plt.savefig('./dist_nba.png')
    plt.savefig(f'./dist_{game}.png')
    

def remove_nums(event):
    a= ["\u03B1" ,"\u03B2" , "\u03B3" , "\u03B4" ,  "\u03B5" ,"\u03B6" , "\u03B8" , "\u03BB" ]
    b= ["\u0391" ,"\u0392" ,"\u0393" ,  "\u0394" ,  "\u0395" ,  "\u0396" ,"\u0398", "\u039B" ]
    candidate =  a + b 
    numbers1 = re.findall(r'\b\d{2,7}(?:,\d{3})*\b', event)
    numbers2 = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', event)
    numbers3 = re.findall(r'[a-zA-Z$]+\d+(?:\.\d+)?[a-zA-Z$]+', event)
    numbers = numbers1 + numbers2 + numbers3
    numbers = sorted(numbers, key=len, reverse=True)
    for i in range(len(numbers)):
        try:
            event = event.replace(numbers[i] ,candidate[i])
        except:
            return event
    return event

def pick_title_wo_nums(events):
    def extract_nums_from_news(news):
        pattern1 = r'\b\d{1,3},\d{3}\b' 
        pattern2 = r'\b\d{5}\b'
        numbers1 = re.findall(pattern1, news)
        numbers2 = re.findall(pattern2, news)
        cleaned_numbers1 = [int(num.replace(',', '')) for num in numbers1]
        cleaned_numbers2 = [int(num.replace(',', '')) for num in numbers2]
        return cleaned_numbers1 + cleaned_numbers2 
    for i in range(len(events)):
        tilte = events[i]
        nums = extract_nums_from_news(tilte)        
        if len(nums) == 0: 
            return tilte
    return tilte
