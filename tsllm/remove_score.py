import numpy as np 
import json
import os 
import random
import re 
import csv 

TASK='blank_rorder'

import string

SCOREA= 2 
SCOREB= 3
EVENT= 1

def further_detect(nameA,nameB,uc_player_name,game_name,opt):
    data_path = '/scratch/wtd3gz/project_TS/llm_game_ts/datasets/nba_timeseries/23-24/chart/'
    def get_players(inputs):
        pattern = r'\b[A-Z][^\s]* [A-Z][^\s]*\b'
        matches = re.findall(pattern, inputs )
        return matches

    count = 0 
    for file_name in os.listdir(data_path):
        if file_name == game_name : continue 
        if nameA in file_name or nameB in file_name:
            count +=1 
            with open(os.path.join(data_path ,file_name ), mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    values_ = list(row.values()) 
                    matches = get_players(values_[EVENT])
                    if uc_player_name in matches:
                        # print(nameA , nameB , file_name )
                        if nameA in file_name  : 
                            return 'Player from team A'
                        if nameB in file_name  :
                            return 'Player from team B'
            if 'shf_d' in opt and count == 3:
                 return None 
    return None 

def get_team_players_relations(game_name):
    data_path = '/scratch/wtd3gz/project_TS/llm_game_ts/datasets/nba_timeseries/23-24/chart/'
    def get_players(inputs):
        pattern = r'\b[A-Z][^\s]* [A-Z][^\s]*\b'
        matches = re.findall(pattern, inputs )
        return matches
    team_A = []
    team_B = []
    for file_name in os.listdir(data_path):
        if file_name != game_name : continue 
        file_path = os.path.join(data_path ,file_name )

        last_score_A = -1
        last_score_B = -1

        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                values_ = list(row.values()) 
                
                if last_score_A == -1 : 
                    last_score_A = int(values_[SCOREA])
                    last_score_B = int(values_[SCOREB])
                    continue
                else: 
                    if last_score_A != int(values_[SCOREA]):
                        team_A.extend(get_players(values_[EVENT]))
                    if last_score_B != int(values_[SCOREB]):
                        team_B.extend(get_players(values_[EVENT])) 

                    last_score_A = int(values_[SCOREA])
                    last_score_B = int(values_[SCOREB])
        
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)

        return set(team_A) , set(team_B) ,column_names[SCOREA].split('_')[0], column_names[SCOREB].split('_')[0]
        

def drop_scores_in_events(inputs):
    pattern = r'\(\d{1,3}-\d{1,3}\)'
    matches = re.findall(pattern, inputs)
    scores = set(matches)
    for m in scores:
        inputs = inputs.replace(m, '' )
    inputs = inputs.replace(', scores', '' )
    return inputs 

def replace_teams_to_random(inputs):
    pattern = r'\b[A-Z]{2,4}-[A-Z]{2,4}\b'
    matches = re.findall(pattern, inputs )
    set_m = set(matches)
    assert len(set_m) ==1
    random_string1 = ''.join(random.choices(string.ascii_uppercase, k=3))
    random_string2 = ''.join(random.choices(string.ascii_uppercase, k=3))
    random_string =f"{random_string1}-{random_string2}"
    inputs = inputs.replace(matches[0], random_string  )
    return inputs
    
def replace_players_to_random(inputs):
    name_list = ["John Mitchell", "James Anderson", "Michael Collins", "Robert Hayes", "David Thompson", "William Scott", "Richard Bennett", "Charles Foster", "Joseph Wright", "Christopher Murphy", "Daniel Carter", "Matthew Morgan", "Anthony Rivera", "Mark Sullivan", "Steven Perry", "Andrew Bryant", "Paul Jenkins", "Joshua Cook", "Thomas Edwards" , "John Smith", "Michael Johnson", "David Williams", "James Brown", "Robert Jones", "William Miller", "Daniel Davis", "Matthew Garcia", "Christopher Wilson", "Joseph Martinez", "Thomas Anderson", "Mark Taylor", "Charles Thomas", "Steven Moore", "Andrew White", "Paul Harris", "Kevin Clark", "Brian Lewis", "Jason Robinson", "Eric Walker", "Anthony Hall", "Scott Young", "Ryan King", "Jonathan Wright", "Justin Green", "Joshua Adams", "Benjamin Baker", "Samuel Gonzalez", "Alexander Nelson", "Adam Carter", "Kyle Mitchell", "Brandon Perez", "Jacob Roberts", "Zachary Turner", "Dylan Phillips", "Nathan Campbell", "Ethan Parker", "Aaron Evans", "Logan Edwards", "Mason Collins", "Caleb Stewart" , "Emily Johnson", "Jessica Lee", "Sarah Brown", "Amanda Clark", "Ashley Lewis", "Megan Hall", "Laura Allen", "Nicole Young", "Rachel King", "Samantha Harris", "Hannah Wright", "Julia Hill", "Olivia Scott", "Sophia Green", "Emma Roberts", "Grace Turner", "Chloe Collins", "Alyssa Campbell", "Brittany Morgan", "Madison Nelson"]
    # pattern = r'[A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z][a-z]+(?:-[A-Z][a-z]+)+'
    pattern = r'\b[A-Z][^\s]* [A-Z][^\s]*\b'
    matches = re.findall(pattern, inputs )
    matches = set(matches)
    for m in matches:
        random_name = random.choice(name_list)
        inputs = inputs.replace(m, random_name )
    return inputs    

def replace_player_team_keep_relation(inputs ,game_name ,opt ):

    def detect_excpetion(m):
        exceptions = ['CALL' , 'COACH\'S', 'Coach', 'DO', 'OT' ,  'Full' , 'REF-INITIATED' , 'SUPPORTS', 'Cavaliers']
        for mm in m.split(' '):
            if mm in exceptions : 
                return True
        return False
        
    def remove_extra_name(inputs):
        for extra in ['Jr.', 'III' , 'II']:
            if extra in inputs:
                inputs = inputs.replace(extra , '' )
        return inputs
        
    pattern = r'\b[A-Z][^\s]* [A-Z][^\s]*\b'
    matches = re.findall(pattern, inputs )
    matches = set(matches)
    
    teamA , teamB , nameA , nameB = get_team_players_relations(game_name)

    for m in matches:
        m = m.replace('\'s' , '' )
        if m in teamA : 
            replace_name = 'Player from team A'
        elif m in teamB:
            replace_name = 'Player from team B'
        else:
            if detect_excpetion(m) : 
                continue 
            else : 
                replace_name = further_detect(nameA,nameB,m,game_name , opt)
                if replace_name is None : 
                    replace_name = random.choice(['Player from team A', 'Player from team B'])

        inputs = inputs.replace(m, replace_name )
            
    inputs = remove_extra_name(inputs)
    
    pattern = r'\b[A-Z]{2,4}-[A-Z]{2,4}\b'
    matches = re.findall(pattern, inputs )
    set_m = set(matches)
    assert len(set_m) ==1
    replace_string = "team A-team B"
    inputs = inputs.replace(matches[0], replace_string )
        
    return inputs

def init_options():
    TASKOPTS = {}
    
    # for opt in ['magn_0' , 'magn_1' , 'magn_2' , 'magn_3' , 'magn_4'  ]:
    #     for series_len in [3]:      
    #         blank_rorder.append(f'{opt}_{series_len}')
    # TASKOPTS['blank_rorder'] = blank_rorder
    
    # blank_rorder = []
    # for opt in ['easy' , 'easy1' , 'easy2' , 'hard1' , 'hard2' , 'hard3' ]:
    #         for series_len in [1, 3 , 5]:
    #             blank_rorder.append(f'{opt}_{series_len}')
    # TASKOPTS['blank_rorder'] = blank_rorder
    
    # blank_rorder = []
    # for series_len in [1]:
    #       for opt in [ 'wo_cot' , 'w_cot']:
    #           blank_rorder.append(f'{opt}_{series_len}')
    # TASKOPTS['blank_rorder'] = blank_rorder
    
    # blank_rorder = []
    # for series_len in [1 , 5 , 10 , 15 , 20]:
    #     for opt in ['nfl' , 'nba' ]:
    #         blank_rorder.append(f'{opt}_{series_len}')
    # TASKOPTS['blank_rorder'] = blank_rorder
    
    blank_rorder = []
    for series_len in [1, 2 ,3, 4 , 5 , 6 , 7]:
        for opt in [ 'nba' , 'nfl' ]:
            blank_rorder.append(f'{opt}_{series_len}')
    TASKOPTS['blank_rorder'] = blank_rorder
        
    return TASKOPTS

TASKOPTS = init_options()
RMSCORE = True 
PRINT = True

for file_name in TASKOPTS[TASK]:
    
    print(file_name)
    file_path = f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{file_name}.json'
    with open(file_path , 'r', encoding='utf-8') as f:
        data = json.load(f) 

    for game_name in data:
        # game_name='HOU_MIL_Dec 17, 2023.csv'
        if PRINT:
            print('label:' , data[game_name]['label'])
            print(data[game_name]['sys_prompt'])
            print(data[game_name]['input'])
            
        # data[game_name]['input'] = replace_player_team_keep_relation(data[game_name]['input'] , game_name , file_name)

        if RMSCORE : 
            data[game_name]['input'] = drop_scores_in_events(data[game_name]['input'])
        data[game_name]['sys_prompt'] = data[game_name]['sys_prompt'].replace(', scores,', '' )

        if PRINT:
            print(data[game_name]['sys_prompt'])
            print(data[game_name]['input'])
            print(data[game_name]['label'])
            print('-'*50)
            PRINT = False 
                
    if RMSCORE : 
        file_path_ = f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{file_name}_n_s.json'
    else :
        file_path_ = f'/scratch/wtd3gz/project_TS/llm_game_ts/prompts/{TASK}/{file_name}_n.json'
    with open(file_path_, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
'''
    python3 ./tsllm/replace_obj.py 
'''