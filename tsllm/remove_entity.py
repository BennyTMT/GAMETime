import os 
import csv 
import numpy as np 
import matplotlib.pyplot as plt
import re 

SCOREA= 2 
SCOREB= 3
EVENT= 1
SUM=0

def get_team_players_relations_nba(game_name , data_path):
    def get_players(inputs):
        pattern = r'\b[A-Z][^\s]* [A-Z][^\s]*\b'
        matches = re.findall(pattern, inputs )
        return matches
    team_A = []
    team_B = []
    file_path = os.path.join(data_path ,game_name )

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
    
def detect_all_nba_games(nameA,nameB,uc_player_name,game_name , data_path):
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
                        if nameA in file_name  : 
                            return 'Player from team A'
                        if nameB in file_name  :
                            return 'Player from team B'
    return None

def detect_all_nfl_games(nameA,nameB,player_name,game_name):
    
    data_path = '/scratch/wtd3gz/project_TS/llm_game_ts/datasets/nfl_timeseries/real_name_games/'
    
    def get_players(inputs):
        pattern = r'\b[A-Z]\.[A-Z][a-z]+\b'
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
                    if player_name in matches:
                        if nameA in file_name  : 
                            return 'A'
                        if nameB in file_name  :
                            return 'B'
    return None

def detect_excpetion(m):
    exceptions = ['CALL' , 'COACH\'S', 'Coach', 'DO', 'OT' ,  'Full' , 'REF-INITIATED' , 'SUPPORTS', 'Cavaliers']
    for mm in m.split(' '):
        if mm in exceptions : 
            return True
    return False

    
def make_nba_relationships():
    # data_path="./datasets/nba_timeseries/23-24/chart/"
    data_path="./datasets/nba_timeseries/24-25/chart/"
    for file_name in os.listdir(data_path):
        if file_name == 'modify' : continue 
        # output_file = os.path.join("./datasets/nba_timeseries/23-24-fakename/" ,file_name ) 
        output_file = os.path.join("./datasets/nba_timeseries/24-25-fakename/" ,file_name ) 
        if os.path.exists(output_file):
            print('done')
            continue    

        print(file_name)
        game = []
        file_path = os.path.join(data_path ,file_name )

        teamA , teamB , nameA , nameB = get_team_players_relations_nba(file_name , data_path)
        
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)
            replace_col = []
            for c in column_names : 
                c = c.replace(nameA , 'team A')
                c = c.replace(nameB , 'team B')
                replace_col.append(c)
            game.append(replace_col)
            
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                values_row = list(row.values())
                values_ =  values_row[EVENT]

                pattern = r'\b[A-Z][^\s]* [A-Z][^\s]*\b'
                matches = re.findall(pattern, values_ )
                matches = set(matches)

                for m in matches:
                    m = m.replace('\'s' , '' )
                    replace_name= None 
                    if m in teamA : 
                        replace_name = 'Player from team A'
                    elif m in teamB:
                        replace_name = 'Player from team B'
                    else:
                        if detect_excpetion(m) : 
                            continue 
                        else :
                            replace_name = detect_all_nba_games(nameA,nameB,m,file_name , data_path)        

                    if replace_name is None :
                        print('err in ' , file_name , m  ) 
                        replace_name = 'A Player'
                        
                    values_ = values_.replace(m, replace_name )
                    
                values_row[EVENT] = values_
                game.append(values_row)

        with open(output_file, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(game)

make_nba_relationships()





def get_team_players_relations_nfl(game_name):
    data_path="./datasets/nfl_timeseries/real_name_games/"
    def get_players(inputs):
        pattern = r'\b[A-Z]\.[A-Z][a-z]+\b'
        matches = re.findall(pattern, inputs )
        return matches
    file_path = os.path.join(data_path ,game_name )
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        column_names = next(csv_reader)
    teamA = column_names[3].split('_')[0]
    teamB = column_names[4].split('_')[0]
    team_A = []
    team_B = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            values_ = list(row.values()) 
            matches = get_players(values_[1])
            for m in matches :        
                if m not in team_A and m not in team_B:
                    team = detect_all_nfl_games(teamA,teamB,m,game_name)
                    if team == None : 
                        print( 'err in searching player : ', m , game_name)
                    if team == 'A': team_A.append(m)
                    if team == 'B': team_B.append(m)
    return set(team_A) , set(team_B) , teamA , teamB

def make_nfl_relationships():
    data_path="./datasets/nfl_timeseries/real_name_games/"
    for file_name in os.listdir(data_path):
        output_file = os.path.join("./datasets/nfl_timeseries/fake_name_games/" ,file_name ) 
        
        if os.path.exists(output_file):
            print('done')
            continue
            
        print(file_name)
        teamA , teamB , nameA , nameB = get_team_players_relations_nfl(file_name)
        game = []
        file_path = os.path.join(data_path ,file_name )
        
        # table title 
        with open(file_path , mode='r') as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)
            replace_col = []
            for c in column_names : 
                if c =='state': continue 
                c = c.replace(nameA , 'team A')
                c = c.replace(nameB , 'team B')
                replace_col.append(c)
            game.append(replace_col)

        # table event  
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                values_row = list(row.values())
                values_ =  values_row[1] + f' ({values_row[2]})'

                if nameA in values_:
                    values_ = values_.replace(nameA , 'Team A')
                if  nameB in values_:
                    values_ = values_.replace(nameB , 'Team B')
                
                pattern = r'\b[A-Z]\.[A-Z][a-z]+\b'
                matches = re.findall(pattern, values_ )
                matches = set(matches)
                
                for m in matches:
                    replace_name = 'A certain player' 
                    if m in teamA : 
                        replace_name = 'Player from team A'
                    elif m in teamB:
                        replace_name = 'Player from team B'
                    values_ = values_.replace(m, replace_name )
                    
                
                new_value_row = [values_row[0] , values_  , values_row[3] , values_row[4] , values_row[5] , values_row[6]]
                game.append(new_value_row)
            
        with open(output_file, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(game)

# make_nfl_relationships()

# path_ = "./datasets/nba_timeseries/23-24-fakename/" 
# print(len(os.listdir(path_)))