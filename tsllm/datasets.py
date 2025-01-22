import pandas as pd
import os 
from utils import mismatch , corr_mismatch , bootstrap_mean 
import csv 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import random
# Configure logging
import logging
# logger = logging.getLogger()

def get_datasets(config , logger ):
    if 'cleaning' in config.experiment.task:
        clean_dataset_nba(config , logger )
    if "calibration" in config.experiment.task :
        game_res , pred_res = get_calibrate_results()
        calibrate(game_res , pred_res)
    if 'build' in config.experiment.task :
        if 'nba' in config.experiment.data_path:
            return load_nba_datasets(config)  
        if 'nfl' in config.experiment.data_path:
            return load_nfl_datasets(config)  
        if 'openw' in config.experiment.data_path:
            if 'crypto' in config.experiment.data_path:
                return load_crypto_datasets(config)  
            if 'trading' in config.experiment.data_path:
                return load_trading_datasets(config)  
            if 'health_US' in config.experiment.data_path:
                return load_health_datasets(config)  
            if 'energy' in config.experiment.data_path:
                return load_energy_datasets(config)  
import json 

def load_energy_datasets(config):
    def get_events_from_news(start_date):
        events = []
        with open(config.experiment.data_path + 'Energy_search.csv', "r", encoding="utf-8") as file:
            file_news = csv.reader(file)
            for row in file_news:
                if start_date == row[1]:
                    events.append(row[3])
        return events

    data =  pd.read_csv(config.experiment.data_path + 'Energy.csv')
    timeseries = list(data["OT"].values)
    start_dates = list(data["date"].values)
    
    datasets = {}
    for i in range(len(timeseries)):
        
        start_date = start_dates[i]
        events = get_events_from_news(start_date)

        datasets[start_date] ={
            'nums' :  round(timeseries[i] , 3 )  , 
            'events' : events
        }
    print('There are #' ,  len(datasets.keys()))
    return datasets

    
def load_health_datasets(config):

    def get_events_from_news(start_date):
        events = []
        with open(config.experiment.data_path + 'Health_US_search.csv', "r", encoding="utf-8") as file:
            file_news = csv.reader(file)
            for row in file_news:
                if start_date == row[1]:
                    events.append(row[3])
        return events

    data =  pd.read_csv(config.experiment.data_path + 'Health_US.csv')
    timeseries = list(data["OT"].values)
    start_dates = list(data["start_date"].values)
        
    datasets = {}
    for i in range(len(timeseries)):
        
        start_date = start_dates[i]
        events = get_events_from_news(start_date)

        datasets[start_date] ={
            'nums' :  round(timeseries[i] , 2 )  , 
            'events' : events
        }
    print('There are #' ,  len(datasets.keys()))
    return datasets
    
def load_trading_datasets(config):
    def convert_dates_to_10th(dates):
        converted_dates = []
        for date_str in dates:
            date_obj = datetime.strptime(date_str, "%B %Y")
            converted_date = date_obj.strftime("%Y-%m")
            converted_dates.append(converted_date)
        return converted_dates

    def get_events_from_news(nums_date):
        events = []
        with open(config.experiment.data_path + 'Economy_search.csv', "r", encoding="utf-8") as file:
            file_news = csv.reader(file)
            for row in file_news:
                if nums_date == row[1][:7]:
                    events.append(row[3])
        return events 

    prices =  pd.read_csv(config.experiment.data_path + 'Economy.csv')
    exports = list(prices["Exports"].values)[::-1]
    imports = list(prices["Imports"].values)[::-1]
    price_detailed_dates = convert_dates_to_10th(list(prices["Month"].values)[::-1])

    assert len(imports) == len(price_detailed_dates) == len(exports)
    
    datasets = {}
    for i in range(len(exports)):
        
        nums_date = price_detailed_dates[i]

        events = get_events_from_news( nums_date )
        
        datasets[nums_date] ={
            'nums' : f"{int(imports[i])} , {int(exports[i])}"  , 
            'events' : events
        }
        
    print('There are #' ,  len(datasets.keys()))
    return datasets
    
    
def load_crypto_datasets(config):
    datasets = {}
    prices =  pd.read_csv(config.experiment.data_path + 'bitcoin_daily_price.csv')
    close_prices = list(prices["close"].values)[::-1]
    price_detailed_dates = list(prices["timeOpen"].values)[::-1]
    
    assert len(close_prices) == len(price_detailed_dates)
    for i in range(len(close_prices)):
        
        nums_date = price_detailed_dates[i].split("T")[0]
        
        with open(f'{config.experiment.data_path}/bitcoin_news/{nums_date}.json', 'r', encoding='utf-8') as file:
            data_list = json.load(file)
            
        titles = [item['title'] for item in data_list ]

        datasets[nums_date] ={
            'nums' : int(round(close_prices[i] , 0)) , 
            'events' : titles
        }
        
    print('There are #' ,  len(datasets.keys()))
    return datasets

def load_nfl_datasets(config):
    keys =[]
    values = []

    lst= os.listdir(config.experiment.data_path)
    cutting_date='Sep 01, 2024'
    
    for file_name in lst:
        
        if datetime.strptime(file_name.split('_')[-1].split(".")[0], "%b %d, %Y") <  datetime.strptime(cutting_date , "%b %d, %Y") :
            continue
        
        game = []
        file_path = os.path.join(config.experiment.data_path ,file_name )
        
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)
            game.append(column_names)
                    
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                values_ = list(row.values()) 
                assert None not in values_ 
                game.append(values_)

        keys.append(file_name)
        values.append(game)
            
        if len(values) == config.experiment.num_of_sampels:
            break 
    print(f'There are totally #{len(values)} games!')
    return dict(zip(keys,values))

def get_calibrate_results():
    data_paths = [ "./datasets/nba_timeseries/23-24/chart/" , "./datasets/nba_timeseries/22-23/chart/" ] 
    game_res = []
    pred_res = []
    for data_path in data_paths:
        for file_name in os.listdir(data_path): 
            # try : 
            if 'DS_Store' in file_name or 'modify' in file_name : continue #For Mac 
            file_path = os.path.join(data_path ,file_name )
            try:
                with open(file_path, mode='r') as file:
                    csv_reader = csv.reader(file)
                    # ['time', 'event', 'SA_Score', 'PHX_Score', 'wp_SA', 'wp_PHX']
                    column_names = next(csv_reader)
            except:
                print(file_name)
                exit
            team1 = column_names[2].split('_')[0]
            team2 = column_names[3].split('_')[0]
            
            df = pd.read_csv(file_path)
            if np.isnan(df['wp_{}'.format(team1)].values).any() :  
                print(file_name , 'Nan!' )
                continue
            win_t1 = 1 if df['{}_Score'.format(team1)].values[-1] > df['{}_Score'.format(team2)].values[-1] else 0 
            wp_ratio = len([x for x in df['wp_{}'.format(team1)].values if x > 0.500])/len(df['wp_{}'.format(team1)].values)

            game_res.append(win_t1) ; game_res.append(1-win_t1)
            pred_res.append(wp_ratio) ; pred_res.append(1.-wp_ratio)
            # except:
            #     print(file_name , 'err')
            #     continue
    return game_res  , pred_res 

def draw_bss(bss_s):
    
    p25 = len([x for x in bss_s if x > 0.25]) / len(bss_s)
    p50 = len([x for x in bss_s if x > 0.50]) / len(bss_s)
    p75 = len([x for x in bss_s if x > 0.75]) / len(bss_s)
    
    lins_bin = [-3 , -2 , -1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ]
    # Calculate histogram values for each bin
    hist, bin_edges = np.histogram(bss_s, bins=lins_bin)

    # Bar chart
    plt.bar(bin_edges[:-1], hist, width=0.2, edgecolor='black', align='edge')

    ft = 16
    # Add labels and title
    plt.xlabel('Bins (BSS)' , fontsize = ft)
    plt.ylabel('# Games' , fontsize = ft)
    plt.title('BSS distribution of ESPN NBA WP' , fontsize = ft)
    plt.legend(["# total :{} \n{} games bss > 0.25 \n{} game bss > 0.50 \n{} game bss > 0.75".format(len(bss_s) , round(p25,2) , round(p50,2), round(p75,2) )], loc="upper left" , fontsize = ft)
    # Display the chart
    plt.show()

def get_BSS_results():
    data_paths = [ "./datasets/nba_timeseries/23-24/chart/" , "./datasets/nba_timeseries/22-23/chart/" ] 
    bss_s = []
    file_names=  []
    count = 0 
    count_nan = 0 
    for data_path in data_paths:
        for file_name in os.listdir(data_path): 
            if 'DS_Store' in file_name or 'modify' in file_name : continue #For Mac 
            
            file_path = os.path.join(data_path ,file_name )
            with open(file_path, mode='r') as file:
                csv_reader = csv.reader(file)
                # ['time', 'event', 'SA_Score', 'PHX_Score', 'wp_SA', 'wp_PHX']
                column_names = next(csv_reader)
                
            team1 = column_names[2].split('_')[0]
            team2 = column_names[3].split('_')[0]
            
            df = pd.read_csv(file_path)
            win_t1 = 1 if df['{}_Score'.format(team1)].values[-1] > df['{}_Score'.format(team2)].values[-1] else 0 
            wp_t1  = df['wp_{}'.format(team1)].values - win_t1 
            # remove Nan
            has_nan = np.isnan(wp_t1).any()
            if has_nan :  
                count_nan +=1 
                print('There is Nan in the Game',file_name , count_nan )
                continue
                
            bss = 1-np.mean(wp_t1 ** 2) / 0.25 
            bss_s.append(bss)
            file_names.append(file_name)
            
            # Print some exception 
            # 385 -1.978931991011236 PHX_LAL_Dec 5, 2023.csv
            if count == 385 : 
                print(bss , file_name , win_t1 ,team1  )
                print(max(wp_t1) , min(wp_t1))
                # exit()
            count += 1 
    
    # Best 
    min_indices = np.argsort(bss_s)[-5:]
    print('Best : ' , min_indices)
    for mi in min_indices:
        print(mi , bss_s[mi] , file_names[mi])
        
    # Worest  
    min_indices = np.argsort(bss_s)[:5]
    print('Worest : ' , min_indices)
    for mi in min_indices:
        print(mi , bss_s[mi] , file_names[mi])
    return bss_s

def calibrate(game_res , pred_res):
    ratios = np.linspace(0,1,11)
    bins = {i: [] for i in range(10)}
    # print(game_res[0:20]) 
    print(pred_res[0:20])
    for i in range(len(pred_res)):
        if pred_res[i] < 0.0 or pred_res[i] > 1.0: exit()

        loc = int(pred_res[i]*10)
        if loc == 10 : loc = 9 
        bins[loc].append(game_res[i])

    for i in range(len(bins)): 
        bootstrap_means, ci_lower, ci_upper = bootstrap_mean(bins[i])
        
        print(f'{ratios[i]:.1f}--{ratios[i+1]:.1f}   {np.mean(bootstrap_means):.3f}   [{ci_lower:.3f},{ci_upper:.3f}]   {len(bins[i])}')

def load_nba_datasets(config):
        
    keys =[]
    values = []
    count= 0 
    cutting_date='Dec 01, 2024'
    
    lst= os.listdir(config.experiment.data_path)
    
    print(config.experiment.data_path ,  len(lst))
    
    for file_name in lst:
        if datetime.strptime(file_name.split('_')[-1].split(".")[0], "%b %d, %Y") <  datetime.strptime(cutting_date , "%b %d, %Y") :
            continue
        count +=1
        game = []
        file_path = os.path.join(config.experiment.data_path ,file_name )
        
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)
            game.append(column_names)
                    
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                values_ = list(row.values()) 
                assert None not in values_ 
                game.append(values_)

        keys.append(file_name)
        values.append(game)
            
        if config.experiment.game_name != '' :
            assert len(values) == 1 
            return dict(zip(keys,values))

        if len(values) == config.experiment.num_of_sampels:
            break
    print(f'There are totally #{len(values)} games!')
    return dict(zip(keys,values))
    
def clean_dataset_nba(config , logger ):
    # data_path: "./datasets/nba_timeseries/23-24/chart/"
    is_wrong = False 
    for file_name in os.listdir(config.experiment.data_path): 
        file_path = os.path.join(config.experiment.data_path , file_name)
        if '24' not in file_name and '23' not in file_name and '22' not in file_name : continue 
        has_mm , new_rows = mismatch(file_path) 
        if has_mm : 
            corr_mismatch(config.experiment.data_path ,file_name  , new_rows , logger)
            is_wrong=True
    if is_wrong == False: print('There is nothing wrong')

def load_chess_datasets(config):
        
    keys =[]
    values = []

    print(config.experiment.data_path)
    count = 0 
    while True:
        for file_name in os.listdir(config.experiment.data_path):
            game = []
            file_path = os.path.join(config.experiment.data_path ,file_name )
            df = pd.read_csv(file_path, encoding='utf-8')
            
            for _, row in df.iterrows():
                game.append([row['Position (FEN)'] , row['Move'], row['Centipawn Score'] , row['Holder'] , row['FEN']]) 
            
            keys.append(file_name.split('.')[0]+f'_{count}')
            values.append(game)

            if len(keys)  == 100 : 
                return dict(zip(keys,values)) 
            count +=1 
    return dict(zip(keys,values))
        
        