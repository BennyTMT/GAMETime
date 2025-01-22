import torch
import os.path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass
import random 
STEP_MULTIPLIER = 1.2
import logging

@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.
    
    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.
    
    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.
        
    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:   
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            x = (x - min_) / q
            print(np.min(x) , np.max(x))
            exit()
            return x
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def truncate_input(input_arr, input_str , describtion , config, tokenizer=None  ):
    """
    Truncate inputs to the maximum context length for a given model.
    
    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    if tokenizer is not None and config.model.context_lengths is not None : 
        tokenization_fn = tokenizer
        context_length  = config.model.context_lengths
        input_str_chuncks = input_str.split(config.model.settings['time_sep'] )
        has_truncated = False 
        
        for i in range(len(input_str_chuncks) - 1):
            truncated_input_str = config.model.settings['time_sep'].join(input_str_chuncks[i:])
            
            if not truncated_input_str.endswith(config.model.settings['time_sep']):
                # add separator if not already present
                truncated_input_str += config.model.settings['time_sep']
            
            if describtion != '' : 
                num_descri_tokens =len(tokenization_fn(describtion)) 
            else : 
                num_descri_tokens = 0 
                
            input_tokens = tokenization_fn(truncated_input_str)
            num_series_tokens = len(input_tokens) 
                
            avg_token_length = num_series_tokens / (len(input_str_chuncks) - i)
            num_output_tokens = avg_token_length * config.model.test_len * STEP_MULTIPLIER
            
            num_input_toekns = num_descri_tokens + num_series_tokens
            
            if num_input_toekns + num_output_tokens  <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if i > 0:
            has_truncated = True 
            print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
        if config.is_test_context_length:
            return has_truncated
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str

def hardSigma(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output
    
def printParams(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    
def makedir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except:
        pass
        
def build_save_path(config):
    game_year = config.experiment.data_path.split('/')[-3]
    save_dir = f'output/{config.model.name}-{config.experiment.task}/{game_year}'
    if not os.path.exists(save_dir) : os.makedirs(save_dir)
    return save_dir
    
def is_completion(save_dir , dsname ):
    if os.path.exists(f'{save_dir}/{dsname}.pkl'):
        print("{} has done!".format(dsname)) ; 
        return True
    else: 
        return False 

import csv
def mismatch(file_path):
    # Open the CSV file for reading
    new_rows = []
    has_mm = False 
    with open(file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        last_row = None      
        for row in csv_reader:
            values_ = list(row.values()) 
            if  '' in values_ or None in values_:
                has_mm = True 
                if '' in last_row or None in last_row:
                    new_rows.append([last_row[0] , last_row[1] ] + values_[1:5])
            else:
                new_rows.append(values_)
            last_row = values_
    return has_mm , new_rows

def corr_mismatch(data_path, file_name, new_rows  , logger ):
    # data_path: "./datasets/nba_timeseries/23-24/chart/"
    with open(data_path+file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        column_names = next(csv_reader)
        
    with open(data_path+file_name, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(column_names)
        for row in new_rows:
            csv_writer.writerow(row)

    logger.info(data_path[26:]+file_name+' [CLEAN]')

def setup_logging(filename=''):
    if filename=='':
        filename='/scratch/wtd3gz/project_TS/llm_game_ts/log/sports.log'
    logger = logging.getLogger('sports_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename , mode="a" )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s' , datefmt='%Y-%m-%d %H:%M:%S' )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
    

# Function to perform bootstrapping
def bootstrap_mean(data, n_bootstrap=1000):
    bootstrap_means = []
    n = len(data)
    # Perform bootstrap sampling
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        # Compute mean of the sample
        bootstrap_means.append(np.mean(sample))
    # Convert list to numpy array for convenience
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute the 95% confidence interval (2.5th and 97.5th percentiles)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return bootstrap_means, ci_lower, ci_upper

import json 
        
def read_prompts(config, file_name):
    file_path = f'./prompts/{config.experiment.task}/{file_name}'
    print('Reading prompt from:', file_path)
    assert os.path.exists(file_path)
    with open(file_path , 'r', encoding='utf-8') as f:
        data = json.load(f) 
    return data
    
def dump_response(config,key_name,llm_resp):
    record_file = './outputs/{}/{}.json'.format(config.experiment.task,config.model.name)
    data={}
    if os.path.exists(record_file ):
        # exists, adding a new result / or updating results 
        with open(record_file  , 'r', encoding='utf-8') as f:
            data = json.load(f) 
            data[key_name] = llm_resp
    else:
        # not exists, starting one 
        data[key_name] = llm_resp
    # dump to the file 
    with open(record_file , 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def exist_in_response(config,key_name):
    record_file = './outputs/{}/{}.json'.format(config.experiment.task,config.model.name)
    if os.path.exists(record_file ):
        with open(record_file  , 'r', encoding='utf-8') as f:
            data = json.load(f) 
        if key_name in data: return True
        else : return False
    else:  return False
        
import re 
        
def match_list(task, context , pre_len=16):
    # Only Select a b c d 
    pattern = r'\*\*(a|b|c|d)\*\*'
    matches = re.findall(pattern, context)
    if len(matches) >0 : 
        return matches[-1]
    else :
        return None 

def match_list_old(task, context , pre_len=16):
    
    if  task == 'fill_blank':
        pattern = r'\*\*(a|b|c|d)\*\*'
        matches = re.findall(pattern, context)
        if len(matches) >0 : 
            return matches[-1]
        else :
            return None 

    if  task == 'middle_event':
        pattern = r'\*\*([a-zA-Z](?:, ?[a-zA-Z])*)\*\*'
        matches = re.findall(pattern, context)
        if len(matches) >0 : 
            return matches[0]
        else :
            return None

    if  task == 'game_traceback':
        pattern = r'\*\*(0|1)\*\*'
        matches = re.findall(pattern, context)
        if len(matches) >0 : 
            return matches[0]
        else :
            return None
            
    if  task == 'order_of_seq':
        pattern = r'\*\*([\d, ]+)\*\*'
        matches = re.findall(pattern, context)
        if len(matches) >0 : 
            return matches[0]
        else :
            return None
    
    if task == 'reason-event' or task == 'reason-actions'  :
        pattern = r'\*\*(.*?)\*\*'
        match = re.search(pattern, context)
        if match : 
            return match.group(1)
        else : 
            return None
        
def match_percentage(context , pre_len = 16 , task='' ):
    if task == 'forecast' : 
        pattern = r'\d+(\.\d+)?%'
        matches = [match.group(0) for match in re.finditer(pattern, context)]
        pred =  [float(m.replace('%','')) for m in matches]
    elif task == 'prdScore' or task == 'pick3imp' : 
        pattern = r'\b\d{2}\b'
        matches = re.findall(pattern, context)
        pred = [int(match) for match in matches]

    if len(pred) > pre_len : 
        return pred[:pre_len]
    elif len(pred) == 0 :
        return [50.0] * pre_len 
    else :  
        return pred + [pred[-1]]*(pre_len-len(pred))
        
def record_llm_process(record ,model_name , answer , ground_truth , task, pre_len=16):
    if not os.path.exists(f'./outputs/{task}'):
        os.makedirs(f'./outputs/{task}')
    with open(f'./outputs/{task}/{model_name}_analysis.txt', 'a') as f :
        f.write('\n'+'*'*100)
        f.write(record)
        f.write(str(ground_truth)+'\n')
        try:
            pred = match_list(task , answer, pre_len=pre_len)
        except:
            f.write(str('err happpend')+'\n')
            f.write('*'*100+'\n')
            return None 
        f.write(str(pred)+'\n')
        f.write('*'*100+'\n')
        return pred
        
def save_pred_label(preds , labels , config , opt):
    task = config.experiment.task 
    model_name = config.model.name
    npy_path = f'./outputs/{task}/npy'
    if not os.path.exists(npy_path):  
        os.makedirs(npy_path)
    np.save(f'{npy_path}/{model_name}_{opt}_out.npy' , np.array(preds) )
    np.save(f'{npy_path}/{model_name}_{opt}_lab.npy' , np.array(labels) )
    
def abstract_event(event):
    categories = {
        0: ['makes three point','makes 26-foot', 'makes 23-foot','makes 24-foot' , 'makes 27-foot'],
        1: [ 'makes 15-foot' , 'makes 1-foot', 'makes two point', 'makes layup', 'makes driving layup','makes 2-foot','makes 9-foot', 'makes 19-foot', 'makes 11-foot'],
        2: ['makes free throw' , 'makes technical free throw'],
        3: ['foul'],
        4: ['rebound'],
        5: ['misses', 'traveling', 'blocks' , 'charge'],
        6: ['steals','lost ball'],
        7: ['enters the game', 'timeout', 'turnover']
    }
    for category, keywords in categories.items():
        if 'makes' in event : 
            pattern = r'\b(\d+)-foot\b'
            matches = re.findall(pattern, event) 
            if len(matches) == 1 : 
                result = int(matches[0])
                if result<= 22 : return 1
                else : return 0
            if 'two point' in event or 'dunk' in event or '': return 1
            elif 'three point' in event : return 0
            elif 'free throw' in event : return 2
        if any(keyword in event for keyword in keywords):
            return category
    return 7

def get_metric(ground_truth , pred , task ):
    if pred == ground_truth : 
        return 1 
    else:
        return 0 



def get_metric_old(ground_truth , pred , task ):

    if task == 'middle_event':
        if len(ground_truth) > 1 :
            try : 
                pred = pred.split(",")
            except:
                return [0] * len(ground_truth)
        else :
            pred = [pred]
            
        count = []
        iter_ = min(len(ground_truth)  , len(pred))
        for i in range(iter_):
            if pred[i] == ground_truth[i]:
                count.append(1)
            else:
                count.append(0)
        return count

    if task == 'game_traceback':
        try : 
            pred = int(pred)
            if pred == ground_truth : 
                return 1 
            else:
                return 0 
        except:
            return 0

    if task == 'order_of_seq':
        try : 
            result = [int(x.strip()) for x in pred.split(',')]
            c = sum(el1 == el2 for el1, el2 in zip(result, ground_truth))
            return c / float(len(ground_truth))
        except:
            return 0    

    if task == 'reason-event':
        if not isinstance(pred , int ):
            pred = abstract_event(pred)
        ground_truth = abstract_event(ground_truth)
        if pred == ground_truth:
            return True
        else:
            return False

    if task == 'reason-actions':
        extra_up  =  ['FIX' , 'OR' , 'LOC' , 'ON' , 'IN' , 'AT' , "A" , '-' , 'LAB']
        extra_low = [e.lower() for e in extra_up]
        def extract_results(input_str):
            if not isinstance(input_str , list):
                parts = input_str.split(',')
            else:
                parts = input_str
                
            cleaned_parts = []
            for part in parts:
                action = part.strip().lower()
                if action not in extra_low:
                    cleaned_parts.append(action)
            return  cleaned_parts             
            
        ground_truth = extract_results(ground_truth)
        print('ground_truth' , ground_truth)
        ground_truth = set(ground_truth)
        
        pred = extract_results(pred)
        print('pred' , pred)
        pred = set(pred)

        return len(ground_truth & pred) > 0

    if task =='pick3imp':   
        corr = 0
        for i in range(3):
            for j in range(3) :
                if ground_truth[i] == pred[j] : corr +=1 
        return corr 
    ground_truth=np.array(ground_truth)
    pred=np.array(pred)
    assert ground_truth.shape == pred.shape 
    mae = np.mean(np.abs(ground_truth -pred))
    if task =='prdScore':
        corr = np.sum(np.subtract(ground_truth, pred) == 0)
        sum_ = len(ground_truth) * len(ground_truth[0])
        ratio_ = corr /sum_ 
        return mae , ratio_
    elif task =='forecast':
        return mae
        
        