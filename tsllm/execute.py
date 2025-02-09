from utils import read_prompts , record_llm_process , get_metric, save_pred_label 
import os, json 
import numpy as np 
import random 
from datetime import datetime
import fcntl

SLOW = False
SAVENAME = 'areason'
SAVEPATH = f'/scratch/wtd3gz/project_TS/llm_game_ts/res_txt/{SAVENAME}/'
if not os.path.exists(SAVEPATH):  os.makedirs(SAVEPATH)
SLOWMODEL = ['llama3p1_70B' , 'qwen_72B' , 'qwen_32B' , 'mistral_7x8B']

TASKOPTS = {}
def init_options():
    global TASKOPTS 

    # Section-Avai 
    # items = ['base'  , '+entity', '+score' , '+timestamp',  '+event' , 'reorder' , '-timeseries']
    
    # Section-CoT 
    # items =['cot']
    # blank_rorder = []
    # for item in items:
    #     for sport in ['nba'  , 'nfl' ]:
    #         blank_rorder.append(f'{sport}_{item}')
    # TASKOPTS['blank_rorder'] = blank_rorder
    
    # Section-Num_Similarity
    # items = [1 , 5 , 10 , 15 , 20] 
    # blank_rorder = []
    # for item in items:
    #     for sport in [ 'nba' ]:
    #         blank_rorder.append(f'{sport}_{item}')
            
    # items = [1, 2 ,3, 4 , 5 ,  6 ,7]
    # for item in items:
    #     for sport in [ 'nba' ]:
    #         blank_rorder.append(f'{sport}_sim_{item}')
    # TASKOPTS['blank_rorder'] = blank_rorder
        
    # Section-Domis 
    # DS的output存在cot
    domains = ['crypto' , 'trading' , 'health_US'  , 'energy']
    blank_rorder = []
    for dom in domains:
        # for opt in [ 'w_num' , 'wo_num']:
        for opt in [ 'timeseries']:
            blank_rorder.append(f'{dom}_{opt}')
    TASKOPTS['blank_rorder'] = blank_rorder
    
    return TASKOPTS
    
init_options()

def benchmark_eval(config, task , save_file ,  query_llm):
    opts = TASKOPTS[task]
    global SLOW
    SLOW = True  if (config.model.name in SLOWMODEL) or ('json' in save_file) else False
    for opt in opts :
        if check_opt(save_file, opt) and (not SLOW): 
            continue
        data = read_prompts(config, f'{opt}.json')
        preds  = []
        labels = []
        fails  = 0
        ratios = []

        if SLOW : data = shuffle_prompts(data)
        
        for game in data: 
            if SLOW and check_slow(save_file , opt , game ):
                continue
            pred , ground_truth = run_save(data, game , config.experiment.task, opt , config.model.name , query_llm)
            if pred is None:
                fails+=1
                if SLOW: logger_slow_models_outs(save_file , pred, ground_truth , opt , game)
                else:  logger_reasoning_results(save_file , ratios , fails , opt )
                continue
            preds.append(pred) 
            labels.append(ground_truth)
            ratios.append(get_metric(ground_truth , pred , config.experiment.task))
            if SLOW: logger_slow_models_outs(save_file , pred, ground_truth , opt , game)
            else:  logger_reasoning_results(save_file , ratios , fails , opt )
            save_pred_label(preds , labels , config , opt)
            
def run_save(data , game , task , opt , model_name, query_llm):
    # ===============LLM 
    question = data[game] 
    answer = query_llm(question)
    ground_truth = question['label']
    sys_p = question['sys_prompt']
    inp   = question['input']
    record = f'\n{game}-{opt}\n{sys_p}{inp}====>  Response:\n{answer}\n\nLab vs Prd\n'
    pred = record_llm_process(record ,model_name , answer , ground_truth , task  , pre_len = 1  )
    # ===============LLM 
    return pred ,  data[game]['label'] 

def check_slow(save_file , opt , game ):
    file_path = SAVEPATH+save_file
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            data = json.load(file) 
            if f"{opt}_{game}" in data.keys(): 
                print(f"{opt}_{game} has done!")
                return True
            fcntl.flock(file, fcntl.LOCK_UN)
        return False
    else:
        return False

def logger_slow_models_outs(save_file , pred, ground_truth , opt , game):
    
    file_path = SAVEPATH+save_file
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump({}, file)
            
    with open(file_path, "r+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        data[f"{opt}_{game}"] = {
            'label':ground_truth, 
            'outs': pred
        }
        file.seek(0)
        file.truncate()
        json.dump(data, file, indent=4, ensure_ascii=False) 
        fcntl.flock(file, fcntl.LOCK_UN)
        
def logger_reasoning_results(save_file , ratios , fails , opt ):
    file_path = SAVEPATH+save_file
    acc = round((np.sum(ratios)/(len(ratios)+fails))*100,2)
    res_content = f'{opt} SuccessRate: {acc}%  {len(ratios)}  --- fails: {fails} \n'   
    with open(file_path, 'a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(res_content)
        if (len(ratios)+fails) % 50 == 0 : 
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M")
            file.write(f"Time Now: {formatted_time}\n")
        fcntl.flock(file, fcntl.LOCK_UN)
    
    
def check_opt(save_file , opt):
    if not os.path.exists(SAVEPATH+save_file):
        return False
    else:
        with open(SAVEPATH+save_file , 'r', encoding='utf-8') as file:
            for line in file:
                if opt in line: return True
        return False

def shuffle_prompts(data):
    keys = list(data.keys())
    random.shuffle(keys) 
    data = {key: data[key] for key in keys}
    return data