from models.gpt import GPTmodel
from models.llama import LLama 
from models.mistral import Mistral 
from models.qwen import Qwen 
from models.phi import PHI

from tqdm import tqdm

import numpy as np 
import pandas as pd

import os , sys 
    
'''
    Note that : 
    This script references https://github.com/ngruver/llmtime and https://arxiv.org/pdf/2310.07820.pdf. Thank you for your work.
'''
STEP_MULTIPLIER = 1.2 
    
def load_model_by_name(config):
    if 'gpt' in config.model.name :
        return GPTmodel(config=config) 
    if 'llama' in  config.model.name :
        return LLama(config=config)
    if 'mistral' in  config.model.name :
        return Mistral(config=config)
    if 'qwen' in  config.model.name :
        return Qwen(config=config)
    if 'phi' in  config.model.name :
        return PHI(config=config)
        
def get_output_format(preds , test , results_list , model_name , input_strs ):
    
    samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
    medians = [sample.median(axis=0) for sample in samples]
    samples = samples if len(samples) > 1 else samples[0]
    medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model_name,
        },
        'completions_list': results_list,
        'input_strs': input_strs,
    }
    return out_dict

def get_predict_results(model ,data, config, batch_size, dsname ):
    results_list = []
    model.run(data , 1 , batch_size ,dsname , config ) 
    return 0 
    # print(res)
    # results_list.append(res)
    # exit()
    
def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]



