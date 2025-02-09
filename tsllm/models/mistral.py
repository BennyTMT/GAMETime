import torch 
import os, json 
import re 
import numpy as np 
from execute import benchmark_eval
import time 
import random 
from datetime import datetime
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

class Mistral(torch.nn.Module):
    
    def __init__(self, config):
        super(Mistral, self).__init__()
        # python3 ./tsllm/main.py experiment=blank_rorder model=mistral_8B
        if config.model.name == 'mistral_8B':
            self.model_path = ""
            self.save_file = 'm8.txt'
            self.load_vllm_model(1)
        # python3 ./tsllm/main.py experiment=blank_rorder model=mistral_7x8B
        if config.model.name == 'mistral_7x8B':
            self.model_path = ""
            self.save_file = 'm7x8.json'
            self.load_vllm_model(4)
        # python3 ./tsllm/main.py experiment=blank_rorder model=mistral_small
        if config.model.name == 'mistral_small':
            self.model_path = ""
            self.save_file = 'msml.txt'
            self.load_vllm_model(2)
            
    def run(self , config ):
        benchmark_eval(config, config.experiment.task , self.save_file , self.query_llm)

    def query_llm(self, question):
        chatgpt_sys_message = question['sys_prompt'] 
        forecast_prompt = question['input']
        messages = [
            {"role": "system", "content": chatgpt_sys_message},
            {"role": "user", "content": forecast_prompt}
        ]
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        answer = outputs[0].outputs[0].text
        # ===============LLM 
        return answer
        
    def load_vllm_model(self , num):
        if '8B' in self.model_path:
            # ===============LLM 
            self.sampling_params = SamplingParams(max_tokens=1024)
            self.llm = LLM(model=self.model_path, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
        else:
            # ===============LLM 
            self.sampling_params = SamplingParams(max_tokens=1024)
            self.llm = LLM(self.model_path , tensor_parallel_size=num)
