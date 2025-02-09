import torch 
import numpy as np 
from execute import benchmark_eval
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 

class Qwen(torch.nn.Module):
    
    def __init__(self, config):
        super(Qwen, self).__init__()
        if config.model.name == 'qwen_7B':
            self.model_path = ""
            self.save_file = 'q7.txt'
        if config.model.name == 'qwen_14B':
            self.model_path = ""
            self.save_file = 'q14.txt'  
        if config.model.name == 'qwen_32B':
            self.model_path = ""
            self.save_file = 'q32.json'
        #  python3 ./tsllm/main.py experiment=blank_rorder model=qwen_72B
        if config.model.name == 'qwen_72B':
            self.model_path = ""
            self.save_file = 'q72.json'
            
        self.fuc_llm = self.query_llm
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
    def run(self , config ):
        benchmark_eval(config, config.experiment.task , self.save_file , self.fuc_llm )
        
    def query_llm(self, question):
        chatgpt_sys_message = question['sys_prompt'] 
        forecast_prompt = question['input']
        
        messages = [
            {"role": "system", "content": chatgpt_sys_message},
            {"role": "user", "content": forecast_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print('0'*50)
        # print(chatgpt_sys_message+forecast_prompt)
        # print(response)
        # print('0'*50)
        return response 
        
    # def query_llm_72B(self, question):
    #     chatgpt_sys_message = question['sys_prompt'] 
    #     forecast_prompt = question['input']
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    #     messages = [
    #         {"role": "system", "content": chatgpt_sys_message},
    #         {"role": "user", "content": forecast_prompt},
    #     ]
    #     # format and tokenize the tool use prompt 
    #     inputs = tokenizer.apply_chat_template(
    #                 messages,
    #                 add_generation_prompt=True,
    #                 return_dict=True,
    #                 return_tensors="pt",
    #     )
    #     model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    #     inputs.to(model.device)
    #     outputs = model.generate(**inputs, max_new_tokens=1000)
    #     # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #     return outputs
