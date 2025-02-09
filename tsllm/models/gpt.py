import tiktoken
import torch 
import os 
import json
import time 
import re 
from execute import benchmark_eval
import openai
from openai import AzureOpenAI

class GPTmodel(torch.nn.Module):

    def __init__(self, config):
        super(GPTmodel, self).__init__()
            
        if  config.model.name == 'gpt4o':
            self.model = "gpt-4o-0513"
            self.save_file = 'gpt4o.txt'
            self.api_key = "Your Key"
            self.endpoint = ""
            self.api_version = "Your_API"
            
        if  config.model.name == 'gpt4o-mini':
            self.model = "gpt4o-mini"
            self.save_file = 'gpt4o-mini.txt'
            self.api_key = "Your Key"
            self.endpoint = ""
            self.api_version = "Your_API"
            
        print('Model is ' , self.model )

    def run(self , config ):
        benchmark_eval(config, config.experiment.task , self.save_file , self.query_llm )
        
    def query_llm(self, question):
        chatgpt_sys_message = question['sys_prompt'] 
        forecast_prompt = question['input']
        count = 0 
        while True:
            count +=1 
            try:
                # ===============LLM 
                client = AzureOpenAI(
                    api_version=self.api_version,
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                )
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "assistant",
                            "content":chatgpt_sys_message + forecast_prompt ,
                        },
                    ],
                )
                answer = completion.choices[0].message.content
                print(answer)
                return answer
                # ===============LLM 
            except:
                time.sleep(20)
                if count > 10 : 
                    print('api error')
                    exit()
