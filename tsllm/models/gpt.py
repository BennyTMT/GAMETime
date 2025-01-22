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
            self.api_key = "0626b133c7b5407d87aa8b93f3331031"
            self.endpoint = "https://rtp2-gpt35.openai.azure.com/"
            self.api_version = "2023-07-01-preview"
            
        if  config.model.name == 'gpt4o-mini':
            self.model = "gpt4o-mini"
            self.save_file = 'gpt4o-mini.txt'
            self.api_key ='0626b133c7b5407d87aa8b93f3331031'
            self.endpoint='https://rtp2-gpt35.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview'
            self.api_version = "2024-08-01-preview"
            
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
        
    def count_tokens(self, datas   , num_samples, temp):
        #  To prevent GPTs from producing unwanted tokens
        tokens = 0 
        sum_token = 0 
        for data in datas : 
            tokens = tokens + len(self.tokenize_fn(data[1], self.model_name))
            sum_token += tokens
            print(data[1] , tokens ,  sum_token )
        print(sum_token)
        exit()
        # avg_tokens_per_step = len(self.tokenize_fn(input_str, self.model_name)) / len(input_str.split(self.settings['time_sep']))
        # logit_bias = self.get_logit_bias()
        # chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     # model='gpt4-1106',
        #     deployment_id=DEPLOYMENT,
        #     messages=[
        #             {"role": "system", "content": chatgpt_sys_message},
        #             {"role": "user", "content": description }
        #         ],
        #     max_tokens=int(avg_tokens_per_step*steps), 
        #     temperature=temp,
        #     logit_bias=logit_bias,
        #     n=num_samples, # Generate num_samples different time series to help get an average.
        # )
        # return [choice.message.content for choice in response.choices]
    
    def get_logit_bias(self):
        # define logit bias to prevent GPT from producing unwanted tokens
        logit_bias = {}
        allowed_tokens = [self.settings['bit_sep'] + str(i) for i in range(self.settings['base'])] 
        allowed_tokens += [self.settings['time_sep'], self.settings['plus_sign'], self.settings['minus_sign']]
        allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
        if (self.model_name not in ['gpt-3.5-turbo','gpt-4',' "gpt-4-turbo"']): # logit bias not supported for chat models
            logit_bias = {id: 30 for id in self.get_allowed_ids(allowed_tokens, self.model_name)}
        return logit_bias
            
    def tokenize_fn(self, str, model):
        """
        This function is to help get the length of input 

        Args:
            str (list of str): str to be tokenized.
            model (str): Name of the LLM model.

        Returns:
            list of int: List of corresponding token IDs.
        """
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode(str)

    def get_allowed_ids(self, strs, model):
        """
        This function is help to limit the output tokens of GPT, to prevent it from
        generating data out of time series. 
        
        Args:
            strs (list of str): strs to be converted.
            model (str): Name of the LLM model.

        Returns:
            list of int: List of corresponding token IDs.
        """
        encoding = tiktoken.encoding_for_model(model)
        ids = []
        for s in strs:
            id = encoding.encode(s)
            ids.extend(id)
        return ids
    