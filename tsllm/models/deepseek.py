import torch 
import transformers
from execute import benchmark_eval
import time 

class Deepseek(torch.nn.Module):
    
    def __init__(self, config):
        super(Deepseek, self).__init__()
        print('model name is', config.model.name )
        if  config.model.name == 'dsR1_70B':
            self.model_path = ""
            self.save_file = 'ds70.json'
        if config.model.name == 'dsR1_8B':
            self.model_path = ""
            self.save_file = 'ds8.json'
        if config.model.name == 'dsR1_32B':
            self.model_path = ""
            self.save_file = 'ds32.json'
            
        self.init_model(config)
        
    def run(self , config ):
        benchmark_eval(config, config.experiment.task , self.save_file , self.query_ds )

    def query_ds(self, question):
        beg = time.time()
        # ===============LLM 
        chatgpt_sys_message = question['sys_prompt'] 
        forecast_prompt = question['input']
        messages = [
            {"role": "system", "content": chatgpt_sys_message},
            {"role": "user", "content": forecast_prompt},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=2048*5,
        )
        answer = outputs[0]["generated_text"][-1]['content']
        # ===============LLM 
        print('cost' , time.time() - beg)
        # print(answer)
        # exit()
        return answer

    def init_model(self,config):
        # if '70B' not in config.model.name:
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
'''
 def query_l70(self, question):
    from transformers import AutoModelForCausalLM, AutoTokenizer , BitsAndBytesConfig
    beg = time.time()
    # ===============LLM 
    chatgpt_sys_message = question['sys_prompt'] 
    forecast_prompt = question['input']
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        self.model_path, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    
    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    input_text = chatgpt_sys_message+forecast_prompt
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = quantized_model.generate(**input_ids, max_new_tokens=2048)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    # ===============LLM 
    print('cost' , time.time() - beg)
    exit()
    return output
'''            
   