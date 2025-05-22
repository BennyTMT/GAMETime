############# Part 1: Basic Setup and Imports ############# 
import random
import torch
import numpy as np
def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
import hashlib
import tarfile
import requests
import re
import wandb
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

############# Part 2: Data Formatting and Answer Extraction ############# 
SPORT_PROMPT = """<|im_start|>
Respond reasoning process in the following format:
<reasoning>
...
</reasoning>
Return your answer in **X**, where **X** is your answer and can only be one of the selected options, such as **a**, **b**, **c**, or **d**.
"""
############# Part 3: Dataset Preparation ############# 
import json 
PRESENT_INPUT = True 
def sports_sft_dataset(file_name='.json'):
    global PRESENT_INPUT
    data_path = f'/path/data/nba/{file_name}'
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # 解析得到一个 list

    formatted_data = []
    for example in data_list:
        # Convert list of messages to a single string prompt.
        if example["process"].count("<think>") != 1 or example["process"].count("</think>") != 1 : 
            continue
                    
        # 手动添加label，确保有**X**
        process_answer = example["process"].split('</think>')[0] + '</reasoning>' +\
                f'\n**{example["label"]}**'
        process_answer = process_answer.replace('<think>' , '<reasoning>')
        
        # Q&A
        # process_answer = f'<reasoning>\n...\n</reasoning>\n**{example["label"]}**'
        
        example['input'] = example['input'].split('Return your final answer in')[0]
        
        if PRESENT_INPUT : 
            print('-'*100)
            print(SPORT_PROMPT)
            print(example["instruction"])
            print(example["input"])
            print(process_answer)
            print('-'*100)
            PRESENT_INPUT = False
            # exit()

        formatted_example = {
                "messages": [
                    {"role": "system", "content": SPORT_PROMPT},
                    {"role": "user", "content": example["instruction"]+'\n'+example['input']},
                    {"role": "assistant", "content": process_answer}
                ]
            }
        formatted_data.append(formatted_example)
    print(f'Finial Dataset length is {len(formatted_data)}')
    return formatted_data

def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    return "\n".join([msg["content"].strip() for msg in messages])

class ChatDataCollator:
    '''
        The ChatDataCollator
        is used in supervised fine-tuning (SFT) to concatenate chat messages 
        into a fixed, padded sequence suitable for training.
    '''
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 1024*4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.present_data = True 
    def __call__(self, batch):
        inputs = []
        labels = []
        for example in batch:
            # Here we assume the last message is the target (assistant's output)
            prompt = build_prompt(example["messages"][:-1])
            target = example["messages"][-1]["content"]
            
            # Concatenate prompt and target (add a newline between them)
            full_text = prompt + "\n" + target +'<|im_end|>'
            tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length)
            
            if self.present_data : 
                print('*'*100)
                print(full_text)
                print(len(tokenized["input_ids"]))
                print('*'*100)
                # exit()
                self.present_data = False 
                
            input_ids = torch.tensor(tokenized["input_ids"])
            inputs.append(input_ids)
            # You can choose to set labels equal to input_ids, or modify as needed.
            labels.append(input_ids)

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": inputs_padded, "labels": labels_padded}
    
    
def prepare_sft_dataset(num_examples=500):
    """
    Prepare SFT examples in the chat format required by your custom collator.
    Each example will be a dict with a "messages" key.
    """
    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    cot_url = "https://github.com/aburkov/theLMbook/releases/download/v1.0.0/cot.tar.gz"
    extract_dir = download_and_extract_cot_archive(cot_url, extract_path="cot_archive")
    data = load_dataset('openai/gsm8k', 'main')["train"]
    
    sft_examples = []
    for example in data:
        question = example["question"].strip()
        # Compute the filename based on the SHA-256 hash of the question.
        filename = hashlib.sha256(question.encode()).hexdigest() + ".txt"
        file_path = os.path.join(extract_dir, filename)

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                cot_output = f.read().strip()

            # Build the chat-format example.
            formatted_example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": cot_output}
                ]
            }
            sft_examples.append(formatted_example)

        if len(sft_examples) >= num_examples:
            break

    if len(sft_examples) < num_examples:
        print(f"Warning: Only found {len(sft_examples)} SFT examples.")
    else:
        print(f"Prepared {len(sft_examples)} SFT examples.")

    return sft_examples

    