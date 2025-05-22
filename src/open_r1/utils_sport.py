"""Reward functions for GRPO training."""
import json
import re
import random 

########### Reward Functions ########### 
def correctness_reward_sport(completions, solution, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list): List of input prompts.
        completions (list): List of model completions, each containing content.
        answer (list): List of expected answers.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of numerical rewards for each completion.

    Explanation:
        1. Extracts the content from each completion.
        2. Extracts the answer portion from each response using extract_answer_from_model_output.
        3. Assigns rewards based on matching criteria:
            - 2.0 points for an exact match
            - 1.5 points for numeric equivalence (when values match but format differs)
            - 0.0 points for incorrect answers
        4. Tracks completion lengths for analysis.
    """
    responses = [completion[0]['content'] for completion in completions]
    # The best situation : **a**, **b**, **c** , or **d**
    extracted = [extract_answer_from_model_sport(r) for r in responses]
    rewards = []
    # print(len(completions))
    # print(completions[0][0]['content'] , solution[0])
    for r, a in zip(extracted, solution):
        # print(r , a) ;  exit() 
        if r == a:  # Exact match case
            rewards.append(4.8)
        else:
            rewards.append(0.0)
            # r_alpha = extract_soft_answer(str(r))
            # if r_alpha is not None and r_alpha == a:
            #     rewards.append(2)
            # else:
            #     rewards.append(0.0)
        if r is not None :
            rewards[-1] = rewards[-1] + 0.8

    # record_training_process(prompts[0] , responses[0] + str(tks[0]) , extracted[0] , answer[0])
    # Log completion lengths
    #  completion_lengths = [len(response.split()) for response in responses]
    # tks = [ _get_num_tokens(r) for r in responses]
    # return rewards , tks
    return rewards 
    
def format_reward_sport(completions, **kwargs):
   """
   Assigns a reward for adhering to the desired XML format.

   Args:
       completions (list): List of model completions, each containing content.
       **kwargs: Additional keyword arguments.

   Returns:
       list: List of format compliance scores for each completion.

   Explanation:
       1. Extracts the content from each completion.
       2. Evaluates format compliance by checking for required XML tags:
          - 0.2 points for each tag present (<reasoning>, </reasoning>, <answer>, </answer>)
          - Maximum score of 0.8 for perfect format compliance
       3. Stores and returns the format compliance scores.
   """
   responses = [completion[0]['content'] for completion in completions]
   rewards = []
   format_scores = []
   for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if response.count("**") == 2 : 
            score += 0.4
        elif response.count("**") > 2: 
            score += 0.2
            
        rewards.append(score)
        format_scores.append(score)
   return rewards

########### Tools ########### 
def extract_answer_from_model_sport(text):
    """
    Extracts the value from the last <answer> tag in the text.

    Args:
        text (str): The model-generated text containing XML-style <answer> tags.

    Returns:
        str or None: The content inside the <answer> tags, or None if no valid answer is found.

    Explanation:
        1. Splits the text on the <answer> tag to isolate content after the tag.
        2. Checks if at least one <answer> tag exists in the text.
        3. For the last <answer> segment:
            - Verifies it contains a closing </answer> tag.
            - Extracts only the content between the tags.
        4. Returns None if the answer is empty (just "...") or if tags are missing.
    """
    # Only Select a b c d 
    pattern = r'\*\*(a|b|c|d)\*\*'
    matches = re.findall(pattern, text)
    if len(matches) >0 : 
        return matches[-1]
    else :
        return None
    
# PROCESS_RECORD = './file/Q1.5-sft-grpo.txt'
# def record_training_process(p , r , ext , label ):
#     with open(PROCESS_RECORD, "a", encoding="utf-8") as f:  # 使用 "a" 模式追加写入
#         f.write(f'Step:{STEP} -- Prompt: {p}\n') 
#         f.write(f"Compeletion ==>:**************\n{r}\n")
#         f.write(f"Extract ==>:**************{ext}\n")
#         f.write(f"Label   ==>:**************{label}\n")
#         f.write("*"*50 + '\n')

########### DataLoading Tool ########### 
SPORT_PROMPT = """<|im_start|>
Respond reasoning process in the following format:
<reasoning>
...
</reasoning>
Return your answer in **X**, where **X** is your answer and can only be one of the selected options, such as **a**, **b**, **c**, or **d**.
"""

def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])
    

def get_process_tokens(reason_proess , tokenizer ):
    encoding = tokenizer(
            reason_proess,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_length=True,
            add_special_tokens=False,
        )
    num_tokens = encoding["length"].item()  
    return num_tokens

import json 
import numpy as np 

def sports_grpo_dataset(data_path = '',  shuffle=True , tokenizer = None ):
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # 解析得到一个 list
    
    formatted_data = []
    nums = []
    for example in data_list:
        # Convert list of messages to a single string prompt.
        
        input_ = example['input'].split('Return your final answer in')[0]
        # prompt
        prompt_str =[
            {"role": "system", "content": SPORT_PROMPT},
            {"role": "user", "content": example["instruction"]+'\n'+input_ }
        ]
        formatted_example = {
            'question':example["instruction"]+'\n'+input_ ,
            "solution": example["output"], 
            "prompt": prompt_str, 
        }
        formatted_data.append(formatted_example)
        
        if len(formatted_data) == 1 : 
            print(SPORT_PROMPT +example["instruction"]+'\n'+input_ +  example["output"] )
        
        numt = get_process_tokens(SPORT_PROMPT +example["instruction"]+'\n'+input_ +  example["output"] , tokenizer )
        nums.append(numt)
        
    print(f'avg.{np.mean(nums)} ,  min:{ np.min(nums)} ,  max:{np.max(nums)}')
    if shuffle : 
        random.shuffle(formatted_data)
    
    return formatted_data