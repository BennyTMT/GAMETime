# reason_events

## How to set up LLMs
The setup of different LLMs is in ./tsllm/models/

## How to run the repo 
./batch.sh  and ./slow.sh  is built to submit task in ./script 

## How do I use multi-gpus to run llama3.1 70B
It is very simple, I firstly build my prompts and they will be save in "prompts" 

and models running in different GPUs will go and draw task from "prompts".  details shown in "./tsllm/execute.py"

## How to setup prompts
./script/build.sh

# To myself 
Remove OpenAI key when publish this repo, and check hisotry branch. 
