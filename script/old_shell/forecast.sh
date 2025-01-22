#!/bin/bash

# gpt4o
# python3 ./tsllm/main.py experiment=forecast model=gpt-4

# LLama2_7B
# python3 ./tsllm/main.py experiment=forecast model=llama2_7B

# llama3.1_8B
# python3 ./tsllm/main.py experiment=forecast model=llama3p1_8B


# Reason Events
# python3 ./tsllm/main.py experiment=reason-event model=llama3p1_8B
# python3 ./tsllm/main.py experiment=reason-actions model=llama3p1_8B
# python3 ./tsllm/main.py experiment=order_of_seq model=llama3p1_8B

python3 ./tsllm/main.py experiment=game_traceback model=llama3p1_8B

