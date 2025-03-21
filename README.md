# Aren't You Curious about the Events behind real-valued Time Series?
![Events Reasoning](./script/pipline.png)

### "Inferring Event Descriptions from Time Series with Language Models" [Paper Link](https://arxiv.org/pdf/2503.14190).
## Dataset📖
We curate a dataset and propose our benchmark **"GAMETime: Generating And Modeling Events from TIME series"**. This dataset contains a **real-valued time series** of **1.7 million timestamps** along with corresponding **event sequences**.

## Setup 🔧
For our environment configuration and required additional packages, please refer to **"environment.yml"**.

## Run Experiments 💁🏼
./batch.sh (or ./slow.sh) is the shell built to submit tasks in **./script/**

Our main configuration files about LLMs and experiments are located in **"./tsllm/config/"**.

## How to setup prompts
run **./script/build.sh** and change the conditions in the shell script for different experiments 

And the prompts will be/have been saved in **"./prompt/"**

## How to set up LLMs
The setup of different LLMs is in **./tsllm/models/**


## Tips: How to use multi-gpus to run 70B LMs
It is quit straightforward, we firstly build prompts and save them in "prompts" directory. 

Then, models running in different GPUs will go and draw task from "prompts". (Details are shown in **"./tsllm/execute.py"**)

## If you're interested in our sports data 🛎️
Please email the author directly. To simplify the process, just send the following:
```bash
  "Hello, GAMETime."
  [Your Name]
```

We will provide you with a download link, **aiming to complete the process within a minute.**

Or you can find the script in the resource file to download the HTML and extract the data yourself.

<img src="./resource/event.png" alt="Game Events" width="500"/>

