import hydra
from omegaconf import DictConfig
from tsllm.models.utils import load_model_by_name , get_predict_results 
from tsllm.datasets import get_datasets 
# from tsllm.pre_processing import pre_processing 
import os , pickle , time
from utils import build_save_path  , is_completion , setup_logging  
from prompt import build_prompts
from avai_prompt import build_avai_prompts
from ow_prompt import build_ow_prompts
        
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    logger = setup_logging()
    if 'build' in config.experiment.task:
        datasets = get_datasets(config , logger )
        if 'avai' in config.experiment.task:
            print('avai...')
            build_avai_prompts(config,datasets)
        elif 'openw' in config.experiment.task:
            print('openw...')
            build_ow_prompts(config,datasets)
        else:
            build_prompts(config,datasets) 
    else:
        model = load_model_by_name(config)
        model.run(config)
if __name__ == "__main__":
    run()