import random
import torch
import os

from sft_tools import * 
set_random_seed(42)
    
def main():
    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration.
    model_name = "/path/pretrain_model/Qwen2.5-1.5B-Instruct"
        
    # Load the pre-trained model on CPU first, then move to GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    ###########################
    # Step 1: SFT Fine-Tuning #
    ###########################
    print("\nPreparing SFT dataset...")
    # /path/data/nba/sft-3k2-corr.json
    sft_dataset = sports_sft_dataset(file_name='sft-3k2-corr.json')
    # /path/data/nfl/sft_5k_mix_nfl.json
    # /path/data/nfl/sft_2k_corr_nfl.json
        
    sft_training_args = TrainingArguments(
        output_dir="sft_output",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        # save_steps=30,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
        logging_steps=5
    )
    
    print("\nStarting SFT fine-tuning...")
    sft_trainer = Trainer(
        model=model,
        args=sft_training_args,
        train_dataset=sft_dataset,
        data_collator=ChatDataCollator(tokenizer)
    )
    sft_trainer.train()
    
    model.save_pretrained("SAVE-NAME")
    tokenizer.save_pretrained("SAVE-NAME")
    wandb.finish()

if __name__ == "__main__":
    main()
    
    