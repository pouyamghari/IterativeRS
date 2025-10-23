import argparse
import random
import time
import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset
import torch.nn as nn
from trl import PPOConfig, PPOTrainer
from torch.optim import Adam
from lib.DNATokenizers import DNA4MerTokenizer
from lib.reward_model import get_reward_model
from lib.value_model import get_value_model
from lib.merge_iterativers import merge, merge_pick_weights

parser = argparse.ArgumentParser()

parser.add_argument('--gen_max_len', type=int, default=52)
parser.add_argument('--num_train_samples', type=int, default=256)
parser.add_argument('--num_eval_samples', type=int, default=64)

parser.add_argument('--num_epochs_per_step', type=int, default=4)
parser.add_argument('--per_device_train_batch_size', type=int, default=32)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--num_mini_batches', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1.41e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--temperature', type=float, default=0.7)

parser.add_argument('--num_tasks_per_epoch', type=int, default=3)
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--epoch', type=int, default=0)

parser.add_argument('--value_head_path', type=str, default="lib/value_heads/value_head_K562.pth")
parser.add_argument('--model_weight_path', type=str, default="lib/fine_tuned_gpt2/model_weights_K562.pth")
parser.add_argument('--model_save_weight_path', type=str, default="model_weights.pth")
parser.add_argument('--merge_strategy', type=str, default="uniform")

def save_first_time(tasks):
    for task in tasks:
        model = AutoModelForCausalLM.from_pretrained("dna_gpt2")
        torch.save(model.state_dict(), f"lib/fine_tuned_gpt2/model_weights_{task}.pth")

        best_reward_head_path = f'lib/reward_heads/best_reward_head_{task}.pth'
        state_dict = torch.load(best_reward_head_path)
        value_head_path = f'lib/value_heads/value_head_{task}.pth'
        torch.save(state_dict, value_head_path)

class PromptDataset(Dataset):
    def __init__(self, tokenizer, prompts):
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        
        encoding = self.tokenizer(prompt, return_tensors="pt")
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def PPOFineTuning(args, model, ref_model, task):
    tokenizer = DNA4MerTokenizer("lib/dna_4mer_tokenizer.json")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    reward_model = get_reward_model(task)
    value_model = get_value_model(args, task)

    ppo_config = {"per_device_train_batch_size": args.per_device_train_batch_size,
                  "per_device_eval_batch_size": args.per_device_eval_batch_size,
                  "num_mini_batches": args.num_mini_batches,
                  "response_length": args.gen_max_len,
                  "temperature": args.temperature,
                  "num_train_epochs": args.num_epochs_per_step,
                  "bf16": True,
                  "fp16": False,
                  }
    config = PPOConfig(**ppo_config)

    train_prompts, eval_prompts = [tokenizer.bos_token] * args.num_train_samples, [tokenizer.bos_token] * args.num_eval_samples
    train_dataset = PromptDataset(tokenizer=tokenizer, prompts=train_prompts)
    eval_dataset = PromptDataset(tokenizer=tokenizer, prompts=eval_prompts)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    ppo_trainer = PPOTrainer(args = config,
        processing_class = tokenizer,
        model = model, 
        ref_model = ref_model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        reward_model = reward_model,
        value_model = value_model,
        optimizers=(optimizer, None)
        )
    
    ppo_trainer.train()

    torch.save(ppo_trainer.value_model.mlp.state_dict(), args.value_head_path)

    torch.save(model.state_dict(), args.model_weight_path)

def main(args):
    random.seed(42)
    tasks = ['K562', 'HepG2', 'SKNSH']
    model = AutoModelForCausalLM.from_pretrained("dna_gpt2")
    merged_weights = model.state_dict()
    save_first_time(tasks)
    w = {}
    for task in tasks:
        w[task] = 1/len(tasks)

    num_epochs = args.num_epochs
    num_tasks_per_epochs = args.num_tasks_per_epoch

    for epoch in range(num_epochs):
        task_indices = random.sample(range(len(tasks)), num_tasks_per_epochs)
        args.epoch = epoch+1
        for i in task_indices:
            start_time = time.time()
            args.value_head_path = f"lib/value_heads/value_head_{tasks[i]}.pth"
            args.model_weight_path = f"lib/fine_tuned_gpt2/model_weights_{tasks[i]}.pth"
            model.load_state_dict(merged_weights)
            ref_model = AutoModelForCausalLM.from_pretrained("dna_gpt2")
            PPOFineTuning(args, model, ref_model, tasks[i])
            end_time = time.time()
            elapsed_time = (end_time - start_time)/60
            print(f"Epoch {epoch+1} for task {tasks[i]} took {elapsed_time:.4f} minutes")
        if args.merge_strategy=="selection":
            set_weights = [[1/3, 1/3, 1/3],
                           [1/6, 5/12, 5/12],
                           [5/12, 1/6, 5/12],
                           [5/12, 5/12, 1/6],
                           [1/6, 1/6, 2/3],
                           [1/6, 2/3, 1/6],
                           [2/3, 1/6, 1/6],
                           [1/2, 1/4, 1/4],
                           [1/4, 1/2, 1/4],
                           [1/4, 1/4, 1/2]
                           ]
            w = merge_pick_weights(tasks, set_weights, args.temperature, args.gen_max_len)
        merged_weights = merge(tasks, w)
        torch.save(merged_weights, args.model_save_weight_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)