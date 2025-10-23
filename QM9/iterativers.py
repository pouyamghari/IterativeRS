import argparse
import random
import time
import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from lib.get_reward import get_reward_model
from lib.get_value import get_value_model
from trl import PPOConfig, PPOTrainer
from torch.optim import Adam

parser = argparse.ArgumentParser()

parser.add_argument('--gen_max_len', type=int, default=100)
parser.add_argument('--num_train_samples', type=int, default=256)
parser.add_argument('--num_eval_samples', type=int, default=64)

parser.add_argument('--num_epochs_per_step', type=int, default=2)
parser.add_argument('--per_device_train_batch_size', type=int, default=32)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--num_mini_batches', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1.41e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--kl_coef', type=float, default=0.05)

parser.add_argument('--num_tasks_per_epoch', type=int, default=3)
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--epoch', type=int, default=0)

parser.add_argument('--value_head_path', type=str, default="lib/value_heads/value_head_alpha.pth")
parser.add_argument('--model_weight_path', type=str, default="lib/fine_tuned_gpt2/model_weights_alpha.pth")
parser.add_argument('--model_merged_weight_path', type=str, default="lib/fine_tuned_gpt2/model_weights_merged.pth")

def save_first_time(tasks):
    for task in tasks:
        model = AutoModelForCausalLM.from_pretrained("lib/trained_gpt_moses")
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
        
        encoding = self.tokenizer(prompt, padding=False, truncation=True, return_tensors="pt")
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def PPOFineTuning(args, model, ref_model, task):
    tokenizer = SmilesTokenizer('lib/vocab_customized.txt')
    tokenizer.padding_side = "right"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
    tokenizer.pad_token = tokenizer.eos_token

    reward_model = get_reward_model(task)
    value_model = get_value_model(args, task)

    ppo_config = {"per_device_train_batch_size": args.per_device_train_batch_size,
                  "per_device_eval_batch_size": args.per_device_eval_batch_size,
                  "num_mini_batches": args.num_mini_batches,
                  "response_length": args.gen_max_len,
                  "temperature": args.temperature,
                  "num_train_epochs": args.num_epochs_per_step,
                  "kl_coef": args.kl_coef,
                  "bf16": True,
                  "fp16": False,
                  }
    config = PPOConfig(**ppo_config)

    train_prompts, eval_prompts = [""] * args.num_train_samples, [""] * args.num_eval_samples
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

def merge(tasks, w):
    merged_weights = torch.load("lib/fine_tuned_gpt2/model_weights_alpha.pth")

    weights = {}
    for task in tasks:
        model_weights = torch.load(f"lib/fine_tuned_gpt2/model_weights_{task}.pth")
        weights[task] = {k: v.cpu() for k, v in model_weights.items()}

    for key in merged_weights:
        combined_layer_weight = sum(w[task] * weights[task][key] for task in tasks)
        merged_weights[key] = combined_layer_weight
    
    return merged_weights

def main(args):
    random.seed(42)
    tasks = ['alpha', 'gap', 'u0']

    model = AutoModelForCausalLM.from_pretrained("lib/trained_gpt_moses")
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
            ref_model = AutoModelForCausalLM.from_pretrained("lib/trained_gpt_moses")
            PPOFineTuning(args, model, ref_model, tasks[i])
            end_time = time.time()
            elapsed_time = (end_time - start_time)/60
            print(f"Epoch {epoch+1} for task {tasks[i]} took {elapsed_time:.4f} minutes")
        merged_weights = merge(tasks, w)
        torch.save(merged_weights, args.model_merged_weight_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)