import argparse
import time
import random
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset
from trl import PPOConfig, PPOTrainer
from trl.models.utils import unwrap_model_for_generation

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default="faithful")
parser.add_argument('--with_sft', type=int, default=1)
parser.add_argument('--num_epochs_per_step', type=int, default=2)
parser.add_argument('--per_device_train_batch_size', type=int, default=8)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--num_mini_batches', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--num_prompts_per_epoch', type=int, default=1024)
parser.add_argument('--gen_max_len', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1.41e-6)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--kl_coef', type=float, default=0.05)
parser.add_argument('--num_tasks_per_epoch', type=int, default=3)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--value_head_path', type=str, default="lib/value_heads/value_head_faithful.pth")
parser.add_argument('--model_path', type=str, default="lib/fine_tuned_llama/Llama-3.2-3B-Instruct_faithful")

class get_reward_model(nn.Module):
    def __init__(self, task):
        super(get_reward_model, self).__init__()
        self.task = task
        self.base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_auth_token=True)
        self.base_model_prefix = "base_model"
        hidden_size = self.base_model.config.hidden_size
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        best_reward_head_path = f'lib/reward_heads/best_reward_head_{task}.pth'
        state_dict = torch.load(best_reward_head_path, map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.mlp = self.mlp.to(self.base_model.dtype)

    def forward(self, query_responses, pad_token_id, **kwargs):
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )

        reward_logits = self.mlp(outputs.hidden_states[-1])
        sequence_lengths = (query_responses != pad_token_id).sum(dim=1) - 1

        return reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
            ].squeeze(-1)

    def score(self, hidden_states):
        target_dtype = self.mlp[0].weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
        logits = self.mlp(hidden_states)
        reward_logits = torch.clamp(logits, 0, 2)

        return reward_logits

class get_value_model(nn.Module):
    def __init__(self, value_head_path):
        super(get_value_model, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_auth_token=True)
        self.base_model_prefix = "base_model"
        hidden_size = self.base_model.config.hidden_size
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
        state_dict = torch.load(value_head_path, map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.mlp = self.mlp.to(self.base_model.dtype)

    def forward(self, query_responses, pad_token_id, **kwargs):
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )

        reward_logits = self.mlp(outputs.hidden_states[-1])
        sequence_lengths = (query_responses != pad_token_id).sum(dim=1) - 1

        return reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
            ].squeeze(-1)

    def score(self, hidden_states):
        logits = self.mlp(hidden_states)
        logits = torch.clamp(logits, 0, 2)

        return logits
    
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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    reward_model = get_reward_model(task)
    value_model = get_value_model(args.value_head_path)

    ppo_config = {"per_device_train_batch_size": args.per_device_train_batch_size,
                  "per_device_eval_batch_size": args.per_device_eval_batch_size,
                  "gradient_accumulation_steps": args.gradient_accumulation_steps,
                  "temperature": args.temperature,
                  "num_mini_batches": args.num_mini_batches,
                  "num_train_epochs": 1,
                  "response_length": args.gen_max_len,
                  "output_dir": "ppo_outputs/Iterativers",
                  "kl_coef": args.kl_coef
                  }
    config = PPOConfig(**ppo_config)

    df_train = pd.read_csv('data_splits/train_samples.csv')
    df_eval = pd.read_csv('data_splits/validation_samples.csv')
    if args.with_sft==1:
        df_train['prompt'] = df_train['prompt'].apply(lambda x: f"{x.strip()}\n\n### Response:\n")
        df_eval['prompt'] = df_eval['prompt'].apply(lambda x: f"{x.strip()}\n\n### Response:\n")
    sampled_df_train = df_train.sample(n=args.num_prompts_per_epoch, random_state=int(time.time()))
    train_prompts = sampled_df_train['prompt'].tolist()
    eval_prompts = df_eval['prompt'].to_list()
    train_dataset = PromptDataset(tokenizer=tokenizer, prompts=train_prompts)
    eval_dataset = PromptDataset(tokenizer=tokenizer, prompts=eval_prompts)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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

    for ep in range(1, args.num_epochs_per_step):
        sampled_df_train = df_train.sample(n=args.num_prompts_per_epoch, random_state=ep)
        train_prompts = sampled_df_train['prompt'].tolist()
        train_dataset = PromptDataset(tokenizer=tokenizer, prompts=train_prompts)
        ppo_trainer.train_dataset = train_dataset
        ppo_trainer.train()

    with unwrap_model_for_generation(
        ppo_trainer.model, ppo_trainer.accelerator, gather_deepspeed3_params=ppo_trainer.args.ds3_gather_for_generation
        ) as unwrapped_model:
        unwrapped_model.policy.save_pretrained(args.model_path)
        state_dict = unwrapped_model.value_model.mlp.state_dict()
        torch.save(state_dict, args.value_head_path)

def main(args):
    start_time = time.time()
    if args.with_sft==1:
        model = AutoModelForCausalLM.from_pretrained("lib/fine_tuned_llama/model_iterativers_SFT")
        args.value_head_path = f"lib/value_heads/value_head_{args.task}_SFT.pth"
        args.model_path = f"lib/fine_tuned_llama/Llama-3.2-3B-Instruct_{args.task}_SFT"
    else:
        model = AutoModelForCausalLM.from_pretrained("lib/fine_tuned_llama/model_iterativers")
        args.value_head_path = f"lib/value_heads/value_head_{args.task}.pth"
        args.model_path = f"lib/fine_tuned_llama/Llama-3.2-3B-Instruct_{args.task}"
    ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_auth_token=True)
    PPOFineTuning(args, model, ref_model, args.task)
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"Epoch {args.epoch} for task {args.task} took {elapsed_time:.4f} minutes")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)