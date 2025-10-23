import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from Iterativers import get_reward_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--with_sft', type=int, default=0)
parser.add_argument('--merge_strategy', type=str, default='selection')
parser.add_argument('--split', type=str, default='train')

def merge(args, tasks, w):
    if args.with_sft==1:
        model = AutoModelForCausalLM.from_pretrained("lib/fine_tuned_llama/model_iterativers_SFT").to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("lib/fine_tuned_llama/model_iterativers").to(device)
    merged_weights = model.state_dict()

    weights = {}
    for task in tasks:
        if args.with_sft==1:
            model_task = AutoModelForCausalLM.from_pretrained(f"lib/fine_tuned_llama/Llama-3.2-3B-Instruct_{task}_SFT")
        else:
            model_task = AutoModelForCausalLM.from_pretrained(f"lib/fine_tuned_llama/Llama-3.2-3B-Instruct_{task}")
        task_weights = model_task.state_dict()
        weights[task] = {k: v.cpu() for k, v in task_weights.items()}

    for key in merged_weights:
        combined_layer_weight = sum(w[task] * weights[task][key] for task in tasks)
        merged_weights[key] = combined_layer_weight
    
    model.load_state_dict(merged_weights)

    return model

def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def compute_val_reward(args, tasks, model, batch_size=16, val_size=256):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    val_rewards = {task: [] for task in tasks}
    reward_models = {task: get_reward_model(task).to(device) for task in tasks}

    if args.split=="train":
        df_val = pd.read_csv('lib/data_splits/train_samples.csv')
    else:
        df_val = pd.read_csv('lib/data_splits/validation_samples.csv')
    if args.with_sft==1:
        df_val['prompt'] = df_val['prompt'].apply(lambda x: f"{x.strip()}\n\n### Response:\n")
    eval_prompts = df_val['prompt'].tolist()
    eval_prompts = eval_prompts[:val_size]

    model.eval()
    
    for i in tqdm(range(0, len(eval_prompts), batch_size)):
        batch_prompts = eval_prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        context_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            query_responses = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=32,
                do_sample=True,
                top_p=0.9,
                temperature=1,
            )
            attention_mask = query_responses != tokenizer.pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long()
            input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True
                )
        
        for task in tasks:
            reward_logits = reward_models[task].score(output.hidden_states[-1])
            sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
            reward_scores = reward_logits[
                torch.arange(reward_logits.size(0), device=reward_logits.device),
                sequence_lengths,
                ].squeeze(-1)
            val_rewards[task].extend(reward_scores.view(-1).tolist())
    
    num_items  = len(val_rewards["faithful"])
    avg_rewards = [
        sum(val_rewards[task][i] for task in tasks) / len(tasks) for i in range(num_items)
        ]

    return sum(avg_rewards)/num_items

def merge_pick_weights(args, tasks, set_weights, save_path):
    curr_reward = 0
    for curr in set_weights:
        w = {task: curr[i] for i, task in enumerate(tasks)}
        model = merge(args, tasks, w)

        if args.split=="train":
            val_size = 256
        else:
            val_size = 1024
        val_reward = compute_val_reward(args, tasks, model, 16, val_size)
        if val_reward>curr_reward:
            model.save_pretrained(save_path)
            curr_reward = val_reward

if __name__ == "__main__":
    args = parser.parse_args()
    if args.with_sft==1:
        save_path = "lib/fine_tuned_llama/model_iterativers_SFT"
    else:
        save_path = "lib/fine_tuned_llama/model_iterativers"
    tasks = ["faithful", "summary", "deberta"]
    set_weights = [[1/3, 1/3, 1/3],
                   [1/6, 5/12, 5/12],
                   [5/12, 1/6, 5/12],
                   [5/12, 5/12, 1/6],
                   [1/6, 1/6, 2/3],
                   [1/6, 2/3, 1/6],
                   [2/3, 1/6, 1/6]
                   ]
    if args.merge_strategy=="selection":
        w = merge_pick_weights(args, tasks, set_weights, save_path)
    else:
        w = {task: 1/len(tasks) for task in tasks}
        model = merge(args, tasks, w)
        model.save_pretrained(save_path)