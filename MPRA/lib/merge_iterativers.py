import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from DNATokenizers import DNA4MerTokenizer
from reward_model import get_reward_model

def merge(tasks, w):
    merged_weights = torch.load("fine_tuned_gpt2/model_weights_K562.pth")

    weights = {}
    for task in tasks:
        model_weights = torch.load(f"fine_tuned_gpt2/model_weights_{task}.pth")
        weights[task] = {k: v.cpu() for k, v in model_weights.items()}

    for key in merged_weights:
        combined_layer_weight = sum(w[task] * weights[task][key] for task in tasks)
        merged_weights[key] = combined_layer_weight
    
    return merged_weights

def merge_pick_weights(tasks, set_weights, temp, gen_max_len):
    curr_reward = 0
    for curr in set_weights:
        w = {task: curr[i] for i, task in enumerate(tasks)}
        merged_weights = merge(tasks, w)

        val_reward = compute_val_reward(tasks, merged_weights, temp, gen_max_len, 32, 1024, -1)
        if val_reward>curr_reward:
            best_weights = w
            curr_reward = val_reward
    return best_weights

def is_pareto_efficient_max(rewards, return_mask=True):
    is_efficient = np.ones(rewards.shape[0], dtype=bool)
    for i, c in enumerate(rewards):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(rewards[is_efficient] > c, axis=1) |
                np.all(rewards[is_efficient] == c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient if return_mask else np.where(is_efficient)[0]

def compute_val_reward(tasks, merged_weights, temp, gen_max_len, batch_size=16, val_size=256, top_k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = DNA4MerTokenizer("dna_4mer_tokenizer.json")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    val_rewards = {task: [] for task in tasks}
    reward_models = {task: get_reward_model(task).to(device) for task in tasks}

    model = AutoModelForCausalLM.from_pretrained("dna_gpt2").to(device)
    model.load_state_dict(merged_weights)

    model.eval()
    
    for i in tqdm(range(0, val_size, batch_size)):
        prompt = [""] * batch_size
        encoded_list = tokenizer(prompt, return_tensors=None)
        input_ids = torch.stack([torch.tensor(enc["input_ids"], dtype=torch.long) for enc in encoded_list]).to(device)
        context_length = input_ids.shape[1]

        with torch.no_grad():
            query_responses = model.generate(input_ids=input_ids, 
                                             temperature=temp, 
                                             do_sample=True,
                                             top_p = 0.9,
                                             max_new_tokens=gen_max_len
                                             )
        
        for task in tasks:
            reward_logits = reward_models[task](query_responses[:,context_length:], tokenizer.pad_token_id)
            reward_scores = torch.sigmoid(reward_logits)
            val_rewards[task].extend(reward_scores.view(-1).tolist())
    
    rewards_matrix = np.array([[val_rewards[task][i] for task in tasks] for i in range(len(val_rewards[tasks[0]]))])
    pareto_mask = is_pareto_efficient_max(rewards_matrix, return_mask=True)
    pareto_indices = np.where(pareto_mask)[0]
    pareto_avg_reward = rewards_matrix[pareto_indices].mean()

    return pareto_avg_reward