import argparse
import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()

parser.add_argument('--init', type=str, default="None")

def save_first_time(tasks, init):
    if init=="SFT":
        model = AutoModelForCausalLM.from_pretrained("lib/Llama-3.2-3B-Instruct-SFT")
        model.save_pretrained("lib/fine_tuned_llama/model_iterativers_SFT")
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_auth_token=True)
        model.save_pretrained("lib/fine_tuned_llama/model_iterativers")
    for task in tasks:
        best_reward_head_path = f'lib/reward_heads/best_reward_head_{task}.pth'
        state_dict = torch.load(best_reward_head_path, map_location='cpu')

        for key in state_dict:
            if state_dict[key].dtype == torch.float32:
                state_dict[key] = state_dict[key].to(dtype=torch.bfloat16)

        value_head_path = f'lib/value_heads/value_head_{task}.pth'
        if init=="SFT":
            value_head_path = f'lib/value_heads/value_head_{task}_SFT.pth'
        torch.save(state_dict, value_head_path)


if __name__ == "__main__":
    args = parser.parse_args()
    tasks = ["faithful", "summary", "deberta"]
    save_first_time(tasks, args.init)