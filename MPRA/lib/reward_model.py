import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class get_reward_model(nn.Module):
    def __init__(self, task):
        super(get_reward_model, self).__init__()
        self.task = task
        self.base_model = AutoModelForCausalLM.from_pretrained("dna_gpt2")
        self.base_model_prefix = "base_model"
        hidden_size = self.base_model.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        best_reward_head_path = f'reward_heads/best_reward_head_{task}.pth'
        self.mlp.load_state_dict(torch.load(best_reward_head_path))


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

        reward_logits = torch.sigmoid(logits)

        return reward_logits