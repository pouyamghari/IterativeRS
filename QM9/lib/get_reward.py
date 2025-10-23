import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class get_reward_model(nn.Module):
    def __init__(self, task):
        super(get_reward_model, self).__init__()
        self.task = task
        self.base_model = AutoModelForCausalLM.from_pretrained("lib/trained_gpt_moses")
        self.base_model_prefix = "base_model"
        hidden_size = self.base_model.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        best_reward_head_path = f'reward_heads/best_reward_head_{task}.pth'
        self.mlp.load_state_dict(torch.load(best_reward_head_path))

    def first_true_indices(self, bools: torch.Tensor, dtype=torch.long):
        row_len = bools.size(-1)
        zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
        return torch.min(zero_or_index, dim=-1).values

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
        sequence_lengths = self.first_true_indices(query_responses == pad_token_id) - 1

        return reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
            ].squeeze(-1)

    def score(self, hidden_states):
        logits = self.mlp(hidden_states)

        if self.task == 'alpha' or self.task == 'u0':
            reward_logits = logits
        elif self.task == 'gap':
            reward_logits = 1 - torch.abs(logits - 0.5)

        reward_logits = torch.clamp(reward_logits, 0, 2)

        return reward_logits