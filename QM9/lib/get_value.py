import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class get_value_model(nn.Module):
    def __init__(self, args, task):
        super(get_value_model, self).__init__()
        self.task = task
        self.base_model = AutoModelForCausalLM.from_pretrained("lib/trained_gpt_moses")
        self.base_model_prefix = "base_model"
        hidden_size = self.base_model.config.hidden_size
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.mlp.load_state_dict(torch.load(args.value_head_path))

    def score(self, hidden_states):
        logits = self.mlp(hidden_states)

        return logits