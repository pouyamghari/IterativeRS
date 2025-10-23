import os
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import List, Union

class DNA4MerTokenizer(PreTrainedTokenizerBase):
    def __init__(self, tokenizer_path: str):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.vocab = self._tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"

        self.special_tokens = set([
            self.bos_token, self.eos_token,
            self.sep_token, self.unk_token
        ])

        self.bos_token_id = self.vocab.get(self.bos_token)
        self.eos_token_id = self.vocab.get(self.eos_token)
        self.sep_token_id = self.vocab.get(self.sep_token)
        self.unk_token_id = self.vocab.get(self.unk_token)

    def __len__(self):
        return self.vocab_size
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.vocab.get(self.unk_token))
        return [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.id_to_token.get(ids, self.unk_token)
        return [self.id_to_token.get(i, self.unk_token) for i in ids]
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def is_fast(self):
        return False

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_file = os.path.join(save_directory, "dnatokenizer.json")
        self._tokenizer.save(tokenizer_file)

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        tokenizer_file = os.path.join(save_directory, "dnatokenizer.json")
        return cls(tokenizer_file)

    def _pre_tokenize_dna(self, sequence: str):
        return [sequence[i:i+4] for i in range(0, len(sequence) - 3, 4)]

    def encode_plus(self, sequence: str, return_tensors=None, **kwargs):
        kmers = self._pre_tokenize_dna(sequence)
        encoded = self._tokenizer.encode(" ".join(kmers))
        result = {
            "input_ids": encoded.ids,
            "attention_mask": [1] * len(encoded.ids)
        }

        if return_tensors == "pt":
            result = {k: torch.tensor([v], dtype=torch.long) for k, v in result.items()}

        return BatchEncoding(result)

    def __call__(self, sequences: Union[str, List[str]], return_tensors=None, **kwargs):
        if isinstance(sequences, str):
            return self.encode_plus(sequences, return_tensors=return_tensors, **kwargs)
        all_encoded = [self.encode_plus(seq, return_tensors=None, **kwargs) for seq in sequences]

        if return_tensors == "pt":
            return [
                BatchEncoding({
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long)
                    }) for enc in all_encoded
                    ]

        return all_encoded

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = [self.id_to_token[i] for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens)