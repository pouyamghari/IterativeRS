import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from lib.DNATokenizers import DNA4MerTokenizer
from sklearn.model_selection import train_test_split
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

parser = argparse.ArgumentParser()

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--per_device_train_batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1.41e-4)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_positions', type=int, default=128)
parser.add_argument('--model_path', type=str, default="dna_gpt2")

class DNADataset(Dataset):
    def __init__(self, tokenizer, dnaseqs):
        self.tokenizer = tokenizer
        self.dnaseqs = dnaseqs

    def __len__(self):
        return len(self.dnaseqs)

    def __getitem__(self, idx):
        dnaseq = self.dnaseqs[idx]
        
        encoding = self.tokenizer(dnaseq, return_tensors="pt")
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        input_ids = torch.cat([
            torch.tensor([self.tokenizer.bos_token_id]), 
            input_ids, 
            torch.tensor([self.tokenizer.eos_token_id])
            ])

        attention_mask = torch.cat([
            torch.tensor([1]), 
            attention_mask, 
            torch.tensor([1])
            ])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

def main(args):
    mpra_19 = pd.read_table('Table_S2__MPRA_dataset.txt', sep='\t', header=0)
    mpra_19 = mpra_19[mpra_19['sequence'].str.len() == 200]

    dna_seqs = mpra_19['sequence'].to_list()

    train_seqs, eval_seqs = train_test_split(dna_seqs, test_size=0.1, random_state=42)

    tokenizer = DNA4MerTokenizer("lib/dna_4mer_tokenizer.json")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = DNADataset(tokenizer=tokenizer, dnaseqs=train_seqs)
    eval_dataset = DNADataset(tokenizer=tokenizer, dnaseqs=eval_seqs)

    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=args.num_positions,
        n_ctx=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    model = GPT2LMHeadModel(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./lib/gpt2-dna",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        logging_dir="./lib/logs",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=1000,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(args.model_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)