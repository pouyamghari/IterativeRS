import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from functools import partial
from lib.sft_data_loader import load_sft_data

parser = argparse.ArgumentParser()

parser.add_argument('--per_device_train_batch_size', type=int, default=8)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1.41e-6)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epochs', type=int, default=2)

def format_prompt(example):
    return {
        "text": f"{example['prompt']}\n\n### Response:\n{example['completion']}"
    }

def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding=False)

def SFTtuning(args, hf_dataset, tokenizer, model):
    train_dataset = hf_dataset["train"].map(format_prompt)
    eval_dataset = hf_dataset["test"].map(format_prompt)
    tokenized_train_dataset = train_dataset.map(partial(tokenize, tokenizer=tokenizer))
    tokenized_eval_dataset = eval_dataset.map(partial(tokenize, tokenizer=tokenizer))

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="### Response:\n"
    )

    training_args = TrainingArguments(
        output_dir="./lib/results_SFT_Llama3.2B",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_epochs,
        logging_steps=100,
        save_strategy="no",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=50,
        fp16=True,
        evaluation_strategy="steps",
        report_to="none"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model("lib/Llama-3.2-3B-Instruct-SFT")

def main(args):
    df = pd.read_csv('lib/sft_data.csv')
    df = df.drop_duplicates(subset='prompt', keep='first')
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_auth_token=True)

    model = SFTtuning(args, hf_dataset, tokenizer, model)

if __name__ == "__main__":
    load_sft_data()
    args = parser.parse_args()
    main(args)