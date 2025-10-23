import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_sft_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("openai/summarize_from_feedback", "comparisons")

    prompt, summary = [], []
    faithful_score, summary_score, deberta_score = [], [], []

    tokenizer_faithful = AutoTokenizer.from_pretrained("CogComp/bart-faithful-summary-detector")
    model_faithful = AutoModelForSequenceClassification.from_pretrained("CogComp/bart-faithful-summary-detector").to(device)

    tokenizer_summ = AutoTokenizer.from_pretrained("Tristan/gpt2_reward_summarization")
    model_summ = AutoModelForSequenceClassification.from_pretrained("Tristan/gpt2_reward_summarization").to(device)

    tokenizer_deberta = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
    model_deberta = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2").to(device)

    idx = 0

    for example in ds["train"]:
        iferror1, iferror2 = 0, 0
        post_text = example['info']['post']
        prompt_text = f"Summarize this Reddit post in a concise way: {post_text}"
        summary_1, summary_2 = example['summaries'][0]['text'], example['summaries'][1]['text']

        with torch.no_grad():
            pair_1 = tokenizer_faithful(text=summary_1, text_pair=post_text, return_tensors='pt').to(device)
            pair_2 = tokenizer_faithful(text=summary_2, text_pair=post_text, return_tensors='pt').to(device)
            try:
                score_1_faithful = model_faithful(**pair_1)
            except RuntimeError as e:
                print(f"RuntimeError: {e} for faithful score of example {idx}")
                iferror1 = 1
            try:
                score_2_faithful= model_faithful(**pair_2)
            except RuntimeError as e:
                print(f"RuntimeError: {e} for faithful score of example {idx}")
                iferror2 = 1

            pair_1 = tokenizer_summ(text=summary_1, text_pair=post_text, return_tensors='pt').to(device)
            pair_2 = tokenizer_summ(text=summary_2, text_pair=post_text, return_tensors='pt').to(device)
            try:
                score_1_summary= model_summ(**pair_1)
            except RuntimeError as e:
                print(f"RuntimeError: {e} for summary score of example {idx}")
                iferror1 = 1
            try:
                score_2_summary = model_summ(**pair_2)
            except RuntimeError as e:
                print(f"RuntimeError: {e} for summary score of example {idx}")
                iferror2 = 1

            pair_1 = tokenizer_deberta(text=summary_1, text_pair=post_text, return_tensors='pt').to(device)
            pair_2 = tokenizer_deberta(text=summary_2, text_pair=post_text, return_tensors='pt').to(device)
            try:
                score_1_deberta= model_deberta(**pair_1)
            except RuntimeError as e:
                print(f"RuntimeError: {e} for deberta score of example {idx}")
                iferror1 = 1
            try:
                score_2_deberta= model_deberta(**pair_2)
            except RuntimeError as e:
                print(f"RuntimeError: {e} for deberta score of example {idx}")
                iferror2 = 1
        
        if iferror1==0 and iferror2==1:
            prompt.append(prompt_text)
            summary.append(summary_1)
        elif iferror1==1 and iferror2==0:
            prompt.append(prompt_text)
            summary.append(summary_2)
        elif iferror1==0 and iferror2==0:
            prompt.append(prompt_text)
            score_1, score_2 = 0, 0
            if score_1_faithful.logits[0,1].cpu().item()>=score_2_faithful.logits[0,1].cpu().item():
                score_1+=1
            else:
                score_2+=1
            if score_1_summary.logits.cpu().item()>=score_2_summary.logits.cpu().item():
                score_1+=1
            else:
                score_2+=1
            if score_1_deberta.logits.cpu().item()>=score_2_deberta.logits.cpu().item():
                score_1+=1
            else:
                score_2+=1
            
            if score_1>score_2:
                summary.append(summary_1)
            else:
                summary.append(summary_2)

        idx+=1
        if idx%1000==1:
            print(f"{idx} samples are processed!")
        
        torch.cuda.empty_cache()

    data = {
        'prompt': prompt,
        'completion': summary
        }

    df = pd.DataFrame(data)

    file_name = "sft_data.csv"
    df.to_csv(file_name, index=False)