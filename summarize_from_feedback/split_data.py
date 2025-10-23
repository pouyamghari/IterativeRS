import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lib.dataset_loader import load_data

def load_split_data():
    os.makedirs('lib/data_splits', exist_ok=True)

    df = pd.read_csv('lib/data/train_data.csv')
    df_cleaned = df.drop_duplicates(subset='prompt', keep='first')
    df_cleaned = df_cleaned[df_cleaned['prompt'].apply(lambda x: isinstance(x, str))]
    
    df_train = df_cleaned['prompt']
    df_train.to_csv('lib/data_splits/train_samples.csv', index=False)

    df_val = pd.read_csv('lib/data/validation_data.csv')
    df_val_cleaned = df_val.drop_duplicates(subset='prompt', keep='first')
    df_val_cleaned = df_val_cleaned[df_val_cleaned['prompt'].apply(lambda x: isinstance(x, str))]

    df_val_test = df_val_cleaned[['prompt', 'summary']]

    df_validation, df_test = train_test_split(df_val_test, test_size=0.5, random_state=42)

    df_validation.to_csv('lib/data_splits/validation_samples.csv', index=False)
    df_test.to_csv('lib/data_splits/test_samples.csv', index=False)

if __name__ == "__main__":
    load_data("train")
    load_data("validation")
    load_split_data()