import argparse
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from lib.smiles_converter import from_smiles_to_PyG

parser = argparse.ArgumentParser()

parser.add_argument('--smiles_dir', type=str, default='lib/generated_smiles.csv')
parser.add_argument('--num_samples', type=int, default=10)

class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)

def main(args):
    df = pd.read_csv(args.smiles_dir)
    df['SMILES'] = df['SMILES'].apply(lambda x: "".join(x.split()))
    df = df.drop_duplicates(subset='SMILES', keep='first')
    smiles_list = df['SMILES'].to_list()

    data_list = []
    i = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES skipped: {smiles}")
            continue
        mol_with_h = Chem.AddHs(mol)
        smiles_with_h = Chem.MolToSmiles(mol_with_h)

        for _ in range(args.num_samples):
            try:
                data = from_smiles_to_PyG(smiles_with_h)
                data_list.append(data)
            except Exception as e:
                print(f"Failed to process SMILES '{smiles}': {e}")
        
        i+=1
        if i%1000==1:
            print(f"{i} samples are processed.")
    
    dataset = MoleculeDataset(data_list)

    torch.save(dataset, "lib/generated_PyG_dataset.pt")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
