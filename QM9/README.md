## Usage
This section provides instructions for running the code for the small molecule generation task.

### Requirements
To run the code, ensure that the following packages are installed:
* Python: 3.9
* PyTorch: 2.8.0
* CUDA: 11.8

Additional dependencies are listed in the `requirements.txt` file in the current directory and can be installed with:
```
pip install -r requirements.txt
```

Model training was performed on four NVIDIA V100 GPUs

### Fine-Tuning
To fine-tune the model on multiple objectives using IterativeRS, run:
```
python iterativers.py
```

After fine-tuning, the IterativeRS model weights will be saved in the current directory as `lib/fine_tuned_gpt2/model_weights_merged.pth`.

### Evaluation
To evaluate performance, load the saved IterativeRS model weights into the pre-trained GPT-2 model and generate molecular SMILES strings.

The pre-trained GPT-2 model can be loaded as:
```
from transformers import AutoModelForCausalLM
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

tokenizer = SmilesTokenizer('lib/vocab_customized.txt')
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('[CLS]')

model = AutoModelForCausalLM.from_pretrained("lib/trained_gpt_moses")
```
The PAMNet model is then used to evaluate the generated SMILES.

First, the generated SMILES should be converted into a PyTorch Geometric (PyG) dataset. To do this, follow the setup instructions in the [PAMNet repository](https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN) to create the required Python environment.

Then, within that environment, run the following command to convert the generated SMILES (assuming they are saved in `lib/generated_smiles.csv`:
```
python smiles_to_mol.py --smiles_dir lib/generated_smiles.csv
```
You can modify the `--smiles_dir` argument to specify a different file containing your generated SMILES.
After running the above command, the resulting dataset will be saved as `lib/generated_PyG_dataset.pt`.

Finally, please follow the instructions in the [PAMNet repository](https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN), to train and run the PAMNet model for evaluation on the generated PyTorch Geometric dataset.
