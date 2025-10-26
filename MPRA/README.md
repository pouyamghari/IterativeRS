## Usage
This section provides instructions for running the code for the DNA sequence generation task.

### Requirements
To run the code, ensure that the following packages are installed:
* Python: 3.9
* PyTorch: 2.7.0
* CUDA: 11.8

Additional dependencies are listed in the `requirements.txt` file in the current directory and can be installed with:
```
pip install -r requirements.txt
```

Model training was performed on four NVIDIA V100 GPUs
### Load Dataset
The MPRA dataset should be placed in the current directory. 

To obtain and load the dataset, please follow the instructions in the [boda2 repository](https://github.com/sjgosai/boda2/tree/main), specifically the tutorial provided in [load_malinois_model.ipynb](https://github.com/sjgosai/boda2/blob/main/tutorials/load_malinois_model.ipynb).

### Pre-Training
A GPT-2 model should be pre-trained on the MPRA dataset before fine-tuning.
To pre-train the model, run:
```
python GPT2PreTrain.py
```

### Fine-Tuning
To fine-tune the model on multiple objectives using IterativeRS, run:
```
python iterativers.py --merge_strategy uniform
```
Setting the `--merge_strategy` argument to `uniform` merges expert models with uniform weights.
To perform experiments with selective merging, set `--merge_strategy` to `selection`.

After fine-tuning, the IterativeRS model weights will be saved in the current directory as `model_weights.pth`.

To evaluate performance, load the saved IterativeRS model weights into the pre-trained GPT-2 model and generate DNA sequences. 

The Malinois model is then used to evaluate the generated sequences. Please follow the instructions in the [boda2 repository](https://github.com/sjgosai/boda2/tree/main), as detailed in the [load_malinois_model.ipynb](https://github.com/sjgosai/boda2/blob/main/tutorials/load_malinois_model.ipynb) to load and run the Malinois model for evaluation.
