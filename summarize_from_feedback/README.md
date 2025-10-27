## Usage
This section provides instructions for running the code for the text summarization task.

### Requirements
To run the code, ensure that the following packages are installed:
* Python: 3.9
* PyTorch: 2.4.0
* CUDA: 11.8

Additional dependencies are listed in the `requirements.txt` file in the current directory and can be installed with:
```
pip install -r requirements.txt
```

Model training was performed on four NVIDIA A100 GPUs.
### Data Processing
The dataset used for the experiments is the [Summariz from Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback/viewer/comparisons) dataset. 

To split the dataset into training, validation, and test sets, run:
```
python split_data.py
```

### Fine-Tuning
To fine-tune the model on multiple objectives using IterativeRS, as described in the paper, the model is iteratively fine-tuned using `Iterativers.py`.
For example, to perform four merging steps during training, where each merging step includes five training epochs per objective, run:
```
python save_first_time_iterativers.py --init None

for epoch in {1..4}
do
  for task in faithful summary deberta
  do
    accelerate launch \
      --config_file accelerate_config.yaml \
      Iterativers.py \
      --epoch $epoch \
      --task $task \
      --num_epochs_per_step 5 \
      --with_sft 0
  done
  python merge_iterativers.py --with_sft 0 --split train
done
```
The above commands perform PPO fine-tuning without SFT. To perform PPO fine-tuning with SFT, first train the SFT model by running:
```
accelerate launch --config_file accelerate_config.yaml SFT.py
```
After the SFT model is trained, perform IterativeRS fine-tuning by running:
```
python save_first_time_iterativers.py --init SFT

for epoch in {1..4}
do
  for task in faithful summary deberta
  do
    accelerate launch \
      --config_file accelerate_config.yaml \
      Iterativers.py \
      --epoch $epoch \
      --task $task \
      --num_epochs_per_step 5 \
      --with_sft 1
  done
  python merge_iterativers.py --with_sft 1 --split train
done
```
After training, the expert models can be merged to obtain the final IterativeRS model:
```
python merge_iterativers.py --with_sft 1 --split test
```
> Note: In this setup, IterativeRS performs selective merging. When `--split` is set to `test`, the validation set is used for determining the merging weights.

The final IterativeRS model without SFT is saved in:
```
lib/fine_tuned_llama/model_iterativers
```
The IterativeRS model fine-tuned with SFT is saved in:
```
lib/fine_tuned_llama/model_iterativers_SFT
```
