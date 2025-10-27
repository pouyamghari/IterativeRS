set -ex

python -u split_data.py

accelerate launch --config_file accelerate_config.yaml SFT.py
python -u save_first_time_iterativers.py --init SFT

for epoch in {1..4}
do
  for task in faithful summary deberta
  do
    accelerate launch --config_file accelerate_config.yaml Iterativers.py --epoch $epoch --task $task --num_epochs_per_step 5 --with_sft 1
  done
  python -u merge_iterativers.py --with_sft 1 --split train
done

python -u merge_iterativers.py --with_sft 1 --split test
