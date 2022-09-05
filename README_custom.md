# Training donut on SROIE dataset

## Setup

```shell
python -m venv venv
source venv/bin/activate
pip install .
```

## Donut SROIE training configuration

Dataset downloaded from <https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2>. download `0325updated.task2train(626p)` to get the training data from this link. However, data only contains the fields {"company", "date", "address", "total"}

`Config.yaml` for SROIE training:

    resume_from_checkpoint_path: null # only used for resume_from_checkpoint option in PL
    result_path: "./result"
    pretrained_model_name_or_path: "naver-clova-ix/donut-base" # loading a pre-trained model (from moldehub or path)
    dataset_name_or_paths: ["datasets/sroie-donut"] # loading datasets (from moldehub or path)
    sort_json_key: False # cord dataset is preprocessed, and publicly available at https://huggingface.co/datasets/naver-clova-ix/cord-v2
    train_batch_sizes: [4]
    val_batch_sizes: [1]
    input_size: [1280, 960]
    max_length: 768
    align_long_axis: False
    num_nodes: 1
    seed: 2022
    lr: 3e-5
    warmup_steps: 300 # 800/8*30/10, 10%
    num_training_samples_per_epoch: 800
    max_epochs: 30
    max_steps: -1  # infinite, since max_epochs is specified
    num_workers: 8
    val_check_interval: 1.0
    check_val_every_n_epoch: 6
    gradient_clip_val: 1.0
    verbose: True

## Convert SROIE into DONUT format

Script for converting SROIE dataset into donut training format:

Make sure the source and destination dataset paths are set inside script before running.

```shell
python dataset/sroie_2_donut.py
```

## Train donut on SROIE

```shell
python train.py --config config/train_sroie.yaml
```
