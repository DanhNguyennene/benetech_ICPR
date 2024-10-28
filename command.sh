#!/bin/bash

python -m pip install --upgrade pip
pip install huggingface_hub
pip install zss
pip install -U albumentations wandb 'numpy<2' joblib nltk

git clone https://huggingface.co/datasets/DanTheGuy/Benetech_PlotQa_DVQA_combined_matcha_complete
rm -r benetech_ICPR
git clone -b dev1 https://github.com/DanhNguyennene/benetech_ICPR.git

cd benetech_ICPR
pip install -r requirements.txt
cd ..

python -c "
import yaml
import os

config = {
  'debug': False,
  'use_random_seed': True,
  'seed': 461,
  'fold': '???',
  'train_folds': '???',
  'valid_folds': '???',
  'use_wandb': False,
  'all_data': True,
  'add_syn': False,
  'add_pl': False,
  'pl_multiplier': 8,
  'extracted_multiplier': 16,
  'original_multiplier': 3,
  'use_augmentations': False,
  'num_workers': 4,
  'tags': ['final'],
  'model': {
    'backbone_path': 'google/matcha-base',
    'max_length': 512,
    'max_patches': 1024,
    'patch_size': 16,
    'len_tokenizer': '???',
    'pad_token_id': '???',
    'decoder_start_token_id': '???',
    'bos_token_id': '???',
    'max_length_generation': 512
  },
  'awp': {
    'use_awp': False,
    'awp_trigger': 0.0,
    'awp_trigger_epoch': 1,
    'adv_lr': 8e-5,
    'adv_eps': 0.001
  },
  'train_params': {
    'train_bs': 1,
    'valid_bs': 2,
    'num_epochs': 10,
    'grad_accumulation': 16,
    'warmup_pct': 0.05,
    'save_trigger': -1.0,
    'use_fp16': True,
    'eval_frequency': 3300,
    'patience': 100,
    'use_ema': True,
    'decay_rate': 0.9925,
    'num_workers': 4,
    'epoch_saved': 2,
    'epoch_frequency': 5
  },
  'optimizer': {
    'lr': 5e-5,
    'weight_decay': 1e-5,
    'grad_clip_value': 1.0
  },
  'outputs': {
    'model_dir': '/workspace/model_dir'
  },
  'fold_metadata': {
    'n_folds': 2,
    'fold_dir': '/workspace/Benetech_PlotQa_DVQA_combined_matcha_complete/data',
    'fold_path': 'train-00000-of-00001.parquet'
  },
  'competition_dataset': {
    'parquet_dict': '/workspace/Benetech_PlotQa_DVQA_combined_matcha_complete/data'
  },
  'wandb': {
    'project': 'mga-dev-a1',
    'run_name': 'rb-exp100-r-final'
  },
  'images': {
    'rsz_height': 512,
    'rsz_width': 512
  },
  'generation': {
    'max_length': 1024
  }
}

os.makedirs('/workspace/model_dir', exist_ok=True)
with open('/workspace/benetech_ICPR/conf/r_final/conf_r_final.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
"

HYDRA_FULL_ERROR=1 python /workspace/benetech_ICPR/code/custom_train.py \
    --config-name conf_r_final \
    fold=0 \
    use_wandb=true \
    debug=true
