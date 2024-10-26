import glob
import json
import os
import random
import time
from datetime import datetime
from copy import deepcopy
from textwrap import wrap
import re
import logging
from logging.handlers import RotatingFileHandler
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Any
import torch
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GenerationConfig, get_cosine_schedule_with_warmup
import torch.distributed as dist
from accelerate import Accelerator
import subprocess
import signal
import psutil
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
def setup(rank, world_size):
    """
    Set up the process group for DDP on Kaggle with NCCL backend.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5553'  # Port must be free

    # Initialize the process group with the correct backend (NCCL for GPUs)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the GPU device for each rank (process)

def cleanup():
    """
    Clean up the process group.
    """
    dist.destroy_process_group()
TOKEN_MAP = {
  "axes": ["<axes>", "</axes>"],
  "chart-type": ["<chart-type>", "</chart-type>"],
  "bars": ["<bars>", "</bars>"],
  "data-series": ["<data-series>", "</data-series>"],
  "plot-bb": ["<plot-bb>", "</plot-bb>"],
  "source": ["<source>", "</source>"],
  "text": ["<text>", "</text>"],
  "text_display": ["<text_display>", "</text_display>"],
  "visual-elements": ["<visual-elements>", "</visual-elements>"],
  "x-axis": ["<x-axis>", "</x-axis>"],
  "y-axis": ["<y-axis>", "</y-axis>"],
  "tick-type": ["<tick-type>", "</tick-type>"],
  "ticks": ["<ticks>", "</ticks>"],
  "values-type": ["<values-type>", "</values-type>"],
  "tick_pt": ["<tick_pt>", "</tick_pt>"],
  "x": ["<x>", "</x>"],
  "y": ["<y>", "</y>"],
  "height": ["<height>", "</height>"],
  "width": ["<width>", "</width>"],
  "x0": ["<x0>", "</x0>"],
  "x1": ["<x1>", "</x1>"],
  "x2": ["<x2>", "</x2>"],
  "x3": ["<x3>", "</x3>"],
  "y0": ["<y0>", "</y0>"],
  "y1": ["<y1>", "</y1>"],
  "y2": ["<y2>", "</y2>"],
  "y3": ["<y3>", "</y3>"],
  "id": ["<id>", "</id>"],
  "polygon": ["<polygon>", "</polygon>"],
  "role": ["<role>", "</role>"],
  "bos_token" : ["</s>"]
}

def cleanup_processes():
    """Clean up any hanging CUDA and Python processes."""
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Kill any existing distributed processes
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    
    # Clean up multiprocessing
    for p in mp.active_children():
        p.terminate()
        p.join()
    
    print("Cleanup completed")
def setup_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.DEBUG)

    f_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    f_handler.setLevel(logging.DEBUG)

    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    logger.addHandler(f_handler)

    return logger


def print_and_log(message, level=logging.INFO):
    print(message)
    if level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    elif level == logging.CRITICAL:
        logger.critical(message)


try:
    from r_final.custom_dataloader import ICPRCollator
    from r_final.custom_dataset import (TOKEN_MAP, ICPRDataset,
                                     create_train_transforms)
    from r_final.custom_model import ICPRModel
    from utils.constants import EXCLUDE_IDS
    from utils.data_utils import process_annotations
    from utils.metric_utils import JSONParseEvaluator
    from utils.metric_utils import compute_metrics
    from utils.train_utils import (EMA, AverageMeter, as_minutes, get_lr,
                                   init_wandb, print_gpu_utilization,
                                   print_line, save_checkpoint,
                                   seed_everything)

except Exception as e:
    print(e)
    raise ImportError

pd.options.display.max_colwidth = 1000
BOS_TOKEN = TOKEN_MAP["bos_token"]



#  -------- Evaluation -------------------------------------------------------------#
def parse_data_series(content: str) -> Dict[str, List[Any]]:
    data = {}
    pairs = re.findall(r'<(\w+)>(.*?)</\w+>', content)
    for key, value in pairs:
        if key not in data:
            data[key] = []
        try:
            data[key].append(float(value))
        except ValueError:
            data[key].append(value)
    return data

def parse_text_display(content: str) -> List[Dict[str, Any]]:
    elements = []
    polygons = re.findall(r'<polygon>(.*?)</polygon>', content)
    texts = re.findall(r'<text>(.*?)</text>', content)
    for polygon, text in zip(polygons, texts):
        element = {'text': text}
        coords = re.findall(r'<(\w+)>(\d+)</\w+>', polygon)
        for key, value in coords:
            element[key] = int(value)
        elements.append(element)
    return elements


def extraction(content: str, bos: str, eos: str) -> str:
    content = content.split(bos)[1]
    content = content.split(eos)[0]
    return content


def detect_nested_tags(content: str, token_map: Dict[str, List[str]]) -> List[str]:
    nested_tags = []
    for token, tags in token_map.items():
        start_tag = tags[0].replace('<', r'\<').replace('>', r'\>')
        if re.search(start_tag, content):
            nested_tags.append(token)
    return nested_tags
    

def build_nested_dict(
    pred_str: str,
    token_map: Dict[str, List[str]],
    token_order: List[str]
)-> Dict[str, Any]:
    result = {}
    
    for token in token_order:
        start_tag, end_tag = token_map[token]
        if start_tag in pred_str and end_tag in pred_str:
            content = extraction(pred_str, start_tag, end_tag)
            
            if token == 'data-series':
                result[token] = parse_data_series(content)
            elif token == 'text_display':
                result[token] = parse_text_display(content)
            else:
                nested_tags = detect_nested_tags(content, token_map)
                if nested_tags:
                    result[token] = build_nested_dict(content, token_map, nested_tags)
                else:
                    try:
                        result[token] = int(content.strip())
                    except ValueError:
                        result[token] = content.strip()
    return result


def post_processing(pred_str: str, token_map: Dict[str, List[str]], token_order: List[str] = ['chart-type', 'plot-bb', 'data-series', 'text_display']) -> Dict[str, Any]:
    return build_nested_dict(pred_str, token_map, token_order)





def run_evaluation(
        cfg: OmegaConf, 
        model, 
        valid_dl, 
        tokenizer, 
        token_map):

    # # config for text generation ---
    conf_g = {
        "max_new_tokens": cfg.model.max_length_generation,  # 256,
        "do_sample": False,
        "top_k": 1,
        "use_cache": True,
    }

    generation_config = GenerationConfig(**conf_g)

    # put model in eval mode ---
    model.eval()

    all_ids = []
    all_texts = []
    label_dict = []
    progress_bar = tqdm(range(len(valid_dl)), desc='Running evaluation...')


    for batch in valid_dl:
        
        with torch.no_grad():
            batch_ids = batch["id"]
            
            generated_ids = model.backbone.generate(
                flattened_patches=batch['flattened_patches'],
                attention_mask=batch['attention_mask'],
                generation_config=generation_config,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


            all_ids.extend(batch_ids)
            all_texts.extend(generated_texts)
            label_dict.extend(batch['texts'])
        progress_bar.update(1)
    progress_bar.close()

    label_dicts = [
        post_processing(
            label_str,
            TOKEN_MAP,
        ) for label_str in label_dict
    ]

    # prepare output dataframe ---
    preds_dict = []
    for this_id, this_text in zip(all_ids, all_texts):
        pred_dictionary = post_processing(this_text, token_map)
        preds_dict.append((this_id,pred_dictionary))
        

    eval_JSON = JSONParseEvaluator()

    f1_score = eval_JSON.cal_f1(
        preds=preds_dict,
        answers = label_dicts
    )

    accuracy = sum([eval_JSON.cal_acc(
        pred=pred,
        answer=label
    ) for pred, label in zip(preds_dict, label_dicts)]) / len(preds_dict)

    return {
        'f1_score': f1_score,
        'accuracy': accuracy
    }


# -------- Main Function ---------------------------------------------------------#



def run_train_ddp(rank, world_size, cfg):
    setup(rank, world_size)  # Set up process group for distributed training
    global logger
    logger = setup_logging()
    print_and_log("Starting training process", logging.INFO)
    
    # Load the datasets
    directory = cfg.competition_dataset.parquet_dict
    train_files = glob.glob(os.path.join(directory, "train*.parquet"))
    mga_train_ds = ICPRDataset(cfg, train_files)

    valid_files = glob.glob(os.path.join(directory, "validation*.parquet"))
    mga_valid_ds = ICPRDataset(cfg, valid_files)
    print_and_log(f"Train dataset size: {len(mga_train_ds)}, Valid dataset size: {len(mga_valid_ds)}", logging.INFO)

    tokenizer = mga_train_ds.processor.tokenizer
    cfg.model.len_tokenizer = len(tokenizer)
    cfg.model.pad_token_id = tokenizer.pad_token_id
    cfg.model.decoder_start_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]
    cfg.model.bos_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]

    # Dataloader Setup
    collate_fn = ICPRCollator(tokenizer=tokenizer)

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(mga_train_ds, num_replicas=world_size, rank=rank)
        train_dl = DataLoader(
            mga_train_ds, batch_size=cfg.train_params.train_bs, collate_fn=collate_fn,
            num_workers=cfg.train_params.num_workers, pin_memory=True, sampler=train_sampler
        )
    else:
        train_dl = DataLoader(
            mga_train_ds, batch_size=cfg.train_params.train_bs, collate_fn=collate_fn,
            num_workers=cfg.train_params.num_workers, pin_memory=True, shuffle=True
        )

    valid_dl = DataLoader(
        mga_valid_ds, batch_size=cfg.train_params.valid_bs, collate_fn=collate_fn,
        shuffle=False, pin_memory=True, num_workers=cfg.train_params.num_workers,
    )

    # WandB Initialization
    if cfg.use_wandb:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg_dict)

    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    # Model Creation and CUDA Placement
    model = ICPRModel(cfg)
    model = model.cuda(rank)  # Move model to GPU at rank

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Optimizer and Scheduler Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.grad_accumulation
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # Accelerator setup explicitly with GPU
    accelerator = Accelerator(mixed_precision='bf16', device_placement=True)  # Ensures CUDA compatibility

    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)
    
    # Training Loop
    best_f1 = 0
    best_accuracy = 0
    save_trigger = cfg.train_params.save_trigger
    patience_tracker = 0
    min_delta = 0.001
    current_iteration = 0

    for epoch in tqdm(range(num_epochs), desc='Processing epoch...'):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        model.train()
        
        print_and_log(f"Starting epoch {epoch + 1}/{num_epochs}", logging.INFO)
        
        for step, batch in enumerate(train_dl):
            # Print a simple debug message every few steps
            if step % 10 == 0:
                print_and_log(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_dl)}", logging.DEBUG)

            # Training logic here
            loss, loss_dict = model(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            accelerator.backward(loss)

            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_meter.update(loss.item())

                # Logging with WandB
                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                current_iteration += 1

        # At the end of the epoch, log average metrics
        print_and_log(f"End of epoch {epoch + 1}: Average Loss: {loss_meter.avg}", logging.INFO)

        # Evaluation and Early Stopping
        if (epoch + 1) % cfg.train_params.epoch_frequency == 0:
            model.eval()
            f1_and_acc = run_evaluation(cfg, model=model, valid_dl=valid_dl, tokenizer=tokenizer)
            f1 = f1_and_acc['f1_score']
            acc = f1_and_acc['accuracy']
            print_and_log(f"Evaluation - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}", logging.INFO)

            # Early stopping logic
            if f1 > best_f1 + min_delta or acc > best_accuracy + min_delta:
                best_f1, best_accuracy = max(best_f1, f1), max(best_accuracy, acc)
                patience_tracker = 0
            else:
                patience_tracker += 1

            if patience_tracker >= cfg_dict['train_params']['patience']:
                print_and_log("Early stopping triggered. Stopping training...", logging.INFO)
                return

    if dist.get_rank() == 0:
        save_checkpoint(cfg_dict, {'step': current_iteration, 'epoch': num_epochs, 'state_dict': model.state_dict()})

# Main DDP Setup
def main_ddp(world_size, cfg):
    cleanup_processes()
    if world_size > 1:
        mp.spawn(run_train_ddp, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        run_train_ddp(rank=0, world_size=1, cfg=cfg)

@hydra.main(version_base=None, config_path="../conf/r_final", config_name="conf_r_final")
def run_training(cfg):
    world_size = torch.cuda.device_count()  # Confirm available GPUs
    main_ddp(world_size, cfg)

if __name__ == "__main__":
    run_training()
