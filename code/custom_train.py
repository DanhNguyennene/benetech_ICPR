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
import datetime
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size, timeout_minutes=60*60*5):
    """
    Set up the process group for DDP on Kaggle with NCCL backend with a timeout.

    Parameters:
    - rank: the rank of the process (each process should have a unique rank).
    - world_size: total number of processes.
    - timeout_minutes: timeout duration in minutes (default is 5 minutes).
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5553'  

    timeout = datetime.timedelta(minutes=timeout_minutes)
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timeout)
    
    torch.cuda.set_device(rank)

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
logger = setup_logging()

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
        token_map,
        rank,
        world_size
    ):

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
    
    if rank == 0:
        progress_bar = tqdm(range(len(valid_dl)), desc='Running evaluation...')
    else:
        progress_bar = None


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
        if rank==0:
            progress_bar.update(1)
    if rank==0:
        progress_bar.close()

    all_ids_gathered = [None for _ in range(world_size)]
    all_texts_gathered = [None for _ in range(world_size)]
    label_dict_gathered = [None for _ in range(world_size)]

    dist.all_gather_object(all_ids_gathered, all_ids)
    dist.all_gather_object(all_texts_gathered, all_texts)
    dist.all_gather_object(label_dict_gathered, label_dict)

    all_ids = []
    all_texts = []
    label_dict = []

    for ids in all_ids_gathered:
        all_ids.extend(ids)

    for texts in all_texts_gathered:
        all_texts.extend(texts)

    for labels in label_dict_gathered:
        label_dict.extend(labels)

    if rank == 0:

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
    return None


# -------- Main Function ---------------------------------------------------------#



def run_train_ddp(rank, world_size, cfg):

    setup(rank, world_size)

  
    global logger
    logger = setup_logging()
    print_and_log("Starting training process", logging.INFO)
    
    print_and_log("Loading datasets...", logging.INFO)
    train_parquet_path = cfg.custom.train_parquet_path
    mga_train_ds = ICPRDataset(cfg, train_parquet_path)
    valid_parquet_path = cfg.custom.valid_parquet_path
    mga_valid_ds = ICPRDataset(cfg, valid_parquet_path)
    print_and_log(f"Train dataset size: {len(mga_train_ds)}, Valid dataset size: {len(mga_valid_ds)}", logging.INFO)
    tokenizer = mga_train_ds.processor.tokenizer
    cfg.model.len_tokenizer = len(tokenizer)
    cfg.model.pad_token_id = tokenizer.pad_token_id
    cfg.model.decoder_start_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]
    cfg.model.bos_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]


    # ------- data collators --------------------------------------------------------------#
    collate_fn = ICPRCollator(tokenizer=tokenizer)
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        mga_train_ds, num_replicas=world_size, rank=rank
    )

    train_dl = DataLoader(
        mga_train_ds,
        batch_size=cfg.train_params.train_bs,
        collate_fn=collate_fn,
        num_workers=cfg.train_params.num_workers,
        sampler=train_sampler,
        drop_last=True
    )


    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        mga_valid_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    valid_dl = DataLoader(
        valid_sampler,
        batch_size=cfg.train_params.valid_bs,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=cfg.train_params.num_workers,
        drop_last=True
    )

    # ------- Wandb --------------------------------------------------------------------#
    print_line()
    if cfg.use_wandb:
        print("initializing wandb run...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg_dict)
    print_line()



    # ------- Config -------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))


    # ------- Model --------------------------------------------------------------------#
    print_line()
    print_and_log("Creating ICPR model...", logging.INFO)
    model = ICPRModel(cfg)
    print_and_log(f"Model architecture:\n{model}", logging.DEBUG)

    model = model.to(rank)

    
    model = DDP(model, device_ids=[rank])
    

    print_line()
    print("creating the optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,

    )

    print_line()
    print("creating the scheduler...")

    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.grad_accumulation
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------- Accelerator --------------------------------------------------------------#
    print_line()
    print("accelerator setup...")

    accelerator = Accelerator(
        mixed_precision='bf16',  # or 'fp16' if you prefer
        device_placement=True,   # Ensure proper device placement
    )  

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- training setup --------------------------------------------------------------#
    best_f1 = 0
    best_accuracy = 0
    save_trigger = cfg.train_params.save_trigger
    patience_tracker = 0

    min_delta = 0.001


    current_iteration = 0

    # ------- EMA -----------------------------------------------------------------------#
    if cfg.train_params.use_ema:
        print_line()
        decay_rate = cfg.train_params.decay_rate
        ema = EMA(model, decay=decay_rate)
        ema.register()

        print(f"EMA will be used during evaluation with decay {round(decay_rate, 4)}...")
        print_line()

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()
    num_vbar = 0
    num_hbar = 0
    num_line = 0
    num_scatter = 0

    for epoch in tqdm(range(num_epochs), desc='Processing epoch...'):
        train_sampler.set_epoch(epoch)
        epoch_progress = 0
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()
        loss_meter_main = AverageMeter()
        loss_meter_cls = AverageMeter()

        model.train()
        for step, batch in enumerate(train_dl):
            num_vbar += len([ct for ct in batch['chart_type'] if ct == 'vertical_bar'])
            num_hbar += len([ct for ct in batch['chart_type'] if ct == 'horizontal_bar'])
            num_line += len([ct for ct in batch['chart_type'] if ct == 'line'])
            num_scatter += len([ct for ct in batch['chart_type'] if ct == 'scatter'])

            loss, loss_dict = model(
                flattened_patches=batch["flattened_patches"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            accelerator.backward(loss)
            epoch_progress += 1

            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_value)  # added gradient clip

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())
                loss_meter_main.update(loss_dict["loss_main"].item())
                loss_meter_cls.update(loss_dict["loss_cls"].item())


                # ema ---
                if cfg.train_params.use_ema:
                    ema.update()

                progress_bar.set_description(
                    f"STEP: {epoch_progress+1:5}/{len(train_dl):5}. "
                    f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )
                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    wandb.log({"main_loss": round(loss_meter_main.avg, 5)}, step=current_iteration)
                    wandb.log({"cls_loss": round(loss_meter_cls.avg, 5)}, step=current_iteration)

                    wandb.log({"num_vbar": num_vbar}, step=current_iteration)
                    wandb.log({"num_hbar": num_hbar}, step=current_iteration)
                    wandb.log({"num_line": num_line}, step=current_iteration)
                    wandb.log({"num_scatter": num_scatter}, step=current_iteration)

                    wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (epoch_progress + 1) % cfg.train_params.eval_frequency == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()

                # apply ema if it is used ---
                if cfg.train_params.use_ema:
                    ema.apply_shadow()

                f1_and_acc = run_evaluation(
                    cfg,
                    model=model,
                    valid_dl=valid_dl,
                    tokenizer=tokenizer,
                    token_map=TOKEN_MAP,
                    rank=rank,
                    world_size=world_size
                )
                if rank == 0:
                    f1 = f1_and_acc['f1_score']
                    acc = f1_and_acc['accuracy']
                    print_and_log(f"Evaluation results - F1 Score: {f1_and_acc['f1_score']:.4f}, Accuracy: {f1_and_acc['accuracy']:.4f}", logging.INFO)
            
                print_line()
                et = as_minutes(time.time()-start_time)
                print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")
                

                f1_improved = (f1 - best_f1) > min_delta
                accuracy_improved = (acc - best_accuracy) > min_delta

                if f1_improved or accuracy_improved:
                    
                    best_f1 = max(best_f1, f1)
                    best_accuracy = max(best_accuracy, acc)
                    patience_tracker = 0
                else:
                   
                    patience_tracker += 1
                print_line()
                print(f"Current f1_score: {round(f1,4)}")
                print(f"Current accuracy score: {round(acc, 4)}")
                print_line()
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': current_iteration,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    # 'lb': lb,
                }

                if rank == 0:  
                    save_checkpoint(cfg_dict, model_state)

                
                model.train()
                torch.cuda.empty_cache()

                # ema ---
                if cfg.train_params.use_ema:
                    if rank == 0:
                        ema.apply_shadow()
                        ema.restore()
                    else:
                        pass

                print_line()
                cleanup()
                if patience_tracker >= cfg_dict['train_params']['patience']:
                    print("Early stopping triggered. Stopping training...")
                    model.eval()
                    return
    cleanup()



def main_ddp(world_size, cfg):
    """Spawn multiple processes for DDP training."""
    mp.spawn(
        run_train_ddp,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )

# Main entry point
@hydra.main(version_base=None, config_path="../conf/r_final", config_name="conf_r_final")
def run_training(cfg):
    """Entry point for training, with Hydra config."""
    world_size = torch.cuda.device_count()  # Get the number of available GPUs

    # Start DDP training
    main_ddp(world_size, cfg)

if __name__ == "__main__":
    run_training()
