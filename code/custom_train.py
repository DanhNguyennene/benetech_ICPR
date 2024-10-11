import glob
import json
import os
import random
import time
from copy import deepcopy
from textwrap import wrap

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

TOKEN_MAP = {
  "axes": ["[<axes>]", "[</axes>]"],
  "chart-type": ["[<chart-type>]", "[</chart-type>]"],
  "bars": ["[<bars>]", "[</bars>]"],
  "data-series": ["[<data-series>]", "[</data-series>]"],
  "plot-bb": ["[<plot-bb>]", "[</plot-bb>]"],
  "source": ["[<source>]", "[</source>]"],
  "text": ["[<text>]", "[</text>]"],
  "visual-elements": ["[<visual-elements>]", "[</visual-elements>]"],
  "x-axis": ["[<x-axis>]", "[</x-axis>]"],
  "y-axis": ["[<y-axis>]", "[</y-axis>]"],
  "tick-type": ["[<tick-type>]", "[</tick-type>]"],
  "ticks": ["[<ticks>]", "[</ticks>]"],
  "values-type": ["[<values-type>]", "[</values-type>]"],
  "tick_pt": ["[<tick_pt>]", "[</tick_pt>]"],
  "x": ["[<x>]", "[</x>]"],
  "y": ["[<y>]", "[</y>]"],
  "height": ["[<height>]", "[</height>]"],
  "width": ["[<width>]", "[</width>]"],
  "x0": ["[<x0>]", "[</x0>]"],
  "x1": ["[<x1>]", "[</x1>]"],
  "x2": ["[<x2>]", "[</x2>]"],
  "x3": ["[<x3>]", "[</x3>]"],
  "y0": ["[<y0>]", "[</y0>]"],
  "y1": ["[<y1>]", "[</y1>]"],
  "y2": ["[<y2>]", "[</y2>]"],
  "y3": ["[<y3>]", "[</y3>]"],
  "id": ["[<id>]", "[</id>]"],
  "polygon": ["[<polygon>]", "[</polygon>]"],
  "role": ["[<role>]", "[</role>]"],
  "bos_token" : ["[</s>]"]
}


try:
    from r_final.custom_dataloader import ICPRCollator
    from r_final.custom_dataset import (TOKEN_MAP, ICPRDataset,
                                     create_train_transforms)
    from r_final.custom_model import ICPRModel
    from utils.constants import EXCLUDE_IDS
    from utils.data_utils import process_annotations
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




# -------- Evaluation -------------------------------------------------------------#


# def post_process(pred_string, token_map, delimiter="|"):
#     # get chart type ---
#     chart_options = [
#         "horizontal_bar",
#         "dot",
#         "scatter",
#         "vertical_bar",
#         "line",
#     ]

#     chart_type = "line"  

#     for ct in chart_options:
#         if token_map[ct] in pred_string:
#             chart_type = ct
#             break

#     # get x series ---
#     x_start_tok = token_map["x_start"]
#     x_end_tok = token_map["x_end"]

#     try:
#         x = pred_string.split(x_start_tok)[1].split(x_end_tok)[0].split(delimiter)
#         x = [elem.strip() for elem in x if len(elem.strip()) > 0]
#     except IndexError:
#         x = []

#     # get y series ---
#     y_start_tok = token_map["y_start"]
#     y_end_tok = token_map["y_end"]

#     try:
#         y = pred_string.split(y_start_tok)[1].split(y_end_tok)[0].split(delimiter)
#         y = [elem.strip() for elem in y if len(elem.strip()) > 0]
#     except IndexError:
#         y = []

#     return chart_type, x, y

def post_processing(pred_str: str, token_map: Dict[str, List[str]], )



def run_evaluation(cfg, model, valid_dl, label_df, tokenizer, token_map):

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

    progress_bar = tqdm(range(len(valid_dl)))
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

        progress_bar.update(1)
    progress_bar.close()

    # prepare output dataframe ---
    preds = []
    extended_preds = []
    for this_id, this_text in zip(all_ids, all_texts):
        id_x = f"{this_id}_x"
        id_y = f"{this_id}_y"
        pred_chart, pred_x, pred_y = post_process(this_text, token_map)

        preds.append([id_x, pred_x, pred_chart])
        preds.append([id_y, pred_y, pred_chart])

        extended_preds.append([id_x, pred_x, pred_chart, this_text])
        extended_preds.append([id_y, pred_y, pred_chart, this_text])

    pred_df = pd.DataFrame(preds)
    pred_df.columns = ["id", "data_series", "chart_type"]

    eval_dict = compute_metrics(label_df, pred_df)

    result_df = pd.DataFrame(extended_preds)
    result_df.columns = ["id", "pred_data_series", "pred_chart_type", "pred_text"]
    result_df = pd.merge(label_df, result_df, on="id", how="left")
    result_df['score'] = eval_dict['scores']  # individual scores

    results = {
        "oof_df": pred_df,
        "result_df": result_df,
    }

    for k, v in eval_dict.items():
        if k != 'scores':
            results[k] = v

    print_line()
    print("Evaluation Results:")
    print(results)
    print_line()

    return results


# -------- Main Function ---------------------------------------------------------#



@hydra.main(version_base=None, config_path="../conf/r_final", config_name="conf_r_final")
def run_training(cfg):
    # ------- Datasets ------------------------------------------------------------------#
    # The datasets for LECR Dual Encoder
    # -----------------------------------------------------------------------------------#

    # label_df = process_annotations(cfg)["ground_truth"][0]
    # label_df["original_id"] = label_df["id"].apply(lambda x: x.split("_")[0])
    # label_df = label_df[label_df["original_id"].isin(valid_ids)].copy()
    # label_df = label_df.drop(columns=["original_id"])
    # label_df = label_df.sort_values(by="source")
    # label_df = label_df.reset_index(drop=True)

    parquet_path = cfg.custom.valid_parquet_path  # Path to the validation Parquet file
    label_df = pd.read_parquet(parquet_path)

    train_parquet_path = cfg.custom.train_parquet_path
    train_transforms = create_train_transforms() if cfg.use_augmentations else None
    mga_train_ds = ICPRDataset(cfg, train_parquet_path, transform=train_transforms)

    valid_parquet_path = cfg.custom.valid_parquet_path
    mga_valid_ds = ICPRDataset(cfg, valid_parquet_path)

    tokenizer = mga_train_ds.processor.tokenizer
    cfg.model.len_tokenizer = len(tokenizer)

    cfg.model.pad_token_id = tokenizer.pad_token_id
    cfg.model.decoder_start_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]
    cfg.model.bos_token_id = tokenizer.convert_tokens_to_ids(BOS_TOKEN)[0]


    # ------- data collators --------------------------------------------------------------#
    collate_fn = ICPRCollator(tokenizer=tokenizer)

    train_dl = DataLoader(
        mga_train_ds,
        batch_size=cfg.train_params.train_bs,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=cfg.train_params.num_workers,
    )

    valid_dl = DataLoader(
        mga_valid_ds,
        batch_size=cfg.train_params.valid_bs,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # ------- Wandb --------------------------------------------------------------------#
    print_line()
    if cfg.use_wandb:
        print("initializing wandb run...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg_dict)
    print_line()

    # --- show batch--------------------------------------------------------------------#
    print_line()
    for idx, b in enumerate(train_dl):
        if idx == 16:
            break
        # run_sanity_check(cfg, b, tokenizer, prefix=f"train_{idx}")

    for idx, b in enumerate(valid_dl):
        if idx == 4:
            break
        # run_sanity_check(cfg, b, tokenizer, prefix=f"valid_{idx}")

    # ------- Config -------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    # ------- Model --------------------------------------------------------------------#
    print_line()
    print("creating the ICPR model...")
    model = ICPRModel(cfg)  # get_model(cfg)
    print_line()

    # # # torch 2.0
    # model = model.to("cuda:0")
    # model = torch.compile(model)  # pytorch 2.0

    # ------- Optimizer ----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,

    )

    # ------- Scheduler -----------------------------------------------------------------#
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
        mixed_precision='bf16',  # changed 'fp16' to 'bf16'
    )  # cpu = True

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl)

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.
    save_trigger = cfg.train_params.save_trigger

    patience_tracker = 0
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
    num_histogram = 0
    num_dot = 0
    num_line = 0
    num_scatter = 0

    for epoch in range(num_epochs):
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
            num_dot += len([ct for ct in batch['chart_type'] if ct == 'dot'])
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
                    wandb.log({"num_dot": num_dot}, step=current_iteration)
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

                run_evaluation(
                    cfg,
                    model=model,
                    valid_dl=valid_dl,
                    label_df=label_df,
                    tokenizer=tokenizer,
                    token_map=TOKEN_MAP,
                )

                # lb = result_dict["lb"]
                # oof_df = result_dict["oof_df"]
                # result_df = result_dict["result_df"]
                #
                # print_line()
                # et = as_minutes(time.time()-start_time)
                # print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")
                #
                # is_best = False
                # if lb >= best_lb:
                #     best_lb = lb
                #     is_best = True
                #     patience_tracker = 0
                #
                #     # ---
                #     best_dict = dict()
                #     for k, v in result_dict.items():
                #         if "df" not in k:
                #             best_dict[f"{k}_at_best"] = v
                #
                # else:
                #     patience_tracker += 1
                #
                # print_line()
                # print(f">>> Current LB = {round(lb, 4)}")
                # for k, v in result_dict.items():
                #     if ("df" not in k) & (k != "lb"):
                #         print(f">>> Current {k}={round(v, 4)}")
                # print_line()
                #
                # if is_best:
                #     oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_fold_{fold}_best.csv"), index=False)
                #     result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_fold_{fold}_best.csv"), index=False)
                #
                # else:
                #     print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                #     print(f">>> current best score: {round(best_lb, 4)}")
                #
                # oof_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"], f"oof_df_fold_{fold}.csv"), index=False)
                # result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_fold_{fold}.csv"), index=False)
                #
                # # save pickle for analysis
                # result_df.to_pickle(os.path.join(cfg.outputs.model_dir, f"result_df_fold_{fold}.pkl"))

                # saving -----
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': current_iteration,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    # 'lb': lb,
                }

                if best_lb > save_trigger:
                    save_checkpoint(cfg_dict, model_state, is_best=is_best)

                # logging ----
                if cfg.use_wandb:
                    wandb.log({"lb": lb}, step=current_iteration)
                    wandb.log({"best_lb": best_lb}, step=current_iteration)

                    # ----
                    for k, v in result_dict.items():
                        if "df" not in k:
                            wandb.log({k: round(v, 4)}, step=current_iteration)

                    # --- log best scores dict
                    for k, v in best_dict.items():
                        if "df" not in k:
                            wandb.log({k: round(v, 4)}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()

                # ema ---
                if cfg.train_params.use_ema:
                    ema.restore()

                print_line()

                # early stopping ----
                if patience_tracker >= cfg_dict['train_params']['patience']:
                    print("stopping early")
                    model.eval()
                    return


if __name__ == "__main__":
    run_training()
