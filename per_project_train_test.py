import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import logging
import time
import helper.logger as logger
from helper.configure import Configure
import helper.utils as utils
from data_modules.dataset import PatchDataset
from data_modules.collator import PatchCollator
from torch.utils.data import DataLoader
from data_modules.data_loader import get_data_loaders
from models.patch_validator import PatchValidator
def perform_final_test_evaluation(cfg, model_to_eval, df_test_final, tokenizer, device,
                                  eval_output_dir_final, checkpoint_name_base):
    logger.info("***** Starting Final Test Set Evaluation *****")
    use_cuda = device.type == 'cuda'
    label_col_cfg = cfg.data.label_col
    project_col = cfg.data.get('project_col', 'project_name')
    baseline_original_time_col_name = cfg.data.get('time_col', 'time_consume')
    baseline_time_col_in_detailed_csv = 'rerun_time_consume'
    test_name_col_name = cfg.data.get('test_name_col', "full_test_name")
    source_file_col_name = cfg.data.get('source_col', "source_file")
    rerun_consistency_col = cfg.data.get('rerun_consistency_col', "rerun_consistency")
    incorrect_type_col = cfg.data.get('incorrect_type_col', "incorrect_type")
    generated_source_marker_col = cfg.data.get('generated_source_marker_col', "augmented_source_marker")
    model_prediction_time_col_name_in_csv = 'prediction_time_secs'
    is_mutation_col = cfg.data.get('is_mutation_col', 'is_mutation')
    mutation_iteration_col = cfg.data.get('mutation_iteration_col', 'mutation_iteration')
    has_project = project_col in df_test_final.columns
    has_baseline_time_orig = baseline_original_time_col_name in df_test_final.columns
    has_test_name = test_name_col_name in df_test_final.columns
    has_source_file = source_file_col_name in df_test_final.columns
    has_rerun_consistency = rerun_consistency_col in df_test_final.columns
    has_incorrect_type = incorrect_type_col in df_test_final.columns
    has_gen_source_marker = generated_source_marker_col in df_test_final.columns
    has_is_mutation = is_mutation_col in df_test_final.columns
    has_mutation_iteration = mutation_iteration_col in df_test_final.columns
    final_test_dataset = PatchDataset(df_test_final, tokenizer, cfg, stage='FINAL_TEST_PROJECT_SPLIT')
    final_test_loader = DataLoader(final_test_dataset, batch_size=cfg.eval.batch_size, shuffle=False,
                                     num_workers=cfg.train.device_setting.get('num_workers', 0),
                                     collate_fn=PatchCollator(tokenizer), pin_memory=True)
    model_to_eval.eval()
    preds_probs_list = []; true_lbls_from_batch_list = []; pred_times_sec_list = []
    if use_cuda: starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_inf_time = 0.0; total_proc = 0
    with torch.no_grad():
        for batch in final_test_loader:
            lbl_batch = batch['label'].to(device).float(); bs = lbl_batch.size(0)
            if use_cuda: starter.record()
            else: st = time.time()
            outs = model_to_eval(input_ids_flaky=batch['input_ids_flaky'].to(device), attention_mask_flaky=batch['attention_mask_flaky'].to(device),
                                 input_ids_patch=batch['input_ids_patch'].to(device), attention_mask_patch=batch['attention_mask_patch'].to(device))
            if use_cuda: ender.record(); torch.cuda.synchronize(); batch_dur_s = starter.elapsed_time(ender) / 1000.0
            else: et = time.time(); batch_dur_s = et - st
            total_inf_time += batch_dur_s; total_proc += bs
            time_sample_s = batch_dur_s / bs if bs > 0 else 0
            pr = torch.sigmoid(outs).squeeze(-1).cpu().numpy()
            preds_probs_list.extend(pr); true_lbls_from_batch_list.extend(lbl_batch.cpu().numpy().astype(int));
            pred_times_sec_list.extend([time_sample_s] * bs)
    if len(true_lbls_from_batch_list) != len(df_test_final):
        logger.error("Final Test Eval: Mismatch in result lengths. Aborting."); return
    true_labels_arr = df_test_final[label_col_cfg].values.astype(int)
    model_probs_arr = np.array(preds_probs_list)
    model_labels_thresh05_arr = (model_probs_arr >= 0.5).astype(int)
    avg_model_pred_time_ms = (total_inf_time / total_proc) * 1000 if total_proc > 0 else np.nan
    logger.info(f"Final Test - Model Prediction Timing: Avg time/sample: {avg_model_pred_time_ms:.4f} ms")
    metrics_model = utils.calculate_metrics(true_labels_arr, model_probs_arr, is_probs=True)
    if len(np.unique(true_labels_arr)) > 1:
        try: fpr_m,tpr_m,_=roc_curve(true_labels_arr,model_probs_arr); metrics_model['auc']=auc(fpr_m,tpr_m)
        except ValueError as e_roc_m: logger.warning(f"Model AUC calc error: {e_roc_m}"); metrics_model['auc']=np.nan
    else: metrics_model['auc']=np.nan; logger.warning("Model: Only one class in true labels, AUC is NaN.")
    metrics_model['avg_run_time_ms'] = avg_model_pred_time_ms
    metrics_baseline = {}
    baseline_labels_arr = None
    avg_baseline_time_ms = np.nan
    if has_rerun_consistency:
        baseline_labels_arr = df_test_final[rerun_consistency_col].values.astype(int)
        metrics_baseline = utils.calculate_metrics(true_labels_arr, baseline_labels_arr, is_probs=False)
        if len(np.unique(true_labels_arr)) > 1 and (baseline_labels_arr is not None and len(np.unique(baseline_labels_arr)) > 1):
            try: fpr_b,tpr_b,_=roc_curve(true_labels_arr,baseline_labels_arr); metrics_baseline['auc']=auc(fpr_b,tpr_b)
            except ValueError as e_roc_b: logger.warning(f"Baseline AUC calc error: {e_roc_b}"); metrics_baseline['auc']=np.nan
        else: metrics_baseline['auc']=np.nan; logger.warning("Baseline: Not enough classes for AUC, AUC is NaN.")
        if has_baseline_time_orig:
            baseline_times_s = pd.to_numeric(df_test_final[baseline_original_time_col_name], errors='coerce').dropna().values
            if len(baseline_times_s) > 0: avg_baseline_time_ms = np.mean(baseline_times_s) * 1000
        metrics_baseline['avg_run_time_ms'] = avg_baseline_time_ms
    logger.info("\n***** FINAL OVERALL METRICS (Project Split Test Set, Model Thr=0.5 vs Baseline) *****")
    header = f"{'Metric':<18} | {'Model':^18} | {'Baseline (Rerun)':^20}"
    logger.info(header); logger.info("-" * len(header))
    for mkey_log in ['accuracy','auc','avg_run_time_ms']:
        m_val_log = metrics_model.get(mkey_log,np.nan)
        b_val_log = metrics_baseline.get(mkey_log,np.nan) if metrics_baseline else np.nan
        unit = " ms" if mkey_log == 'avg_run_time_ms' else ""
        b_val_str = f"{b_val_log:.4f}{unit}" if pd.notna(b_val_log) else "N/A"
        logger.info(f"{mkey_log.replace('_',' ').capitalize():<18} | {m_val_log:>18.4f}{unit} | {b_val_str:>20}")
    for cls_k_log, cls_suf_log in [('positive','Positive'),('negative','Negative')]:
        for met_k_log, met_suf_log in [('precision','Precision'),('recall','Recall'),('f1','F1')]:
            m_val_log = metrics_model.get(cls_k_log,{}).get(met_k_log,np.nan)
            b_val_log = metrics_baseline.get(cls_k_log,{}).get(met_k_log,np.nan) if metrics_baseline else np.nan
            b_val_str = f"{b_val_log:.4f}" if pd.notna(b_val_log) else "N/A"
            logger.info(f"{f'{met_suf_log} ({cls_suf_log})':<18} | {m_val_log:>18.4f} | {b_val_str:>20}")
    logger.info("-" * len(header))
    if len(np.unique(true_labels_arr)) > 1:
        utils.plot_roc_curve(true_labels_arr, model_probs_arr, eval_output_dir_final,
                             filename=f"project_split_overall_roc_{checkpoint_name_base}.png",
                             extra_threshold_data=None)
    per_project_summary_data_list = []
    if has_project:
        logger.info("\n***** FINAL PER-PROJECT METRICS (Project Split Test Set) *****")
        temp_eval_df_for_grouping = df_test_final.copy()
        temp_eval_df_for_grouping['model_pred_prob_temp'] = model_probs_arr
        if baseline_labels_arr is not None:
            temp_eval_df_for_grouping['baseline_pred_label_temp'] = baseline_labels_arr
        if model_prediction_time_col_name_in_csv:
            temp_eval_df_for_grouping[model_prediction_time_col_name_in_csv] = np.array(pred_times_sec_list)
        for project_name_iter, group_df in temp_eval_df_for_grouping.groupby(project_col):
            proj_true_labels = group_df[label_col_cfg].values.astype(int)
            proj_model_probs = group_df['model_pred_prob_temp'].values
            num_gt_correct_patches = np.sum(proj_true_labels == 1)
            num_gt_incorrect_patches = np.sum(proj_true_labels == 0)
            proj_metrics_m = utils.calculate_metrics(proj_true_labels, proj_model_probs, is_probs=True)
            if len(np.unique(proj_true_labels)) > 1:
                try: fpr_pm,tpr_pm,_=roc_curve(proj_true_labels,proj_model_probs); proj_metrics_m['auc']=auc(fpr_pm,tpr_pm)
                except ValueError: proj_metrics_m['auc']=np.nan
            else: proj_metrics_m['auc']=np.nan
            avg_model_time_proj_ms_p = np.nan
            if model_prediction_time_col_name_in_csv in group_df.columns:
                proj_model_times_s = group_df[model_prediction_time_col_name_in_csv].dropna().values
                if len(proj_model_times_s) > 0: avg_model_time_proj_ms_p = np.mean(proj_model_times_s) * 1000
            proj_metrics_m['avg_run_time_ms'] = avg_model_time_proj_ms_p
            proj_metrics_b = {}
            avg_baseline_time_proj_ms_p = np.nan
            if 'baseline_pred_label_temp' in group_df.columns:
                proj_baseline_labels = group_df['baseline_pred_label_temp'].values
                proj_metrics_b = utils.calculate_metrics(proj_true_labels, proj_baseline_labels, is_probs=False)
                if len(np.unique(proj_true_labels)) > 1 and len(np.unique(proj_baseline_labels)) > 1:
                    try: fpr_pb,tpr_pb,_=roc_curve(proj_true_labels,proj_baseline_labels); proj_metrics_b['auc']=auc(fpr_pb,tpr_pb)
                    except ValueError: proj_metrics_b['auc']=np.nan
                else: proj_metrics_b['auc']=np.nan
                if baseline_original_time_col_name in group_df.columns:
                    proj_baseline_times_s = pd.to_numeric(group_df[baseline_original_time_col_name], errors='coerce').dropna().values
                    if len(proj_baseline_times_s) > 0: avg_baseline_time_proj_ms_p = np.mean(proj_baseline_times_s) * 1000
            proj_metrics_b['avg_run_time_ms'] = avg_baseline_time_proj_ms_p
            logger.info(f"  Project: {project_name_iter} (Samples: {len(proj_true_labels)}, GT Correct: {num_gt_correct_patches}, GT Incorrect: {num_gt_incorrect_patches})")
            logger.info(f"    Model   : Acc={proj_metrics_m.get('accuracy',np.nan):.4f}, AUC={proj_metrics_m.get('auc',np.nan):.4f}, P_Pos={proj_metrics_m.get('positive',{}).get('precision',np.nan):.4f}, R_Pos={proj_metrics_m.get('positive',{}).get('recall',np.nan):.4f}, F1_Pos={proj_metrics_m.get('positive',{}).get('f1',np.nan):.4f}, P_Neg={proj_metrics_m.get('negative',{}).get('precision',np.nan):.4f}, R_Neg={proj_metrics_m.get('negative',{}).get('recall',np.nan):.4f}, F1_Neg={proj_metrics_m.get('negative',{}).get('f1',np.nan):.4f}, AvgTime={proj_metrics_m.get('avg_run_time_ms',np.nan):.2f}ms")
            if proj_metrics_b: logger.info(f"    Baseline: Acc={proj_metrics_b.get('accuracy',np.nan):.4f}, AUC={proj_metrics_b.get('auc',np.nan):.4f}, P_Pos={proj_metrics_b.get('positive',{}).get('precision',np.nan):.4f}, R_Pos={proj_metrics_b.get('positive',{}).get('recall',np.nan):.4f}, F1_Pos={proj_metrics_b.get('positive',{}).get('f1',np.nan):.4f}, P_Neg={proj_metrics_b.get('negative',{}).get('precision',np.nan):.4f}, R_Neg={proj_metrics_b.get('negative',{}).get('recall',np.nan):.4f}, F1_Neg={proj_metrics_b.get('negative',{}).get('f1',np.nan):.4f}, AvgTime={proj_metrics_b.get('avg_run_time_ms',np.nan):.2f}ms")
            else: logger.info(f"    Baseline: N/A")
            per_project_summary_data_list.append({
                'project': project_name_iter, 'num_samples': len(proj_true_labels),
                'num_correct_patches': num_gt_correct_patches,
                'num_incorrect_patches': num_gt_incorrect_patches,
                'model_accuracy': proj_metrics_m.get('accuracy', np.nan), 'model_auc': proj_metrics_m.get('auc', np.nan),
                'model_precision_pos': proj_metrics_m.get('positive',{}).get('precision',np.nan), 'model_recall_pos': proj_metrics_m.get('positive',{}).get('recall',np.nan),'model_f1_pos': proj_metrics_m.get('positive',{}).get('f1',np.nan),
                'model_precision_neg': proj_metrics_m.get('negative',{}).get('precision',np.nan), 'model_recall_neg': proj_metrics_m.get('negative',{}).get('recall',np.nan),'model_f1_neg': proj_metrics_m.get('negative',{}).get('f1',np.nan),
                'model_avg_time_ms': proj_metrics_m.get('avg_run_time_ms',np.nan),
                'baseline_accuracy': proj_metrics_b.get('accuracy', np.nan), 'baseline_auc': proj_metrics_b.get('auc', np.nan),
                'baseline_precision_pos': proj_metrics_b.get('positive',{}).get('precision',np.nan), 'baseline_recall_pos': proj_metrics_b.get('positive',{}).get('recall',np.nan),'baseline_f1_pos': proj_metrics_b.get('positive',{}).get('f1',np.nan),
                'baseline_precision_neg': proj_metrics_b.get('negative',{}).get('precision',np.nan), 'baseline_recall_neg': proj_metrics_b.get('negative',{}).get('recall',np.nan),'baseline_f1_neg': proj_metrics_b.get('negative',{}).get('f1',np.nan),
                'baseline_avg_time_ms': proj_metrics_b.get('avg_run_time_ms',np.nan),
            })
    if per_project_summary_data_list:
        df_project_summary = pd.DataFrame(per_project_summary_data_list)
        project_summary_column_order = [
            'project',
            'num_samples',
            'num_gt_correct_patches',
            'num_gt_incorrect_patches',
            'model_accuracy', 'baseline_accuracy',
            'model_auc', 'baseline_auc',
            'model_precision_pos', 'baseline_precision_pos',
            'model_recall_pos', 'baseline_recall_pos',
            'model_f1_pos', 'baseline_f1_pos',
            'model_precision_neg', 'baseline_precision_neg',
            'model_recall_neg', 'baseline_recall_neg',
            'model_f1_neg', 'baseline_f1_neg',
            'model_avg_time_ms', 'baseline_avg_time_ms'
        ]
        ordered_summary_cols = [col for col in project_summary_column_order if col in df_project_summary.columns]
        df_project_summary = df_project_summary[ordered_summary_cols]
        project_summary_path = os.path.join(eval_output_dir_final, f"project_split_per_project_summary_{checkpoint_name_base}.csv")
        try:
            df_project_summary.to_csv(project_summary_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"Per-project summary saved: {project_summary_path}")
        except Exception as e: logger.error(f"Failed to save per-project summary: {e}")
    logger.info("\nPreparing detailed per-row evaluation results CSV for entire test set...")
    detailed_results_data_final = {label_col_cfg: true_labels_arr,
                                   'model_predicted_prob': model_probs_arr,
                                   'model_predicted_label_thresh0.5': model_labels_thresh05_arr,
                                   model_prediction_time_col_name_in_csv: np.array(pred_times_sec_list)}
    if has_project: detailed_results_data_final[project_col] = df_test_final[project_col].values
    if has_test_name: detailed_results_data_final[test_name_col_name] = df_test_final[test_name_col_name].values
    if has_source_file: detailed_results_data_final[source_file_col_name] = df_test_final[source_file_col_name].values
    if has_is_mutation: detailed_results_data_final[is_mutation_col] = df_test_final[is_mutation_col].values
    if has_mutation_iteration: detailed_results_data_final[mutation_iteration_col] = df_test_final[mutation_iteration_col].values
    if has_baseline_time_orig: detailed_results_data_final[baseline_time_col_in_detailed_csv] = df_test_final[baseline_original_time_col_name].values
    if has_rerun_consistency: detailed_results_data_final[rerun_consistency_col] = df_test_final[rerun_consistency_col].values
    if has_incorrect_type: detailed_results_data_final[incorrect_type_col] = df_test_final[incorrect_type_col].values
    if has_gen_source_marker: detailed_results_data_final[generated_source_marker_col] = df_test_final[generated_source_marker_col].values
    results_df_detailed_final = pd.DataFrame(detailed_results_data_final)
    detailed_column_order_final = [project_col, test_name_col_name, source_file_col_name, generated_source_marker_col,
                                 label_col_cfg, incorrect_type_col,
                                 is_mutation_col, mutation_iteration_col,
                                 'model_predicted_prob', 'model_predicted_label_thresh0.5',
                                 rerun_consistency_col, model_prediction_time_col_name_in_csv,
                                 baseline_time_col_in_detailed_csv]
    present_cols_final_ordered = [col for col in detailed_column_order_final if col in results_df_detailed_final.columns]
    remaining_cols_final = [col for col in results_df_detailed_final.columns if col not in present_cols_final_ordered]
    results_df_detailed_final = results_df_detailed_final[present_cols_final_ordered + remaining_cols_final]
    detailed_csv_path_final = os.path.join(eval_output_dir_final, f"project_split_overall_details_{checkpoint_name_base}.csv")
    try: results_df_detailed_final.to_csv(detailed_csv_path_final, index=False, encoding='utf-8', float_format='%.6f', na_rep='NaN'); logger.info(f"Overall detailed results saved: {detailed_csv_path_final}")
    except Exception as e: logger.error(f"Failed to save overall detailed results: {e}")
    logger.info("***** Final Test Set Evaluation Finished *****")
def main_project_split_workflow(overall_cfg):
    device_ids_config = overall_cfg.train.device_setting.get('device_ids', "0")
    dp_device_ids = []
    if isinstance(device_ids_config, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids_config))
        if torch.cuda.is_available() and device_ids_config: dp_device_ids = list(range(len(device_ids_config))); device_str = f"cuda:{dp_device_ids[0]}" if dp_device_ids else "cuda"
        else: device_str = "cpu"
    elif isinstance(device_ids_config, (str, int)):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids_config); parsed_ids_str = [s.strip() for s in str(device_ids_config).split(',') if s.strip().isdigit()]
        if torch.cuda.is_available() and all(s.isdigit() for s in parsed_ids_str) and parsed_ids_str: dp_device_ids = list(range(len(parsed_ids_str))); device_str = "cuda"
        else: device_str = "cpu"
    else: logger.warning("device_ids format unrecognized."); device_str = "cuda" if torch.cuda.is_available() else "cpu"; dp_device_ids = [0] if torch.cuda.is_available() and torch.cuda.device_count() > 0 else []
    device = torch.device(device_str if device_str != "cpu" and torch.cuda.is_available() else "cpu")
    num_gpus_for_dp = len(dp_device_ids) if device.type == 'cuda' else 0
    logger.info(f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','N/A')}'. Main device: {device}. Num GPUs for DP: {num_gpus_for_dp}")
    logger.info("Loading full dataset...")
    data_path = os.path.join(overall_cfg.data.data_dir, overall_cfg.data.combined_csv_file)
    try: df_full = pd.read_csv(data_path); logger.info(f"Loaded full dataset: {data_path}, Shape: {df_full.shape}")
    except Exception as e: logger.error(f"Failed to load full dataset: {e}"); return
    df_train, df_val, df_test = utils.split_data_for_project_eval(df_full, overall_cfg)
    if df_train is None : logger.error("Data splitting failed (train is None). Exiting."); return
    if df_train.empty: logger.error("Training data empty. Exiting."); return
    output_dir_train = overall_cfg.train.output_dir
    os.makedirs(output_dir_train, exist_ok=True)
    df_train.to_csv(os.path.join(output_dir_train, "train_split_used.csv"), index=False)
    if df_val is not None and not df_val.empty: df_val.to_csv(os.path.join(output_dir_train, "val_split_used.csv"), index=False)
    if df_test is not None and not df_test.empty: df_test.to_csv(os.path.join(output_dir_train, "test_split_used.csv"), index=False)
    logger.info(f"Project-based data splits saved to {output_dir_train}")
    try: tokenizer = AutoTokenizer.from_pretrained(overall_cfg.model.encoder_name)
    except Exception as e: logger.error(f"Tokenizer fail: {e}"); return
    train_loader, val_loader, test_loader_for_final_eval = get_data_loaders(overall_cfg, tokenizer, df_train, df_val if df_val is not None else pd.DataFrame(), df_test if df_test is not None else pd.DataFrame())
    if train_loader is None: logger.error("Train loader None. Exit."); return
    model = PatchValidator(overall_cfg)
    if device.type == 'cuda' and num_gpus_for_dp > 1: model = nn.DataParallel(model, device_ids=dp_device_ids)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=overall_cfg.train.learning_rate, weight_decay=overall_cfg.train.weight_decay)
    label_col_cfg_train = overall_cfg.data.label_col
    criterion = None
    if overall_cfg.train.get('use_weighted_loss', False):
        pos_mult = overall_cfg.train.get('pos_weight_multiplier', 1.0)
        counts = df_train[label_col_cfg_train].value_counts(); n_neg=counts.get(0,0); n_pos=counts.get(1,0)
        if n_pos > 0 and n_neg > 0: base_w = n_neg/n_pos; final_w = base_w * pos_mult; criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([final_w]).to(device))
        else: criterion = nn.BCEWithLogitsLoss()
    else: criterion = nn.BCEWithLogitsLoss()
    scheduler = None; num_steps = len(train_loader) * overall_cfg.train.epochs
    if overall_cfg.train.get('lr_scheduler'):
        s_type = overall_cfg.train.lr_scheduler.lower(); ratio = overall_cfg.train.get('warmup_steps_ratio',0.0); wu_steps = int(num_steps*ratio)
        if s_type == 'linear': scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=wu_steps,num_training_steps=num_steps)
        elif s_type == 'cosine': scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=wu_steps,num_training_steps=num_steps)
        if scheduler: logger.info(f"Scheduler: {s_type.upper()} ({wu_steps} warmup / {num_steps} total).")
    best_val_met = -float('inf'); epochs_no_imp = 0; best_ep = 0
    patience = overall_cfg.train.get('early_stopping_patience', 0)
    early_stop_enabled = patience > 0 and val_loader is not None
    if early_stop_enabled: logger.info(f"Early stopping: patience={patience}")
    comp_epochs = 0
    for ep in range(overall_cfg.train.epochs):
        comp_epochs = ep + 1; model.train(); total_loss_ep = 0.0; steps_in_ep = 0
        for batch in train_loader:
            labels_b = batch['label'].to(device).float()
            outputs_b = model(input_ids_flaky=batch['input_ids_flaky'].to(device), attention_mask_flaky=batch['attention_mask_flaky'].to(device),
                              input_ids_patch=batch['input_ids_patch'].to(device), attention_mask_patch=batch['attention_mask_patch'].to(device))
            loss_b = criterion(outputs_b.squeeze(-1), labels_b)
            optimizer.zero_grad(); loss_b.backward(); optimizer.step()
            if scheduler: scheduler.step()
            total_loss_ep += loss_b.item(); steps_in_ep += 1
        avg_loss_ep = total_loss_ep / steps_in_ep if steps_in_ep > 0 else np.nan
        logger.info(f"End Ep {comp_epochs}: TrainLoss={avg_loss_ep:.4f}")
        if val_loader and overall_cfg.train.eval_strategy == "epoch":
            model.eval(); val_probs = []; val_lbls = []
            with torch.no_grad():
                for batch_v in val_loader:
                    lbls_v = batch_v['label'].to(device).float()
                    outs_v = model(input_ids_flaky=batch_v['input_ids_flaky'].to(device), attention_mask_flaky=batch_v['attention_mask_flaky'].to(device),
                                   input_ids_patch=batch_v['input_ids_patch'].to(device), attention_mask_patch=batch_v['attention_mask_patch'].to(device))
                    probs_v = torch.sigmoid(outs_v).squeeze(-1).cpu().numpy(); val_probs.extend(probs_v); val_lbls.extend(lbls_v.cpu().numpy().astype(int))
            if not val_lbls: logger.warning(f"Val Ep {comp_epochs}: No val preds. Skip metrics."); continue
            val_mets = utils.calculate_metrics(val_lbls, val_probs, is_probs=True)
            f1p, f1n = val_mets['positive']['f1'], val_mets['negative']['f1']
            cur_met = (f1p + f1n) / 2.0; met_name = "AvgF1_Val"
            logger.info(f"Val Ep {comp_epochs}: Acc={val_mets['accuracy']:.4f}, {met_name}={cur_met:.4f} (PosF1:{f1p:.4f}, NegF1:{f1n:.4f})")
            if early_stop_enabled:
                if cur_met > best_val_met:
                    best_val_met = cur_met; best_ep = comp_epochs; epochs_no_imp = 0
                    best_mdl_path = os.path.join(output_dir_train, f"best_model_project_split_ep{best_ep}.pt")
                    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, best_mdl_path); logger.info(f"Best val model saved: {best_mdl_path}")
                else: epochs_no_imp += 1
                if epochs_no_imp >= patience: logger.info(f"Early stopping. Best ep {best_ep}"); break
    final_model_to_eval_path = ""
    ckpt_base_name_eval = ""
    state_to_save_final_ep = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    last_ep_model_path = os.path.join(output_dir_train, f"last_epoch_model_project_split_ep{comp_epochs}.pt")
    torch.save(state_to_save_final_ep, last_ep_model_path)
    if early_stop_enabled and best_ep > 0 and os.path.exists(os.path.join(output_dir_train, f"best_model_project_split_ep{best_ep}.pt")):
        final_model_to_eval_path = os.path.join(output_dir_train, f"best_model_project_split_ep{best_ep}.pt")
        ckpt_base_name_eval = f"best_ep{best_ep}_project_split"
    else:
        final_model_to_eval_path = last_ep_model_path
        ckpt_base_name_eval = f"last_ep{comp_epochs}_project_split"
    logger.info(f"Using model from {final_model_to_eval_path} for final test evaluation.")
    if df_test is not None and not df_test.empty and test_loader_for_final_eval is not None:
        model_final_eval = PatchValidator(overall_cfg)
        model_final_eval.load_state_dict(torch.load(final_model_to_eval_path, map_location=device))
        if device.type == 'cuda' and num_gpus_for_dp > 1:
            model_final_eval = nn.DataParallel(model_final_eval, device_ids=dp_device_ids)
        model_final_eval.to(device)
        perform_final_test_evaluation(overall_cfg, model_final_eval, df_test, tokenizer, device,
                                      output_dir_train,
                                      checkpoint_name_base=ckpt_base_name_eval)
    else:
        logger.warning("Test data is empty or test loader not created. Final evaluation skipped.")
    logger.info("Overall project-split training and evaluation process finished.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flaky Patch Validator Model with Project-Based Split")
    parser.add_argument('--config_file', type=str, default='config/default_config.json', help='Path to config JSON')
    args_cli = parser.parse_args()
    try:
        cfg_main = Configure(config_json_file=args_cli.config_file)
    except FileNotFoundError: print(f"FATAL: Config file not found: {args_cli.config_file}"); exit(1)
    except Exception as e_cfg: print(f"FATAL: Error loading config {args_cli.config_file}: {e_cfg}"); exit(1)
    main_project_split_workflow(cfg_main)