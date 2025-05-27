import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import json
from sklearn.model_selection import KFold, train_test_split as sklearn_train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import logging
import helper.logger as logger
from helper.configure import Configure
import helper.utils as utils
from data_modules.dataset import PatchDataset
from data_modules.collator import PatchCollator
from torch.utils.data import DataLoader
from data_modules.data_loader import get_data_loaders
from models.patch_validator import PatchValidator
def evaluate_and_save_fold_results(cfg, model, df_fold_test, tokenizer, device,
                                   fold_output_dir, fold_num,
                                   checkpoint_name_base="model_for_fold"):
    logger.info(f"[Fold {fold_num}] ***** Starting Evaluation on Test Fold *****")
    use_cuda = device.type == 'cuda'
    label_col_from_cfg = cfg.data.label_col
    project_col = cfg.data.get('project_col', 'project_name')
    baseline_original_time_col_name = cfg.data.get('time_col', 'time_consume')
    baseline_time_col_in_detailed_csv = 'rerun_time_consume'
    test_name_col_name = cfg.data.get('test_name_col', "full_test_name")
    source_file_col_name = cfg.data.get('source_col', "source_file")
    rerun_consistency_col_name = cfg.data.get('rerun_consistency_col', "rerun_consistency")
    incorrect_type_col_name = cfg.data.get('incorrect_type_col', "incorrect_type")
    generated_source_marker_col = cfg.data.get('generated_source_marker_col', "augmented_source_marker")
    model_prediction_time_col_name_in_csv = 'prediction_time_secs'
    can_add_project = project_col in df_fold_test.columns
    can_add_orig_baseline_time_from_df = baseline_original_time_col_name in df_fold_test.columns
    can_add_test_name = test_name_col_name in df_fold_test.columns
    can_add_source_file = source_file_col_name in df_fold_test.columns
    can_add_rerun_consistency = rerun_consistency_col_name in df_fold_test.columns
    can_analyze_incorrect_type = incorrect_type_col_name in df_fold_test.columns
    has_source_marker = generated_source_marker_col in df_fold_test.columns
    fold_test_dataset = PatchDataset(df_fold_test, tokenizer, cfg, stage=f'Fold{fold_num}_TEST')
    fold_test_loader = DataLoader(fold_test_dataset, batch_size=cfg.eval.batch_size, shuffle=False,
                                  num_workers=cfg.train.device_setting.get('num_workers', 0),
                                  collate_fn=PatchCollator(tokenizer), pin_memory=True)
    model.eval()
    test_preds_probs_fold = []; test_true_labels_fold_from_batch = []; prediction_times_secs_fold = []
    if use_cuda: starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_inference_time_fold = 0.0; total_samples_processed_fold = 0
    with torch.no_grad():
        for batch in fold_test_loader:
            input_ids_flaky = batch['input_ids_flaky'].to(device); attention_mask_flaky = batch['attention_mask_flaky'].to(device)
            input_ids_patch = batch['input_ids_patch'].to(device); attention_mask_patch = batch['attention_mask_patch'].to(device)
            labels_in_batch = batch['label'].to(device).float()
            current_batch_size = labels_in_batch.size(0)
            if use_cuda: starter.record()
            else: start_time = time.time()
            outputs = model(input_ids_flaky=input_ids_flaky, attention_mask_flaky=attention_mask_flaky, input_ids_patch=input_ids_patch, attention_mask_patch=attention_mask_patch)
            if use_cuda: ender.record(); torch.cuda.synchronize(); batch_duration_secs = starter.elapsed_time(ender) / 1000.0
            else: end_time = time.time(); batch_duration_secs = end_time - start_time
            total_inference_time_fold += batch_duration_secs; total_samples_processed_fold += current_batch_size
            time_per_sample_sec = batch_duration_secs / current_batch_size if current_batch_size > 0 else 0
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            test_preds_probs_fold.extend(probs)
            test_true_labels_fold_from_batch.extend(labels_in_batch.cpu().numpy().astype(int))
            prediction_times_secs_fold.extend([time_per_sample_sec] * current_batch_size)
    logger.info(f"[Fold {fold_num}] Evaluation Loop Finished on test data.")
    current_fold_results_for_aggregation = {
        'fold_num': fold_num, 'model_metrics': {}, 'baseline_metrics': {},
        'incorrect_type_detection': [],
        'true_labels_for_overall_roc': [], 'model_probs_for_overall_roc': []
    }
    if len(test_true_labels_fold_from_batch) != len(df_fold_test):
        logger.error(f"[Fold {fold_num}] Mismatch results ({len(test_true_labels_fold_from_batch)}) vs test DF ({len(df_fold_test)})."); return None
    true_labels_all_fold_df = df_fold_test[label_col_from_cfg].values.astype(int)
    model_pred_probs_all_fold = np.array(test_preds_probs_fold)
    model_pred_labels_all_thresh05_fold = (model_pred_probs_all_fold >= 0.5).astype(int)
    current_fold_results_for_aggregation['true_labels_for_overall_roc'] = true_labels_all_fold_df.tolist()
    current_fold_results_for_aggregation['model_probs_for_overall_roc'] = model_pred_probs_all_fold.tolist()
    avg_model_pred_time_ms_fold = np.nan
    if total_samples_processed_fold > 0:
        avg_model_pred_time_ms_fold = (total_inference_time_fold / total_samples_processed_fold) * 1000
        logger.info(f"[Fold {fold_num}] Model Prediction Timing: Avg time/sample: {avg_model_pred_time_ms_fold:.4f} ms")
    metrics_model_fold = utils.calculate_metrics(true_labels_all_fold_df, model_pred_probs_all_fold, is_probs=True)
    if len(np.unique(true_labels_all_fold_df)) > 1:
        try:
            fpr, tpr, _ = roc_curve(true_labels_all_fold_df, model_pred_probs_all_fold)
            metrics_model_fold['auc'] = auc(fpr, tpr) if not (np.isnan(fpr).any() or np.isnan(tpr).all()) else np.nan
        except ValueError as e_roc:
            logger.warning(f"[Fold {fold_num}] Model ROC/AUC calculation error: {e_roc}")
            metrics_model_fold['auc'] = np.nan
    else:
        logger.warning(f"[Fold {fold_num}] Model: Only one class in true labels for fold. AUC is NaN.")
        metrics_model_fold['auc'] = np.nan
    metrics_model_fold['avg_run_time_ms'] = avg_model_pred_time_ms_fold
    current_fold_results_for_aggregation['model_metrics'] = metrics_model_fold
    baseline_pred_labels_all_fold = None
    metrics_baseline_fold = {}
    avg_baseline_rerun_time_ms_fold = np.nan
    if can_add_rerun_consistency:
        baseline_pred_labels_all_fold = df_fold_test[rerun_consistency_col_name].values.astype(int)
        metrics_baseline_fold = utils.calculate_metrics(true_labels_all_fold_df, baseline_pred_labels_all_fold, is_probs=False)
        if len(np.unique(true_labels_all_fold_df)) > 1 and len(np.unique(baseline_pred_labels_all_fold)) > 1 :
            try:
                fpr_b, tpr_b, _ = roc_curve(true_labels_all_fold_df, baseline_pred_labels_all_fold)
                metrics_baseline_fold['auc'] = auc(fpr_b, tpr_b) if not (np.isnan(fpr_b).all() or np.isnan(tpr_b).all()) else np.nan
            except ValueError as e_roc_b:
                logger.warning(f"[Fold {fold_num}] Baseline ROC/AUC calculation error: {e_roc_b}")
                metrics_baseline_fold['auc'] = np.nan
        else:
            logger.warning(f"[Fold {fold_num}] Baseline: Not enough classes for AUC. AUC is NaN.")
            metrics_baseline_fold['auc'] = np.nan
        if can_add_orig_baseline_time_from_df and baseline_original_time_col_name in df_fold_test.columns:
            baseline_times_raw_this_fold = pd.to_numeric(df_fold_test[baseline_original_time_col_name], errors='coerce').dropna().values
            if len(baseline_times_raw_this_fold) > 0:
                avg_b_time_fold_raw_this_fold = np.mean(baseline_times_raw_this_fold)
                avg_baseline_rerun_time_ms_fold = avg_b_time_fold_raw_this_fold * 1000
        metrics_baseline_fold['avg_run_time_ms'] = avg_baseline_rerun_time_ms_fold
    current_fold_results_for_aggregation['baseline_metrics'] = metrics_baseline_fold
    logger.info(f"[Fold {fold_num}] Test Metrics (Model): Acc={metrics_model_fold.get('accuracy',np.nan):.4f}, AUC={metrics_model_fold.get('auc',np.nan):.4f}, F1_Neg={metrics_model_fold.get('negative',{}).get('f1',np.nan):.4f}, AvgTime={metrics_model_fold.get('avg_run_time_ms',np.nan):.2f}ms")
    if baseline_pred_labels_all_fold is not None and metrics_baseline_fold:
        logger.info(f"[Fold {fold_num}] Test Metrics (Baseline): Acc={metrics_baseline_fold.get('accuracy',np.nan):.4f}, AUC={metrics_baseline_fold.get('auc',np.nan):.4f}, F1_Neg={metrics_baseline_fold.get('negative',{}).get('f1',np.nan):.4f}, AvgTime={metrics_baseline_fold.get('avg_run_time_ms',np.nan):.2f}ms")
    incorrect_type_detection_fold_data_list = []
    unique_incorrect_types_actually_analyzed = []
    if can_analyze_incorrect_type:
        analysis_mask = np.ones(len(df_fold_test), dtype=bool)
        if has_source_marker: analysis_mask = df_fold_test[generated_source_marker_col] != 'groundtruth_copy'
        true_labels_for_analysis = true_labels_all_fold_df[analysis_mask]
        model_preds_for_analysis = model_pred_labels_all_thresh05_fold[analysis_mask]
        incorrect_types_values = df_fold_test[incorrect_type_col_name].values[analysis_mask]
        baseline_preds_for_analysis = baseline_pred_labels_all_fold[analysis_mask] if baseline_pred_labels_all_fold is not None else None
        actual_negative_incorrect_types = incorrect_types_values[true_labels_for_analysis == 0]
        unique_incorrect_types_found_in_fold = sorted(pd.Series(actual_negative_incorrect_types).dropna().unique())
        logger.info(f"[Fold {fold_num}] Unique incorrect types for analysis: {unique_incorrect_types_found_in_fold}")
        for type_val_float in unique_incorrect_types_found_in_fold:
            if pd.isna(type_val_float): continue
            type_val_int = int(type_val_float)
            unique_incorrect_types_actually_analyzed.append(type_val_int)
            actual_mask = (true_labels_for_analysis == 0) & (incorrect_types_values == type_val_int)
            total_actual = np.sum(actual_mask); m_detected, m_recall, b_detected, b_recall = 0, np.nan, 0, np.nan
            if total_actual > 0:
                m_detected = np.sum(actual_mask & (model_preds_for_analysis == 0)); m_recall = m_detected / total_actual
                if baseline_preds_for_analysis is not None:
                    b_detected = np.sum(actual_mask & (baseline_preds_for_analysis == 0)); b_recall = b_detected / total_actual
            incorrect_type_detection_fold_data_list.append({'incorrect_type': type_val_int, 'total_actual': total_actual, 'model_detected': m_detected, 'model_recall': m_recall, 'baseline_detected': b_detected, 'baseline_recall': b_recall})
    current_fold_results_for_aggregation['incorrect_type_detection'] = incorrect_type_detection_fold_data_list
    detailed_results_data = {label_col_from_cfg: true_labels_all_fold_df,
                             'model_predicted_prob': model_pred_probs_all_fold,
                             'model_predicted_label_thresh0.5': model_pred_labels_all_thresh05_fold,
                             model_prediction_time_col_name_in_csv: np.array(prediction_times_secs_fold)}
    if can_add_project: detailed_results_data[project_col] = df_fold_test[project_col].values
    if can_add_test_name: detailed_results_data[test_name_col_name] = df_fold_test[test_name_col_name].values
    if can_add_source_file: detailed_results_data[source_file_col_name] = df_fold_test[source_file_col_name].values
    if can_add_orig_baseline_time_from_df: detailed_results_data[baseline_time_col_in_detailed_csv] = df_fold_test[baseline_original_time_col_name].values
    if can_add_rerun_consistency: detailed_results_data[rerun_consistency_col_name] = df_fold_test[rerun_consistency_col_name].values
    if can_analyze_incorrect_type: detailed_results_data[incorrect_type_col_name] = df_fold_test[incorrect_type_col_name].values
    if has_source_marker: detailed_results_data[generated_source_marker_col] = df_fold_test[generated_source_marker_col].values
    results_df_detailed_fold = pd.DataFrame(detailed_results_data)
    detailed_column_order = [project_col, test_name_col_name, source_file_col_name, generated_source_marker_col,
                             label_col_from_cfg, incorrect_type_col_name, 'model_predicted_prob',
                             'model_predicted_label_thresh0.5', rerun_consistency_col_name,
                             model_prediction_time_col_name_in_csv, baseline_time_col_in_detailed_csv]
    present_cols_ordered = [col for col in detailed_column_order if col in results_df_detailed_fold.columns]
    remaining_cols = [col for col in results_df_detailed_fold.columns if col not in present_cols_ordered]
    results_df_detailed_fold = results_df_detailed_fold[present_cols_ordered + remaining_cols]
    detailed_csv_path_fold = os.path.join(fold_output_dir, f"fold_{fold_num}_evaluation_details_{checkpoint_name_base}.csv")
    try: results_df_detailed_fold.to_csv(detailed_csv_path_fold, index=False, encoding='utf-8', float_format='%.6f', na_rep='NaN'); logger.info(f"[Fold {fold_num}] Detailed results saved: {detailed_csv_path_fold}")
    except Exception as e: logger.error(f"[Fold {fold_num}] Failed to save detailed results: {e}")
    fold_summary_list_of_dicts = []
    model_fold_summary_dict = {'Method': 'Model (Threshold 0.5)'}
    for k_m, v_m in metrics_model_fold.items():
        if k_m == 'avg_run_time_ms': model_fold_summary_dict['Avg_Run_Time_Ms'] = v_m
        elif isinstance(v_m, (int, float)): model_fold_summary_dict[k_m.capitalize()] = v_m
        elif isinstance(v_m, dict):
            for m_name_m, m_val_m in v_m.items(): model_fold_summary_dict[f"{m_name_m.capitalize()}_{k_m.capitalize()}"] = m_val_m
    fold_summary_list_of_dicts.append(model_fold_summary_dict)
    if metrics_baseline_fold:
        baseline_fold_summary_dict = {'Method': 'Baseline (Rerun)'}
        for k_b, v_b in metrics_baseline_fold.items():
            if k_b == 'avg_run_time_ms': baseline_fold_summary_dict['Avg_Run_Time_Ms'] = v_b
            elif isinstance(v_b, (int, float)): baseline_fold_summary_dict[k_b.capitalize()] = v_b
            elif isinstance(v_b, dict):
                for m_name_b, m_val_b in v_b.items(): baseline_fold_summary_dict[f"{m_name_b.capitalize()}_{k_b.capitalize()}"] = m_val_b
        fold_summary_list_of_dicts.append(baseline_fold_summary_dict)
    type_specific_summary_cols_for_fold_header = []
    for type_val_int in unique_incorrect_types_actually_analyzed:
        total_col_name = f"Total_Actual_IncorrectType_{type_val_int}"
        detected_col_name = f"Detected_IncorrectType_{type_val_int}"
        recall_col_name = f"Recall_IncorrectType_{type_val_int}"
        type_specific_summary_cols_for_fold_header.extend([total_col_name, detected_col_name, recall_col_name])
        current_type_item = next((item for item in incorrect_type_detection_fold_data_list if item['incorrect_type'] == type_val_int), None)
        if current_type_item:
            fold_summary_list_of_dicts[0][total_col_name] = current_type_item['total_actual']
            fold_summary_list_of_dicts[0][detected_col_name] = current_type_item['model_detected']
            fold_summary_list_of_dicts[0][recall_col_name] = current_type_item['model_recall']
            if len(fold_summary_list_of_dicts) > 1:
                fold_summary_list_of_dicts[1][total_col_name] = current_type_item['total_actual']
                fold_summary_list_of_dicts[1][detected_col_name] = current_type_item['baseline_detected']
                fold_summary_list_of_dicts[1][recall_col_name] = current_type_item['baseline_recall']
        else:
            fold_summary_list_of_dicts[0][total_col_name] = 0
            fold_summary_list_of_dicts[0][detected_col_name] = 0
            fold_summary_list_of_dicts[0][recall_col_name] = np.nan
            if len(fold_summary_list_of_dicts) > 1:
                fold_summary_list_of_dicts[1][total_col_name] = 0
                fold_summary_list_of_dicts[1][detected_col_name] = 0
                fold_summary_list_of_dicts[1][recall_col_name] = np.nan
    fold_summary_df = pd.DataFrame(fold_summary_list_of_dicts)
    summary_base_cols_fold = ['Method', 'Accuracy', 'AUC', 'Avg_Run_Time_Ms',
                              'Precision_Positive', 'Recall_Positive', 'F1_Positive',
                              'Precision_Negative', 'Recall_Negative', 'F1_Negative']
    final_summary_cols_fold = summary_base_cols_fold + sorted(list(set(type_specific_summary_cols_for_fold_header)))
    for col_s_f in final_summary_cols_fold:
        if col_s_f not in fold_summary_df.columns:
            fold_summary_df[col_s_f] = np.nan
    fold_summary_df = fold_summary_df.reindex(columns=final_summary_cols_fold)
    fold_summary_csv_path = os.path.join(fold_output_dir, f"fold_{fold_num}_summary_metrics_{checkpoint_name_base}.csv")
    try: fold_summary_df.to_csv(fold_summary_csv_path, index=False, float_format='%.4f', na_rep='NaN'); logger.info(f"[Fold {fold_num}] Summary metrics saved: {fold_summary_csv_path}")
    except Exception as e: logger.error(f"[Fold {fold_num}] Failed to save summary: {e}")
    return current_fold_results_for_aggregation
def train_and_evaluate_kfold(overall_cfg):
    device_ids_config = overall_cfg.train.device_setting.get('device_ids', "0")
    dp_device_ids = []
    if isinstance(device_ids_config, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids_config))
        if torch.cuda.is_available() and device_ids_config:
            dp_device_ids = list(range(len(device_ids_config)))
            device_str = f"cuda:{dp_device_ids[0]}" if dp_device_ids else "cuda"
        else: device_str = "cpu"
    elif isinstance(device_ids_config, (str, int)):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids_config)
        parsed_ids_str = [s.strip() for s in str(device_ids_config).split(',') if s.strip().isdigit()]
        if torch.cuda.is_available() and all(s.isdigit() for s in parsed_ids_str) and parsed_ids_str:
            dp_device_ids = list(range(len(parsed_ids_str)))
            device_str = "cuda"
        else: device_str = "cpu"
    else:
        logger.warning("device_ids format unrecognized. Using default."); device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available(): dp_device_ids = [0] if torch.cuda.device_count() > 0 else []
    device = torch.device(device_str if device_str != "cpu" and torch.cuda.is_available() else "cpu")
    num_gpus_for_dp = len(dp_device_ids) if device.type == 'cuda' else 0
    logger.info(f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','N/A')}'. Main device: {device}. Num GPUs for DP: {num_gpus_for_dp}")
    logger.info("Loading full dataset for K-Fold cross-validation...")
    data_path = os.path.join(overall_cfg.data.data_dir, overall_cfg.data.combined_csv_file)
    try: df_full = pd.read_csv(data_path); logger.info(f"Loaded full dataset: {data_path}, Shape: {df_full.shape}")
    except Exception as e: logger.error(f"Failed to load full dataset: {e}"); return
    source_file_col = overall_cfg.data.source_file_col
    rerun_consistency_col = overall_cfg.data.rerun_consistency_col
    label_col_cfg = overall_cfg.data.label_col
    if source_file_col not in df_full.columns: logger.error(f"Source file col '{source_file_col}' not in dataset."); return
    if rerun_consistency_col not in df_full.columns: logger.warning(f"Rerun consistency col '{rerun_consistency_col}' not in dataset.");
    special_data_condition = (df_full[source_file_col] == "flaky_fix_181_gpt35_labeled.csv") & (df_full[rerun_consistency_col].isna())
    df_special_train_fixed = df_full[special_data_condition]
    df_remaining = df_full[~special_data_condition]
    logger.info(f"Isolated {len(df_special_train_fixed)} special samples. {len(df_remaining)} remaining for K-Fold.")
    if df_remaining.empty and df_special_train_fixed.empty : logger.error("No data at all. Exiting."); return
    n_folds = overall_cfg.data.get('n_folds', 5)
    split_data_source = df_remaining if not df_remaining.empty else df_special_train_fixed
    if len(split_data_source) < n_folds and n_folds > 1:
        logger.warning(f"Adjusting n_folds to {len(split_data_source)} due to insufficient samples.")
        n_folds = len(split_data_source)
        if n_folds < 1 : logger.error("Not enough data for even 1 fold after adjustment for KFold."); return
    if split_data_source.empty: logger.error("Data source for KFold splitting is empty. Cannot proceed."); return
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=overall_cfg.seed)
    all_folds_evaluation_summaries = []
    overall_true_labels_all_folds = []
    overall_model_probs_all_folds = []
    base_output_dir = overall_cfg.train.output_dir_base
    os.makedirs(base_output_dir, exist_ok=True)
    for fold_num, (train_val_idx, test_idx) in enumerate(kf.split(split_data_source)):
        fold_num_display = fold_num + 1
        logger.info(f"\n===== Starting Fold {fold_num_display}/{n_folds} =====")
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold_num_display}")
        os.makedirs(fold_output_dir, exist_ok=True)
        df_fold_train_val_main = split_data_source.iloc[train_val_idx]
        df_fold_test = split_data_source.iloc[test_idx]
        df_fold_train_temp, df_fold_val = pd.DataFrame(), pd.DataFrame()
        if not df_fold_train_val_main.empty:
            can_stratify = False
            if label_col_cfg in df_fold_train_val_main:
                counts = df_fold_train_val_main[label_col_cfg].value_counts()
                if len(counts) > 1 and all(c >= 2 for c in counts): can_stratify = True
            current_val_size = 0.2
            if len(df_fold_train_val_main) < 5 and len(df_fold_train_val_main) > 1 :
                current_val_size = 1 / len(df_fold_train_val_main)
            if current_val_size > 0 and len(df_fold_train_val_main) > 1 :
                 df_fold_train_temp, df_fold_val = sklearn_train_test_split(
                    df_fold_train_val_main, test_size=current_val_size, random_state=overall_cfg.seed,
                    stratify=df_fold_train_val_main[label_col_cfg] if can_stratify else None )
            else: df_fold_train_temp = df_fold_train_val_main.copy()
        df_fold_train = pd.concat([df_fold_train_temp, df_special_train_fixed], ignore_index=True) if not df_special_train_fixed.empty else df_fold_train_temp
        if df_fold_train.empty :
            logger.error(f"[F{fold_num_display}] Training data empty. Skip.");
            all_folds_evaluation_summaries.append({'fold_num': fold_num_display, 'model_metrics': {}, 'baseline_metrics': {}, 'incorrect_type_detection': [], 'true_labels_for_overall_roc': [], 'model_probs_for_overall_roc': []})
            continue
        logger.info(f"[F{fold_num_display}] Sizes: Tr={len(df_fold_train)}, Vl={len(df_fold_val)}, Te={len(df_fold_test)}")
        try: tokenizer = AutoTokenizer.from_pretrained(overall_cfg.model.encoder_name)
        except Exception as e: logger.error(f"[F{fold_num_display}] Tokenizer fail: {e}"); continue
        train_loader, val_loader, _ = get_data_loaders(overall_cfg, tokenizer, df_fold_train, df_fold_val, pd.DataFrame(), current_fold_num=fold_num_display)
        if train_loader is None: logger.error(f"[F{fold_num_display}] Train loader None. Skip."); continue
        model = PatchValidator(overall_cfg)
        if device.type == 'cuda' and num_gpus_for_dp > 1:
            model = nn.DataParallel(model, device_ids=dp_device_ids)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=overall_cfg.train.learning_rate, weight_decay=overall_cfg.train.weight_decay)
        criterion = None
        if overall_cfg.train.get('use_weighted_loss', False):
            pos_mult = overall_cfg.train.get('pos_weight_multiplier', 1.0)
            counts = df_fold_train[label_col_cfg].value_counts(); n_neg=counts.get(0,0); n_pos=counts.get(1,0)
            if n_pos > 0 and n_neg > 0:
                base_w = n_neg / n_pos; final_w = base_w * pos_mult
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([final_w]).to(device))
            else: criterion = nn.BCEWithLogitsLoss()
        else: criterion = nn.BCEWithLogitsLoss()
        scheduler = None; num_steps_fold = len(train_loader) * overall_cfg.train.epochs
        if overall_cfg.train.get('lr_scheduler'):
            s_type = overall_cfg.train.lr_scheduler.lower(); ratio = overall_cfg.train.get('warmup_steps_ratio',0.0); wu_steps = int(num_steps_fold*ratio)
            if s_type == 'linear': scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=wu_steps,num_training_steps=num_steps_fold)
            elif s_type == 'cosine': scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=wu_steps,num_training_steps=num_steps_fold)
            if scheduler: logger.info(f"[F{fold_num_display}] Scheduler: {s_type.upper()} ({wu_steps} warmup / {num_steps_fold} total).")
        best_val_met_f = -float('inf'); epochs_no_imp_f = 0; best_ep_f = 0
        patience_f = overall_cfg.train.get('early_stopping_patience', 0)
        early_stop_f_enabled = patience_f > 0 and val_loader is not None
        if early_stop_f_enabled: logger.info(f"[F{fold_num_display}] Early stop: patience={patience_f}")
        comp_f_epochs = 0
        for ep in range(overall_cfg.train.epochs):
            comp_f_epochs = ep + 1; model.train(); total_loss_f = 0.0; steps_in_ep_f = 0
            for batch in train_loader:
                labels_b = batch['label'].to(device).float()
                outputs_b = model(input_ids_flaky=batch['input_ids_flaky'].to(device), attention_mask_flaky=batch['attention_mask_flaky'].to(device),
                                input_ids_patch=batch['input_ids_patch'].to(device), attention_mask_patch=batch['attention_mask_patch'].to(device))
                loss_b = criterion(outputs_b.squeeze(-1), labels_b)
                optimizer.zero_grad(); loss_b.backward(); optimizer.step()
                if scheduler: scheduler.step()
                total_loss_f += loss_b.item(); steps_in_ep_f += 1
            avg_loss_f = total_loss_f / steps_in_ep_f if steps_in_ep_f > 0 else np.nan
            logger.info(f"[F{fold_num_display}] End Ep {comp_f_epochs}: TrainLoss={avg_loss_f:.4f}")
            if val_loader and overall_cfg.train.eval_strategy == "epoch":
                model.eval(); val_probs_f = []; val_lbls_f = []
                with torch.no_grad():
                    for batch_v in val_loader:
                        lbls_v = batch_v['label'].to(device).float()
                        outs_v = model(input_ids_flaky=batch_v['input_ids_flaky'].to(device), attention_mask_flaky=batch_v['attention_mask_flaky'].to(device),
                                       input_ids_patch=batch_v['input_ids_patch'].to(device), attention_mask_patch=batch_v['attention_mask_patch'].to(device))
                        probs_v = torch.sigmoid(outs_v).squeeze(-1).cpu().numpy(); val_probs_f.extend(probs_v); val_lbls_f.extend(lbls_v.cpu().numpy().astype(int))
                if not val_lbls_f: logger.warning(f"[F{fold_num_display}] Val Ep {comp_f_epochs}: No val preds. Skip metrics."); continue
                val_mets_f = utils.calculate_metrics(val_lbls_f, val_probs_f, is_probs=True)
                f1p, f1n = val_mets_f['positive']['f1'], val_mets_f['negative']['f1']
                cur_met_f = (f1p + f1n) / 2.0; met_name_f = "AvgF1"
                logger.info(f"[F{fold_num_display}] Val Ep {comp_f_epochs}: Acc={val_mets_f['accuracy']:.4f}, {met_name_f}={cur_met_f:.4f} (PosF1:{f1p:.4f}, NegF1:{f1n:.4f})")
                if early_stop_f_enabled:
                    if cur_met_f > best_val_met_f:
                        best_val_met_f = cur_met_f; best_ep_f = comp_f_epochs; epochs_no_imp_f = 0
                        best_mdl_f_path = os.path.join(fold_output_dir, f"best_model_fold_{fold_num_display}.pt")
                        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                        torch.save(state_dict, best_mdl_f_path); logger.info(f"[F{fold_num_display}] Best val saved: {best_mdl_f_path}")
                    else: epochs_no_imp_f += 1
                    if epochs_no_imp_f >= patience_f: logger.info(f"[F{fold_num_display}] Early stopping. Best ep {best_ep_f}"); break
        final_model_path_for_eval, ckpt_base_name_for_file = "", ""
        state_to_save_last = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        last_ep_path = os.path.join(fold_output_dir, f"last_epoch_model_fold_{fold_num_display}_epoch_{comp_f_epochs}.pt")
        torch.save(state_to_save_last, last_ep_path)
        if early_stop_f_enabled and best_ep_f > 0 and os.path.exists(os.path.join(fold_output_dir, f"best_model_fold_{fold_num_display}.pt")):
            final_model_path_for_eval = os.path.join(fold_output_dir, f"best_model_fold_{fold_num_display}.pt")
            ckpt_base_name_for_file = f"best_fold_{fold_num_display}_ep_{best_ep_f}"
        else:
            final_model_path_for_eval = last_ep_path
            ckpt_base_name_for_file = f"last_fold_{fold_num_display}_ep_{comp_f_epochs}"
        logger.info(f"[F{fold_num_display}] Using model from {final_model_path_for_eval} for test.")
        if df_fold_test.empty:
            logger.warning(f"[F{fold_num_display}] Test data empty. Skip eval.")
            all_folds_evaluation_summaries.append({'fold_num': fold_num_display, 'model_metrics': {}, 'baseline_metrics': {}, 'incorrect_type_detection': [], 'true_labels_for_overall_roc': [], 'model_probs_for_overall_roc': []})
        else:
            model_for_eval = PatchValidator(overall_cfg)
            model_for_eval.load_state_dict(torch.load(final_model_path_for_eval, map_location=device))
            if device.type == 'cuda' and num_gpus_for_dp > 1:
                model_for_eval = nn.DataParallel(model_for_eval, device_ids=dp_device_ids)
            model_for_eval.to(device)
            fold_eval_summary_dict = evaluate_and_save_fold_results(overall_cfg, model_for_eval, df_fold_test, tokenizer, device, fold_output_dir, fold_num_display, checkpoint_name_base=ckpt_base_name_for_file)
            if fold_eval_summary_dict:
                all_folds_evaluation_summaries.append(fold_eval_summary_dict)
                overall_true_labels_all_folds.extend(fold_eval_summary_dict.get('true_labels_for_overall_roc', []))
                overall_model_probs_all_folds.extend(fold_eval_summary_dict.get('model_probs_for_overall_roc', []))
            else:
                all_folds_evaluation_summaries.append({'fold_num': fold_num_display, 'model_metrics': {}, 'baseline_metrics': {}, 'incorrect_type_detection': [], 'true_labels_for_overall_roc': [], 'model_probs_for_overall_roc': []})
        logger.info(f"===== Finished Fold {fold_num_display}/{n_folds} =====")
        del model, optimizer, scheduler, train_loader, val_loader, criterion, model_for_eval
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("\n\n===== K-Fold Cross-Validation Finished: Aggregating Results =====")
    if not all_folds_evaluation_summaries:
        logger.error("No results from any fold. Cannot generate final summary."); return
    plot_filename_base = "kfold_overall_final"
    if overall_true_labels_all_folds and overall_model_probs_all_folds:
        overall_true_np = np.array(overall_true_labels_all_folds)
        overall_probs_np = np.array(overall_model_probs_all_folds)
        if len(np.unique(overall_true_np)) > 1:
            utils.plot_roc_curve(overall_true_np, overall_probs_np, base_output_dir, filename=f"{plot_filename_base}_concatenated_roc.png", extra_threshold_data=None)
            utils.plot_precision_recall_curve(overall_true_np, overall_probs_np, base_output_dir, filename=f"{plot_filename_base}_concatenated_pr.png")
            tprs_interp = []; aucs_folds_collected = []
            mean_fpr = np.linspace(0,1,100)
            fig_m_roc, ax_m_roc = plt.subplots(figsize=(8,8))
            for fold_summary in all_folds_evaluation_summaries:
                fold_true_roc = np.array(fold_summary.get('true_labels_for_overall_roc',[]))
                fold_probs_roc = np.array(fold_summary.get('model_probs_for_overall_roc',[]))
                fold_auc_val = fold_summary.get('model_metrics',{}).get('auc', np.nan)
                if len(fold_true_roc) > 0 and len(fold_probs_roc) > 0 and len(np.unique(fold_true_roc)) > 1:
                    fpr_f, tpr_f, _ = roc_curve(fold_true_roc, fold_probs_roc)
                    ax_m_roc.plot(fpr_f, tpr_f, lw=1, alpha=0.3)
                    tprs_interp.append(np.interp(mean_fpr, fpr_f, tpr_f))
                    if not np.isnan(fold_auc_val) : aucs_folds_collected.append(fold_auc_val)
            if tprs_interp:
                mean_tpr = np.mean(tprs_interp, axis=0); mean_tpr[0]=0.0; mean_tpr[-1]=1.0
                std_tpr = np.std(tprs_interp, axis=0)
                mean_auc = np.nanmean(aucs_folds_collected) if aucs_folds_collected else np.nan
                std_auc = np.nanstd(aucs_folds_collected) if aucs_folds_collected else np.nan
                ax_m_roc.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.3f} \u00B1 {std_auc:.3f})', lw=2.5, alpha=.8)
                ax_m_roc.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr,0), np.minimum(mean_tpr+std_tpr,1), color='grey', alpha=.2, label=r'$\pm$ 1 standard deviation')
            ax_m_roc.plot([0,1],[0,1],linestyle='--',lw=2,color='red',label='Random Chance',alpha=.8)
            ax_m_roc.set_xlim([-0.05,1.05]); ax_m_roc.set_ylim([-0.05,1.05]); ax_m_roc.set_xlabel('False Positive Rate'); ax_m_roc.set_ylabel('True Positive Rate')
            ax_m_roc.set_title('Mean ROC Curve (K-Fold)'); ax_m_roc.legend(loc='lower right'); ax_m_roc.grid(True,alpha=0.3); ax_m_roc.set_aspect('equal', adjustable='box'); fig_m_roc.tight_layout()
            fig_m_roc.savefig(os.path.join(base_output_dir, f"{plot_filename_base}_mean_roc.png"), dpi=300)
            plt.close(fig_m_roc)
            logger.info(f"Mean ROC plot saved to {os.path.join(base_output_dir, f'{plot_filename_base}_mean_roc.png')}")
        else: logger.warning("Not enough data/classes for overall ROC/AUC plots.")
    else: logger.warning("No data for overall ROC/AUC plots.")
    final_summary_rows_list = []
    header_base_final = ['Fold', 'Method', 'Accuracy', 'AUC',
                   'Precision_Positive', 'Recall_Positive', 'F1_Positive',
                   'Precision_Negative', 'Recall_Negative', 'F1_Negative',
                   'Avg_Run_Time_Ms']
    type_headers_final_set = set()
    for fold_summary_item in all_folds_evaluation_summaries:
        f_num = fold_summary_item['fold_num']
        m_metrics = fold_summary_item.get('model_metrics', {})
        b_metrics = fold_summary_item.get('baseline_metrics', {})
        inc_types_data = fold_summary_item.get('incorrect_type_detection', [])
        m_row = {'Fold': f_num, 'Method': 'Model (Thr=0.5)'}
        m_row['Accuracy'] = m_metrics.get('accuracy', np.nan)
        m_row['AUC'] = m_metrics.get('auc', np.nan)
        m_row['Avg_Run_Time_Ms'] = m_metrics.get('avg_run_time_ms', np.nan)
        for cls_k,cls_v_suffix in [('positive','Positive'),('negative','Negative')]:
            for met_k,met_v_suffix in [('precision','Precision'),('recall','Recall'),('f1','F1')]:
                m_row[f"{met_v_suffix}_{cls_v_suffix}"] = m_metrics.get(cls_k,{}).get(met_k, np.nan)
        b_row = {'Fold': f_num, 'Method': 'Baseline (Rerun)'}
        if b_metrics:
            b_row['Accuracy'] = b_metrics.get('accuracy', np.nan)
            b_row['AUC'] = b_metrics.get('auc', np.nan)
            b_row['Avg_Run_Time_Ms'] = b_metrics.get('avg_run_time_ms', np.nan)
            for cls_k,cls_v_suffix in [('positive','Positive'),('negative','Negative')]:
                for met_k,met_v_suffix in [('precision','Precision'),('recall','Recall'),('f1','F1')]:
                    b_row[f"{met_v_suffix}_{cls_v_suffix}"] = b_metrics.get(cls_k,{}).get(met_k, np.nan)
        else:
            for col_n_base_metric in header_base_final[2:]: b_row[col_n_base_metric] = np.nan
        for item in inc_types_data:
            t_val = item['incorrect_type']
            tk_total=f"Total_Actual_IncorrectType_{t_val}"; tk_detected=f"Detected_IncorrectType_{t_val}"; tk_recall=f"Recall_IncorrectType_{t_val}"
            type_headers_final_set.update([tk_total, tk_detected, tk_recall])
            m_row[tk_total]=item['total_actual']; m_row[tk_detected]=item['model_detected']; m_row[tk_recall]=item['model_recall']
            b_row[tk_total]=item['total_actual']; b_row[tk_detected]=item['baseline_detected']; b_row[tk_recall]=item['baseline_recall']
        final_summary_rows_list.append(m_row); final_summary_rows_list.append(b_row)
    final_summary_df_all_folds = pd.DataFrame(final_summary_rows_list)
    if not final_summary_df_all_folds.empty:
        avg_m_row = {'Fold': 'Average', 'Method': 'Model (Thr=0.5)'}
        avg_b_row = {'Fold': 'Average', 'Method': 'Baseline (Rerun)'}
        all_model_fold_avg_times = [f_sum.get('model_metrics',{}).get('avg_run_time_ms', np.nan) for f_sum in all_folds_evaluation_summaries]
        all_baseline_fold_avg_times = [f_sum.get('baseline_metrics',{}).get('avg_run_time_ms', np.nan) for f_sum in all_folds_evaluation_summaries]
        overall_avg_model_pred_time_ms_final = np.nanmean([t for t in all_model_fold_avg_times if pd.notna(t)]) if any(pd.notna(t) for t in all_model_fold_avg_times) else np.nan
        overall_avg_baseline_rerun_time_ms_final = np.nanmean([t for t in all_baseline_fold_avg_times if pd.notna(t)]) if any(pd.notna(t) for t in all_baseline_fold_avg_times) else np.nan
        num_cols_avg = [c for c in final_summary_df_all_folds.columns if c not in ['Fold','Method', 'Avg_Run_Time_Ms'] and pd.api.types.is_numeric_dtype(final_summary_df_all_folds[c])]
        for c in num_cols_avg:
            avg_m_row[c] = final_summary_df_all_folds[(final_summary_df_all_folds['Method']=='Model (Thr=0.5)') & (pd.to_numeric(final_summary_df_all_folds['Fold'],errors='coerce').notna())][c].mean()
            avg_b_row[c] = final_summary_df_all_folds[(final_summary_df_all_folds['Method']=='Baseline (Rerun)') & (pd.to_numeric(final_summary_df_all_folds['Fold'],errors='coerce').notna())][c].mean()
        avg_m_row['Avg_Run_Time_Ms'] = overall_avg_model_pred_time_ms_final
        avg_b_row['Avg_Run_Time_Ms'] = overall_avg_baseline_rerun_time_ms_final
        final_summary_df_all_folds = pd.concat([final_summary_df_all_folds, pd.DataFrame([avg_m_row, avg_b_row])], ignore_index=True)
        ordered_type_cols_f = sorted(list(type_headers_final_set))
        final_cols_ordered_f = header_base_final + ordered_type_cols_f
        for c_ord in final_cols_ordered_f:
            if c_ord not in final_summary_df_all_folds.columns: final_summary_df_all_folds[c_ord] = np.nan
        final_summary_df_all_folds = final_summary_df_all_folds.reindex(columns=final_cols_ordered_f)
        final_summary_path_overall = os.path.join(base_output_dir, "kfold_TRAINING_SCRIPT_aggregated_summary.csv")
        try: final_summary_df_all_folds.to_csv(final_summary_path_overall, index=False, float_format='%.4f', na_rep='NaN'); logger.info(f"K-Fold overall summary saved: {final_summary_path_overall}")
        except Exception as e: logger.error(f"Failed save K-Fold overall summary: {e}")
    else: logger.warning("Final summary DF empty. No overall summary CSV.")
    logger.info("Overall K-Fold training and evaluation process finished.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flaky Patch Validator Model with K-Fold CV")
    parser.add_argument('--config_file', type=str, default='config/default_config.json', help='Path to config JSON')
    args_cli = parser.parse_args()
    try:
        cfg_main = Configure(config_json_file=args_cli.config_file)
    except FileNotFoundError: print(f"FATAL: Config file not found: {args_cli.config_file}"); exit(1)
    except Exception as e_cfg: print(f"FATAL: Error loading config {args_cli.config_file}: {e_cfg}"); exit(1)
    train_and_evaluate_kfold(cfg_main)