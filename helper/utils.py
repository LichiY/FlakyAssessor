import pandas as pd
import numpy as np
import random
import torch
import random
import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt

import helper.logger as logger
import os



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data from {csv_path}. Shape: {df.shape}")
        critical_cols = ['isCorrect', 'flaky_code', 'generated_patch']
        if df[critical_cols].isnull().values.any():
            nan_counts = df[critical_cols].isnull().sum()
            logger.warning(f"Dataset contains missing values in critical columns:\n{nan_counts[nan_counts > 0]}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {csv_path}: {e}")
        raise
def split_by_project(df, config):
    pass
def calculate_metrics(y_true, y_pred_input, threshold=0.5, is_probs=True):
    y_true_np = np.array(y_true)
    if is_probs:
        y_pred_np = (np.array(y_pred_input) >= threshold).astype(int)
    else:
        y_pred_np = np.array(y_pred_input).astype(int)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)
    metrics['positive'] = {
        'precision': precision_score(y_true_np, y_pred_np, labels=[1], pos_label=1, average='binary', zero_division=0),
        'recall':    recall_score(y_true_np, y_pred_np, labels=[1], pos_label=1, average='binary', zero_division=0),
        'f1':        f1_score(y_true_np, y_pred_np, labels=[1], pos_label=1, average='binary', zero_division=0)
    }
    metrics['negative'] = {
        'precision': precision_score(y_true_np, y_pred_np, labels=[0], pos_label=0, average='binary', zero_division=0),
        'recall':    recall_score(y_true_np, y_pred_np, labels=[0], pos_label=0, average='binary', zero_division=0),
        'f1':        f1_score(y_true_np, y_pred_np, labels=[0], pos_label=0, average='binary', zero_division=0)
    }
    return metrics
def calculate_failure_type_recall(true_labels, predicted_labels, failure_types, target_type):
    labels = np.array(true_labels); predictions = np.array(predicted_labels); failure_types = np.array(failure_types)
    is_truly_incorrect = (labels == 0)
    is_target_failure_type = (failure_types == target_type)
    is_predicted_incorrect = (predictions == 0)
    true_target_failure_mask = is_truly_incorrect & is_target_failure_type
    true_target_failures_count = np.sum(true_target_failure_mask)
    if true_target_failures_count == 0: return np.nan
    correctly_predicted_target_failures_mask = true_target_failure_mask & is_predicted_incorrect
    correctly_predicted_target_failures_count = np.sum(correctly_predicted_target_failures_mask)
    recall = correctly_predicted_target_failures_count / true_target_failures_count
    return {'recall': recall, 'detected': correctly_predicted_target_failures_count, 'total_actual': true_target_failures_count}
def plot_roc_curve(y_true, y_pred_probs, output_dir, filename="roc_curve.png", extra_threshold_data=None):
    y_true = np.array(y_true); y_pred_probs = np.array(y_pred_probs)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs); roc_auc = auc(fpr, tpr)
    logger.info(f"Calculated AUC: {roc_auc:.4f}")
    try:
        idx_thresh_05 = np.argmin(np.abs(thresholds - 0.5)); thresh_05_val = thresholds[idx_thresh_05]
        plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2.8, label=f'ROC Curve (AUC = {roc_auc:.3f})', zorder=3); plt.fill_between(fpr, tpr, color='darkorange', alpha=0.2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance', zorder=2)
        plt.scatter(fpr[idx_thresh_05], tpr[idx_thresh_05], marker='o', s=120, facecolors='none', edgecolors='red', linewidths=2, zorder=5, label=f'Threshold ≈ {thresh_05_val:.2f}')
        if extra_threshold_data:
            extra_thresh_val = extra_threshold_data['value']; idx_extra = np.argmin(np.abs(thresholds - extra_thresh_val)); extra_label = f"{extra_threshold_data.get('label_prefix', 'Extra')} Thr ≈ {thresholds[idx_extra]:.2f}"
            plt.scatter(fpr[idx_extra], tpr[idx_extra], marker=extra_threshold_data.get('marker', '*'), s=extra_threshold_data.get('size', 150), facecolors='none', edgecolors=extra_threshold_data.get('color', 'blue'), linewidths=extra_threshold_data.get('linewidth', 2), zorder=5, label=extra_label)
        plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02]); plt.xlabel('False Positive Rate', fontsize=13); plt.ylabel('True Positive Rate', fontsize=13); plt.title('Receiver Operating Characteristic (ROC)', fontsize=15, pad=15)
        plt.legend(loc="lower right", fontsize=10, frameon=True, facecolor='white', framealpha=0.8); plt.grid(True, linestyle=':', alpha=0.6); plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(pad=1.5)
        os.makedirs(output_dir, exist_ok=True); save_path = os.path.join(output_dir, filename); plt.savefig(save_path, dpi=300, bbox_inches='tight'); logger.info(f"ROC plot saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Failed ROC plot: {e}"); plt.close()
    return roc_auc
def plot_precision_recall_curve(y_true, y_pred_probs, output_dir, filename="precision_recall_curve.png"):
    y_true = np.array(y_true); y_pred_probs = np.array(y_pred_probs)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_probs); avg_precision = average_precision_score(y_true, y_pred_probs)
    logger.info(f"Calculated Avg Precision (AP): {avg_precision:.4f}")
    try:
        metrics_05 = calculate_metrics(y_true, y_pred_probs, threshold=0.5); pr_05, rec_05 = metrics_05['positive']['precision'], metrics_05['positive']['recall']
        plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(8, 7))
        plt.plot(recall, precision, color='steelblue', lw=2.5, label=f'PR Curve (AP = {avg_precision:.3f})')
        if pr_05 is not None and rec_05 is not None: plt.scatter(rec_05, pr_05, marker='o', color='red', s=100, zorder=5, label=f'Threshold ≈ 0.5 (P={pr_05:.2f}, R={rec_05:.2f})')
        plt.xlim([0.0, 1.02]); plt.ylim([0.0, 1.02]); plt.xlabel('Recall (Pos)', fontsize=12); plt.ylabel('Precision (Pos)', fontsize=12); plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=11); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True); save_path = os.path.join(output_dir, filename); plt.savefig(save_path, dpi=300); logger.info(f"PR curve saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Failed PR plot: {e}"); plt.close()
    return avg_precision
def plot_metrics_vs_threshold(y_true, y_pred_probs, output_dir, filename="metrics_vs_threshold.png", steps=100):
    y_true = np.array(y_true); y_pred_probs = np.array(y_pred_probs); thresholds = np.linspace(0.01, 0.99, steps)
    metrics_neg = {'precision': [], 'recall': [], 'f1': []}; metrics_pos = {'precision': [], 'recall': [], 'f1': []}; accuracy = []
    logger.info("Calculating metrics across thresholds...")
    for thresh in thresholds: results = calculate_metrics(y_true, y_pred_probs, threshold=thresh); metrics_neg['precision'].append(results['negative']['precision']); metrics_neg['recall'].append(results['negative']['recall']); metrics_neg['f1'].append(results['negative']['f1']); metrics_pos['precision'].append(results['positive']['precision']); metrics_pos['recall'].append(results['positive']['recall']); metrics_pos['f1'].append(results['positive']['f1']); accuracy.append(results['accuracy'])
    try:
        plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(10, 7))
        plt.plot(thresholds, metrics_pos['precision'], '--', color='blue', label='Precision (1)'); plt.plot(thresholds, metrics_pos['recall'], '-.', color='blue', label='Recall (1)'); plt.plot(thresholds, metrics_pos['f1'], '-', color='blue', lw=2, label='F1-Score (1)')
        plt.plot(thresholds, metrics_neg['precision'], '--', color='green', label='Precision (0)'); plt.plot(thresholds, metrics_neg['recall'], '-.', color='green', label='Recall (0)'); plt.plot(thresholds, metrics_neg['f1'], '-', color='green', lw=2, label='F1-Score (0)')
        plt.axvline(x=0.5, color='red', linestyle=':', lw=2, label='Threshold=0.5')
        f1_neg_scores = np.array(metrics_neg['f1'])
        if len(f1_neg_scores) > 0 and not np.all(np.isnan(f1_neg_scores)):
             valid_f1_neg = f1_neg_scores[~np.isnan(f1_neg_scores)]
             if len(valid_f1_neg) > 0 and np.max(valid_f1_neg) > 0: best_f1_neg_idx = np.nanargmax(f1_neg_scores); best_thresh_neg = thresholds[best_f1_neg_idx]; plt.axvline(x=best_thresh_neg, color='darkgreen', linestyle=':', lw=1.5, label=f'Max F1 (0) at ~{best_thresh_neg:.2f}')
        plt.xlabel('Classification Threshold', fontsize=12); plt.ylabel('Metric Score', fontsize=12); plt.title('Metrics vs. Classification Threshold', fontsize=14)
        plt.legend(loc='best', fontsize=10); plt.grid(True, linestyle='--', alpha=0.7); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.02]); plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True); save_path = os.path.join(output_dir, filename); plt.savefig(save_path, dpi=300); logger.info(f"Metrics vs Threshold plot saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Failed Metrics vs Threshold plot: {e}"); plt.close
def split_data_for_project_eval(df_full, cfg):
    """
    Splits data project-disjointly targeting ~60:20:20 (Train:Val:Test).
    Test set projects must have full rerun_consistency.
    Special data is added to the training set.
    """
    project_col = cfg.data.project_col
    rerun_consistency_col = cfg.data.rerun_consistency_col
    source_file_col = cfg.data.source_file_col
    label_col = cfg.data.label_col
    overall_test_ratio = cfg.data.overall_test_project_ratio
    train_ratio_of_remaining = cfg.data.train_ratio_of_remaining_pool
    logger.info(f"Starting project-based split (target ~60:20:20). Project col: '{project_col}'.")
    if project_col not in df_full.columns or \
       rerun_consistency_col not in df_full.columns or \
       source_file_col not in df_full.columns or \
       label_col not in df_full.columns:
        logger.error("One or more critical columns for splitting are missing in the dataset. Exiting.")
        return None, None, None
    all_unique_projects = df_full[project_col].unique()
    np.random.seed(cfg.seed)
    np.random.shuffle(all_unique_projects)
    test_candidate_projects = []
    non_test_candidate_projects = []
    for project in all_unique_projects:
        project_data = df_full[df_full[project_col] == project]
        if project_data[rerun_consistency_col].notna().all():
            test_candidate_projects.append(project)
        else:
            non_test_candidate_projects.append(project)
    logger.info(f"Found {len(test_candidate_projects)} test candidate projects (all non-NaN rerun_consistency).")
    logger.info(f"Found {len(non_test_candidate_projects)} non-test candidate projects (some NaN rerun_consistency).")
    num_target_test_projects = int(len(all_unique_projects) * overall_test_ratio)
    selected_test_projects = []
    if len(test_candidate_projects) <= num_target_test_projects:
        selected_test_projects.extend(test_candidate_projects)
        logger.info(f"Taking all {len(selected_test_projects)} test candidate projects for the test set.")
    else:
        selected_test_projects.extend(np.random.choice(test_candidate_projects, num_target_test_projects, replace=False).tolist())
        logger.info(f"Randomly selected {len(selected_test_projects)} projects from test candidates for the test set.")
    if not selected_test_projects:
        logger.warning("No projects selected for the test set based on criteria. Test set will be empty.")
        df_test = pd.DataFrame(columns=df_full.columns)
    else:
        df_test = df_full[df_full[project_col].isin(selected_test_projects)].copy()
    projects_for_pool = [p for p in all_unique_projects if p not in selected_test_projects]
    df_pool_initial = df_full[df_full[project_col].isin(projects_for_pool)].copy()
    df_special_train_fixed = pd.DataFrame(columns=df_full.columns)
    if not df_pool_initial.empty and source_file_col in df_pool_initial.columns and rerun_consistency_col in df_pool_initial.columns:
        special_condition = (df_pool_initial[source_file_col] == "flaky_fix_181_gpt35_labeled.csv") & \
                            (df_pool_initial[rerun_consistency_col].isna())
        df_special_train_fixed = df_pool_initial[special_condition].copy()
        df_pool_for_train_val_split = df_pool_initial[~special_condition].copy()
        logger.info(f"Isolated {len(df_special_train_fixed)} special samples from non-test data.")
    else:
        df_pool_for_train_val_split = df_pool_initial.copy()
        logger.info("No special training data isolated or relevant columns missing from pool.")
    df_train_main = pd.DataFrame(columns=df_full.columns)
    df_val = pd.DataFrame(columns=df_full.columns)
    if not df_pool_for_train_val_split.empty:
        pool_projects_for_split = df_pool_for_train_val_split[project_col].unique()
        np.random.shuffle(pool_projects_for_split)
        if len(pool_projects_for_split) == 0:
            logger.warning("No projects in the pool for train/val splitting.")
        elif len(pool_projects_for_split) == 1:
            logger.warning(f"Only one project ('{pool_projects_for_split[0]}') in pool for train/val. Assigning all to main training.")
            df_train_main = df_pool_for_train_val_split.copy()
        else:
            num_train_main_projects = int(len(pool_projects_for_split) * train_ratio_of_remaining)
            if num_train_main_projects == len(pool_projects_for_split) and len(pool_projects_for_split) > 1:
                num_train_main_projects -= 1
            if num_train_main_projects == 0 and len(pool_projects_for_split) > 0:
                num_train_main_projects = 1
            train_main_projects_list = pool_projects_for_split[:num_train_main_projects]
            val_projects_list = pool_projects_for_split[num_train_main_projects:]
            df_train_main = df_pool_for_train_val_split[df_pool_for_train_val_split[project_col].isin(train_main_projects_list)].copy()
            df_val = df_pool_for_train_val_split[df_pool_for_train_val_split[project_col].isin(val_projects_list)].copy()
            logger.info(f"Split project pool: {len(train_main_projects_list)} projects for main train, {len(val_projects_list)} for validation.")
    else:
        logger.warning("Pool for train/val splitting (after special data removal) is empty.")
    if not df_special_train_fixed.empty:
        df_train_final = pd.concat([df_train_main, df_special_train_fixed], ignore_index=True).reset_index(drop=True)
    else:
        df_train_final = df_train_main.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    logger.info(f"Final sizes: Train={len(df_train_final)}, Val={len(df_val)}, Test={len(df_test)}")
    if not df_train_final.empty: logger.info(f"Train final label distribution:\n{df_train_final[label_col].value_counts(normalize=True, dropna=False)}")
    if not df_val.empty: logger.info(f"Validation label distribution:\n{df_val[label_col].value_counts(normalize=True, dropna=False)}")
    if not df_test.empty: logger.info(f"Test label distribution:\n{df_test[label_col].value_counts(normalize=True, dropna=False)}")
    return df_train_final, df_val, df_test
