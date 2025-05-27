import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import logging
import helper.logger as logger
from helper.configure import Configure
import helper.utils as utils
def aggregate_fold_results(config_file_path, base_kfold_output_dir, num_folds):
    try:
        cfg = Configure(config_json_file=config_file_path)
        log_init = logger.Logger(cfg)
    except Exception as e:
        print(f"Error loading configuration for aggregation: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')
        logger.error(f"Failed to initialize custom logger: {e}. Using basic logging.")
    logger.info("***** Starting K-Fold Results Aggregation *****")
    logger.info(f"Base K-Fold output directory: {base_kfold_output_dir}")
    logger.info(f"Number of folds to process: {num_folds}")
    all_fold_true_labels_list = []
    all_fold_model_probs_list = []
    all_fold_model_prediction_times_secs_collected = []
    all_fold_baseline_rerun_times_raw_collected = []
    all_fold_metrics_model_list = []
    all_fold_metrics_baseline_list = []
    all_fold_incorrect_type_data_list = []
    label_col_in_csv = 'true_label'
    rerun_consistency_col_name = cfg.data.get('rerun_consistency_col', "rerun_consistency")
    incorrect_type_col_name = cfg.data.get('incorrect_type_col', "incorrect_type")
    generated_source_marker_col = cfg.data.get('generated_source_marker_col', "augmented_source_marker")
    baseline_original_time_col_name = 'rerun_time_consume'
    baseline_time_col_in_detailed_csv = 'rerun_time_consume'
    model_prediction_time_col_in_csv = 'prediction_time_secs'
    for i in range(1, num_folds + 1):
        fold_num_display = i
        logger.info(f"--- Processing Fold {fold_num_display} ---")
        fold_dir_path = os.path.join(base_kfold_output_dir, f"fold_{fold_num_display}")
        detail_csv_files = []
        if os.path.exists(fold_dir_path):
            detail_csv_files = [f for f in os.listdir(fold_dir_path) if f.startswith(f"fold_{fold_num_display}_evaluation_details_") and f.endswith(".csv")]
        if not detail_csv_files:
            logger.warning(f"No detailed CSV for Fold {fold_num_display}. Appending placeholders.")
            all_fold_metrics_model_list.append({'avg_run_time_ms': np.nan});
            all_fold_metrics_baseline_list.append({'avg_run_time_ms': np.nan});
            all_fold_incorrect_type_data_list.append([])
            continue
        detail_csv_path = os.path.join(fold_dir_path, detail_csv_files[0])
        logger.info(f"Reading: {detail_csv_path}")
        try: df_fold_details = pd.read_csv(detail_csv_path)
        except Exception as e: logger.error(f"Failed read CSV Fold {fold_num_display}: {e}"); continue
        if df_fold_details.empty: logger.warning(f"CSV Fold {fold_num_display} empty. Skipping."); continue
        if label_col_in_csv not in df_fold_details.columns:
            logger.error(f"Col '{label_col_in_csv}' not in {detail_csv_path}. Skipping.");
            all_fold_metrics_model_list.append({'avg_run_time_ms': np.nan});
            all_fold_metrics_baseline_list.append({'avg_run_time_ms': np.nan});
            all_fold_incorrect_type_data_list.append([])
            continue
        fold_true_labels = df_fold_details[label_col_in_csv].values.astype(int)
        fold_model_probs = df_fold_details['model_predicted_prob'].values.astype(float)
        fold_model_preds_thresh05 = df_fold_details['model_predicted_label_thresh0.5'].values.astype(int)
        current_fold_model_pred_times_ms = np.nan
        if model_prediction_time_col_in_csv in df_fold_details.columns:
            model_times_sec = df_fold_details[model_prediction_time_col_in_csv].dropna().values
            if len(model_times_sec) > 0:
                current_fold_model_pred_times_ms = np.mean(model_times_sec) * 1000
                all_fold_model_prediction_times_secs_collected.extend(model_times_sec.tolist())
        current_fold_baseline_rerun_times_ms = np.nan
        if baseline_time_col_in_detailed_csv in df_fold_details.columns:
            baseline_times_raw_fold = pd.to_numeric(df_fold_details[baseline_time_col_in_detailed_csv], errors='coerce').dropna().values
            if len(baseline_times_raw_fold) > 0:
                avg_b_time_fold_raw = np.mean(baseline_times_raw_fold)
                all_fold_baseline_rerun_times_raw_collected.extend(baseline_times_raw_fold.tolist())
                current_fold_baseline_rerun_times_ms = avg_b_time_fold_raw * 1000
        all_fold_true_labels_list.append(fold_true_labels)
        all_fold_model_probs_list.append(fold_model_probs)
        metrics_m = utils.calculate_metrics(fold_true_labels, fold_model_probs, is_probs=True)
        try:
            fpr_m, tpr_m, _ = roc_curve(fold_true_labels, fold_model_probs)
            metrics_m['auc'] = auc(fpr_m, tpr_m) if not (np.isnan(fpr_m).all() or np.isnan(tpr_m).all()) else np.nan
        except ValueError: metrics_m['auc'] = np.nan
        metrics_m['avg_run_time_ms'] = current_fold_model_pred_times_ms
        all_fold_metrics_model_list.append(metrics_m)
        fold_baseline_preds_labels = None
        metrics_b = {}
        if rerun_consistency_col_name in df_fold_details.columns:
            fold_baseline_preds_labels = df_fold_details[rerun_consistency_col_name].values.astype(int)
            metrics_b = utils.calculate_metrics(fold_true_labels, fold_baseline_preds_labels, is_probs=False)
            try:
                fpr_b, tpr_b, _ = roc_curve(fold_true_labels, fold_baseline_preds_labels)
                metrics_b['auc'] = auc(fpr_b, tpr_b) if not (np.isnan(fpr_b).all() or np.isnan(tpr_b).all()) else np.nan
            except ValueError: metrics_b['auc'] = np.nan
            metrics_b['avg_run_time_ms'] = current_fold_baseline_rerun_times_ms
        all_fold_metrics_baseline_list.append(metrics_b)
        fold_incorrect_type_data_list_current_fold = []
        if incorrect_type_col_name in df_fold_details.columns:
            analysis_mask = np.ones(len(df_fold_details), dtype=bool)
            if generated_source_marker_col in df_fold_details.columns:
                analysis_mask = df_fold_details[generated_source_marker_col] != 'groundtruth_copy'
            true_labels_for_analysis = fold_true_labels[analysis_mask]
            model_preds_for_analysis = fold_model_preds_thresh05[analysis_mask]
            incorrect_types_values = df_fold_details[incorrect_type_col_name].values[analysis_mask]
            baseline_preds_for_analysis = fold_baseline_preds_labels[analysis_mask] if fold_baseline_preds_labels is not None else None
            unique_incorrect_types = sorted(pd.Series(incorrect_types_values[true_labels_for_analysis == 0]).dropna().unique())
            for type_val_float in unique_incorrect_types:
                if pd.isna(type_val_float): continue
                type_val_int = int(type_val_float)
                actual_mask = (true_labels_for_analysis == 0) & (incorrect_types_values == type_val_int)
                total_actual = np.sum(actual_mask)
                m_detected, m_recall, b_detected, b_recall = 0, np.nan, 0, np.nan
                if total_actual > 0:
                    m_detected = np.sum(actual_mask & (model_preds_for_analysis == 0))
                    m_recall = m_detected / total_actual
                    if baseline_preds_for_analysis is not None:
                        b_detected = np.sum(actual_mask & (baseline_preds_for_analysis == 0))
                        b_recall = b_detected / total_actual
                fold_incorrect_type_data_list_current_fold.append({
                    'incorrect_type': type_val_int, 'total_actual': total_actual,
                    'model_detected': m_detected, 'model_recall': m_recall,
                    'baseline_detected': b_detected, 'baseline_recall': b_recall
                })
        all_fold_incorrect_type_data_list.append(fold_incorrect_type_data_list_current_fold)
    if not all_fold_true_labels_list:
        logger.error("No data loaded. Cannot proceed."); return
    overall_avg_model_pred_time_ms = np.nanmean(all_fold_model_prediction_times_secs_collected) * 1000 if all_fold_model_prediction_times_secs_collected else np.nan
    overall_avg_baseline_rerun_time_ms = np.nanmean(all_fold_baseline_rerun_times_raw_collected) * 1000 if all_fold_baseline_rerun_times_raw_collected else np.nan
    logger.info(f"\n--- Overall Average Times (across all test samples from all folds) ---")
    logger.info(f"  Avg Model Prediction Time per sample: {overall_avg_model_pred_time_ms:.4f} ms")
    logger.info(f"  Avg Baseline Rerun Time ('{baseline_original_time_col_name}' converted to ms): {overall_avg_baseline_rerun_time_ms:.4f} ms")
    logger.info("\n--- Overall Model Performance on Concatenated Test Folds (for plotting) ---")
    overall_true_concat = np.concatenate(all_fold_true_labels_list)
    overall_probs_concat = np.concatenate(all_fold_model_probs_list)
    plot_filename_base_overall = "kfold_overall"
    if len(overall_true_concat) > 0 and len(np.unique(overall_true_concat)) > 1 :
        logger.info("Generating overall ROC curve from concatenated fold data...")
        try:
            utils.plot_roc_curve(
                overall_true_concat, overall_probs_concat,
                output_dir=base_kfold_output_dir,
                filename=f"{plot_filename_base_overall}_concatenated_roc_curve.png",
                extra_threshold_data=None
            )
        except Exception as e: logger.error(f"Failed overall concatenated ROC: {e}")
        logger.info("Generating overall PR curve from concatenated fold data...")
        try:
            utils.plot_precision_recall_curve(
                overall_true_concat, overall_probs_concat,
                output_dir=base_kfold_output_dir,
                filename=f"{plot_filename_base_overall}_concatenated_pr_curve.png"
            )
        except Exception as e: logger.error(f"Failed overall concatenated PR: {e}")
    else:
        logger.warning("Not enough data or classes for overall concatenated ROC/PR curves.")
    logger.info("\nGenerating Mean ROC curve across folds...")
    tprs_interp_all_folds = []
    aucs_from_each_fold_actual = [res.get('auc', np.nan) for res in all_fold_metrics_model_list if isinstance(res, dict)]
    mean_fpr_overall = np.linspace(0, 1, 100)
    fig_mean_roc, ax_mean_roc = plt.subplots(figsize=(8, 8))
    for i in range(num_folds):
        if i < len(all_fold_true_labels_list) and i < len(all_fold_model_probs_list):
            fold_true, fold_probs = all_fold_true_labels_list[i], all_fold_model_probs_list[i]
            if len(fold_true) > 0 and len(fold_probs) > 0 and len(np.unique(fold_true)) > 1:
                try:
                    fpr, tpr, _ = roc_curve(fold_true, fold_probs)
                    ax_mean_roc.plot(fpr, tpr, lw=1, alpha=0.3)
                    tprs_interp_all_folds.append(np.interp(mean_fpr_overall, fpr, tpr))
                except ValueError as ve: logger.warning(f"ROC error fold {i+1} for mean plot: {ve}")
            elif len(np.unique(fold_true)) <= 1: logger.warning(f"Fold {i+1} single class for mean ROC.")
    if tprs_interp_all_folds:
        mean_tpr_overall = np.mean(tprs_interp_all_folds, axis=0); mean_tpr_overall[0], mean_tpr_overall[-1] = 0.0, 1.0
        std_tpr_overall = np.std(tprs_interp_all_folds, axis=0)
        tprs_upper = np.minimum(mean_tpr_overall + std_tpr_overall, 1); tprs_lower = np.maximum(mean_tpr_overall - std_tpr_overall, 0)
        mean_auc_kfold = np.nanmean(aucs_from_each_fold_actual) if aucs_from_each_fold_actual else np.nan
        std_auc_kfold = np.nanstd(aucs_from_each_fold_actual) if aucs_from_each_fold_actual else np.nan
        ax_mean_roc.plot(mean_fpr_overall, mean_tpr_overall, color='blue', label=f'Mean ROC (AUC = {mean_auc_kfold:.3f} \u00B1 {std_auc_kfold:.3f})', lw=2.5, alpha=.8)
        ax_mean_roc.fill_between(mean_fpr_overall, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 standard deviation')
    ax_mean_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Random Chance', alpha=.8)
    ax_mean_roc.set_xlim([-0.05, 1.05]); ax_mean_roc.set_ylim([-0.05, 1.05])
    ax_mean_roc.set_xlabel('False Positive Rate'); ax_mean_roc.set_ylabel('True Positive Rate')
    ax_mean_roc.set_title('Mean ROC Curve Across 5-Folds'); ax_mean_roc.legend(loc="lower right")
    ax_mean_roc.grid(True, linestyle=':', alpha=0.6); ax_mean_roc.set_aspect('equal', adjustable='box'); fig_mean_roc.tight_layout()
    mean_roc_path_overall = os.path.join(base_kfold_output_dir, f"{plot_filename_base_overall}_mean_roc_curve.png")
    try: fig_mean_roc.savefig(mean_roc_path_overall, dpi=300); logger.info(f"Mean K-Fold ROC saved: {mean_roc_path_overall}")
    except Exception as e: logger.error(f"Failed save Mean K-Fold ROC: {e}")
    plt.close(fig_mean_roc)
    logger.info("\nPreparing K-Fold final aggregated summary CSV...")
    final_summary_rows = []
    header_base = ['Fold', 'Method', 'Accuracy', 'AUC',
                   'Precision_Positive', 'Recall_Positive', 'F1_Positive',
                   'Precision_Negative', 'Recall_Negative', 'F1_Negative',
                   'Avg_Run_Time_ms']
    type_specific_headers_collected = set()
    for fold_idx in range(num_folds):
        fold_num = fold_idx + 1
        model_fold_metrics = all_fold_metrics_model_list[fold_idx]
        baseline_fold_metrics = all_fold_metrics_baseline_list[fold_idx]
        incorrect_types_this_fold = all_fold_incorrect_type_data_list[fold_idx]
        model_row = {'Fold': fold_num, 'Method': 'Model (Thr=0.5)'}
        model_row['Accuracy'] = model_fold_metrics.get('accuracy', np.nan)
        model_row['AUC'] = model_fold_metrics.get('auc', np.nan)
        model_row['Avg_Run_Time_ms'] = model_fold_metrics.get('avg_run_time_ms', np.nan)
        for cls_key, cls_name_suffix in [('positive', 'Positive'), ('negative', 'Negative')]:
            for metric_key, metric_name_suffix in [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]:
                model_row[f"{metric_name_suffix}_{cls_name_suffix}"] = model_fold_metrics.get(cls_key, {}).get(metric_key, np.nan)
        baseline_row = {'Fold': fold_num, 'Method': 'Baseline (Rerun)'}
        if baseline_fold_metrics:
            baseline_row['Accuracy'] = baseline_fold_metrics.get('accuracy', np.nan)
            baseline_row['AUC'] = baseline_fold_metrics.get('auc', np.nan)
            baseline_row['Avg_Run_Time_ms'] = baseline_fold_metrics.get('avg_run_time_ms', np.nan)
            for cls_key, cls_name_suffix in [('positive', 'Positive'), ('negative', 'Negative')]:
                for metric_key, metric_name_suffix in [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]:
                    baseline_row[f"{metric_name_suffix}_{cls_name_suffix}"] = baseline_fold_metrics.get(cls_key, {}).get(metric_key, np.nan)
        else:
            for col_name_base_metric in header_base[2:]: baseline_row[col_name_base_metric] = np.nan
        for type_item in incorrect_types_this_fold:
            type_val = type_item['incorrect_type']
            tk_total = f"Total_Actual_IncorrectType_{type_val}"
            tk_detected = f"Detected_IncorrectType_{type_val}"
            tk_recall = f"Recall_IncorrectType_{type_val}"
            type_specific_headers_collected.update([tk_total, tk_detected, tk_recall])
            model_row[tk_total] = type_item['total_actual']
            model_row[tk_detected] = type_item['model_detected']
            model_row[tk_recall] = type_item['model_recall']
            baseline_row[tk_total] = type_item['total_actual']
            baseline_row[tk_detected] = type_item['baseline_detected']
            baseline_row[tk_recall] = type_item['baseline_recall']
        final_summary_rows.append(model_row)
        final_summary_rows.append(baseline_row)
    final_summary_df = pd.DataFrame(final_summary_rows)
    if not final_summary_df.empty:
        avg_model_row = {'Fold': 'Average', 'Method': 'Model (Thr=0.5)'}
        avg_baseline_row = {'Fold': 'Average', 'Method': 'Baseline (Rerun)'}
        numeric_cols_for_avg = [col for col in final_summary_df.columns if col not in ['Fold', 'Method', 'Avg_Run_Time_ms'] and pd.api.types.is_numeric_dtype(final_summary_df[col])]
        for col in numeric_cols_for_avg:
            avg_model_row[col] = final_summary_df[(final_summary_df['Method'] == 'Model (Thr=0.5)') & (pd.to_numeric(final_summary_df['Fold'], errors='coerce').notna())][col].mean()
            avg_baseline_row[col] = final_summary_df[(final_summary_df['Method'] == 'Baseline (Rerun)') & (pd.to_numeric(final_summary_df['Fold'], errors='coerce').notna())][col].mean()
        avg_model_row['Avg_Run_Time_ms'] = overall_avg_model_pred_time_ms
        avg_baseline_row['Avg_Run_Time_ms'] = overall_avg_baseline_rerun_time_ms
        final_summary_df = pd.concat([final_summary_df, pd.DataFrame([avg_model_row, avg_baseline_row])], ignore_index=True)
        ordered_type_cols_final = sorted(list(type_specific_headers_collected))
        if 'Avg_Run_Time_ms' not in header_base: header_base.append('Avg_Run_Time_ms')
        final_ordered_columns_all = header_base + ordered_type_cols_final
        current_cols = final_summary_df.columns.tolist()
        for col_name_ordered in final_ordered_columns_all:
            if col_name_ordered not in current_cols: final_summary_df[col_name_ordered] = np.nan
        final_summary_df = final_summary_df.reindex(columns=final_ordered_columns_all)
        final_summary_path = os.path.join(base_kfold_output_dir, f"{plot_filename_base_overall}_kfold_aggregated_summary.csv")
        try:
            final_summary_df.to_csv(final_summary_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"K-Fold final aggregated summary saved: {final_summary_path}")
        except Exception as e: logger.error(f"Failed to save K-Fold final aggregated summary: {e}")
    else:
        logger.warning("Final summary DataFrame is empty. No aggregated summary CSV saved.")
    logger.info("***** K-Fold Results Aggregation Finished *****")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate K-Fold Evaluation Results")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the original config JSON.')
    parser.add_argument('--kfold_output_dir', type=str, required=True, help='Path to base dir of K-Fold outputs.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds processed.')
    cli_args = parser.parse_args()
    aggregate_fold_results(cli_args.config_file, cli_args.kfold_output_dir, cli_args.num_folds)