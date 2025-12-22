from data_modules.dataset import PatchDataset
from data_modules.collator import PatchCollator
from torch.utils.data import DataLoader, WeightedRandomSampler
import helper.logger as logger
import numpy as np
import pandas as pd
import torch
import random


def _seed_worker(worker_id):
    """Initialize worker seeds for reproducibility across DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loaders(config, tokenizer, df_train, df_val, df_test, current_fold_num=None):
    """
    Get data loaders for training, validation, and testing for a given fold.
    df_test is typically empty when called from train.py for train/val loaders.
    """
    collate_fn = PatchCollator(tokenizer)
    log_prefix = f"[Fold {current_fold_num}] " if current_fold_num is not None else ""
    # Set base generator using config.seed for deterministic shuffling/sampling
    base_seed = getattr(config, 'seed', 42)
    torch_generator = torch.Generator()
    try:
        torch_generator.manual_seed(int(base_seed))
    except Exception:
        torch_generator.manual_seed(42)

    train_loader = None
    if not df_train.empty:
        train_sampler = None
        use_sampler = config.train.get('use_weighted_sampler', False)
        label_col = config.data.label_col
        if use_sampler:
            logger.info(f"{log_prefix}Attempting WeightedRandomSampler for training data.")
            try:
                if label_col not in df_train.columns:
                    logger.error(f"{log_prefix}Label column '{label_col}' not found in training data for sampler. Skipping sampler.")
                else:
                    train_labels = df_train[label_col].values
                    class_counts = np.bincount(train_labels)
                    if len(class_counts) < 2 or np.any(class_counts == 0):
                        logger.warning(f"{log_prefix}Training data for sampler contains only one class or zero counts for a class. Sampler may not be effective/needed.")
                        weights = np.ones(len(train_labels))
                    else:
                        weights_per_class = 1. / class_counts
                        weights = weights_per_class[train_labels]
                    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=torch_generator)
                    logger.info(f"{log_prefix}Created WeightedRandomSampler.")
            except Exception as e:
                logger.error(f"{log_prefix}Failed to create WeightedRandomSampler: {e}. Falling back to standard shuffling.")
                train_sampler = None
        train_dataset = PatchDataset(df_train, tokenizer, config, stage=f'{log_prefix}TRAIN')
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.train.batch_size,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  num_workers=config.train.device_setting.get('num_workers', 0),
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  worker_init_fn=_seed_worker,
                                  generator=torch_generator)
        logger.info(f"{log_prefix}Train DataLoader created (Batch: {config.train.batch_size}, Sampler: {'Yes' if train_sampler else 'No'}, Shuffle: {'Yes' if train_sampler is None else 'No'}).")
    else:
        logger.warning(f"{log_prefix}Training DataFrame is empty. Train DataLoader will be None.")
    val_loader = None
    if not df_val.empty:
        val_dataset = PatchDataset(df_val, tokenizer, config, stage=f'{log_prefix}VAL')
        val_loader = DataLoader(val_dataset,
                                batch_size=config.eval.batch_size,
                                shuffle=False,
                                num_workers=config.train.device_setting.get('num_workers', 0),
                                collate_fn=collate_fn,
                                pin_memory=True,
                                worker_init_fn=_seed_worker,
                                generator=torch_generator)
        logger.info(f"{log_prefix}Validation DataLoader created (Batch: {config.eval.batch_size}).")
    else:
        logger.warning(f"{log_prefix}Validation DataFrame is empty. Validation DataLoader will be None.")
    test_loader = None
    if not df_test.empty:
        test_dataset = PatchDataset(df_test, tokenizer, config, stage=f'{log_prefix}TEST')
        test_loader = DataLoader(test_dataset,
                                 batch_size=config.eval.batch_size,
                                 shuffle=False,
                                 num_workers=config.train.device_setting.get('num_workers', 0),
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 worker_init_fn=_seed_worker,
                                 generator=torch_generator)
        logger.info(f"{log_prefix}Test DataLoader created (Batch: {config.eval.batch_size}).")
    return train_loader, val_loader, test_loader