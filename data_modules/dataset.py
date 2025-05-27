import torch
from torch.utils.data.dataset import Dataset
import helper.logger as logger
import pandas as pd
class PatchDataset(Dataset):
    def __init__(self, dataframe, tokenizer, config, stage='TRAIN'):
        """
        Dataset for flaky test patch validation.
        Args:
            dataframe (pd.DataFrame): DataFrame containing the data for this split (train/val/test).
            tokenizer: Pre-trained tokenizer (e.g., from UniXcoder).
            config: Configure object with settings like max_length, column names.
            stage (str): 'TRAIN', 'VAL', or 'TEST' for logging purposes.
        """
        super(PatchDataset, self).__init__()
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = config.model.max_length
        self.config = config
        self.stage = stage
        self.flaky_col = config.data.flaky_col
        self.patch_col = config.data.patch_col
        self.label_col = config.data.label_col
        required_cols = [self.flaky_col, self.patch_col, self.label_col]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame for stage {stage}.")
        logger.info(f"{stage} Dataset created with {len(self.df)} samples.")
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    def __getitem__(self, index):
        """
        Gets a single sample (flaky code, patch code, label) and tokenizes the code.
        Args:
            index (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing tokenized inputs and the label.
                  Keys: 'input_ids_flaky', 'attention_mask_flaky',
                        'input_ids_patch', 'attention_mask_patch', 'label'.
        """
        if index >= self.__len__():
            raise IndexError(f"Index {index} out of bounds for dataset with size {self.__len__()}")
        sample_row = self.df.iloc[index]
        flaky_code = str(sample_row[self.flaky_col])
        patch_code = str(sample_row[self.patch_col])
        label = int(sample_row[self.label_col])
        flaky_encodings = self.tokenizer(
            flaky_code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        patch_encodings = self.tokenizer(
            patch_code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids_flaky = flaky_encodings['input_ids'].squeeze(0)
        attention_mask_flaky = flaky_encodings['attention_mask'].squeeze(0)
        input_ids_patch = patch_encodings['input_ids'].squeeze(0)
        attention_mask_patch = patch_encodings['attention_mask'].squeeze(0)
        return {
            'input_ids_flaky': input_ids_flaky,
            'attention_mask_flaky': attention_mask_flaky,
            'input_ids_patch': input_ids_patch,
            'attention_mask_patch': attention_mask_patch,
            'label': torch.tensor(label, dtype=torch.long)
        }
