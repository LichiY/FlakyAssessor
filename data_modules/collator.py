import torch
class PatchCollator(object):
    def __init__(self, tokenizer):
        """
        Collator object for the PatchDataset.
        Args:
            tokenizer: The tokenizer used in the dataset (needed for padding info if manual padding were used).
                       Currently not strictly needed as dataset handles padding.
        """
        super(PatchCollator, self).__init__()
    def __call__(self, batch):
        """
        Collates a batch of samples from PatchDataset.
        Args:
            batch (list[dict]): A list of dictionaries, where each dict is an output
                                from PatchDataset.__getitem__.
        Returns:
            dict: A dictionary of batched tensors ready for the model.
                  Keys: 'input_ids_flaky', 'attention_mask_flaky',
                        'input_ids_patch', 'attention_mask_patch', 'label'.
        """
        input_ids_flaky = torch.stack([sample['input_ids_flaky'] for sample in batch])
        attention_mask_flaky = torch.stack([sample['attention_mask_flaky'] for sample in batch])
        input_ids_patch = torch.stack([sample['input_ids_patch'] for sample in batch])
        attention_mask_patch = torch.stack([sample['attention_mask_patch'] for sample in batch])
        labels = torch.stack([sample['label'] for sample in batch])
        return {
            'input_ids_flaky': input_ids_flaky,
            'attention_mask_flaky': attention_mask_flaky,
            'input_ids_patch': input_ids_patch,
            'attention_mask_patch': attention_mask_patch,
            'label': labels
        }
