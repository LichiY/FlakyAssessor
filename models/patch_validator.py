import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import helper.logger as logger
class PatchValidator(nn.Module):
    def __init__(self, config):
        super(PatchValidator, self).__init__()
        self.config = config
        encoder_name = config.model.encoder_name
        logger.info(f"Loading encoder model configuration: {encoder_name}")
        try:
            encoder_config = AutoConfig.from_pretrained(encoder_name)
        except Exception as e:
            logger.error(f"Failed to load configuration for {encoder_name}: {e}")
            raise
        if hasattr(encoder_config, 'return_dict') and not encoder_config.return_dict:
            logger.warning(f"Configuration for {encoder_name} had return_dict=False. Setting to True for model loading.")
            encoder_config.return_dict = True
        elif not hasattr(encoder_config, 'return_dict'):
            logger.warning(f"Configuration for {encoder_name} lacks return_dict attribute. Adding return_dict=True.")
            encoder_config.return_dict = True
        logger.info(f"Loading encoder model weights: {encoder_name}")
        try:
            self.encoder = AutoModel.from_pretrained(
                encoder_name,
                config=encoder_config
            )
        except Exception as e:
            logger.error(f"Failed to load model weights for {encoder_name}: {e}")
            raise
        encoder_hidden_size = encoder_config.hidden_size
        dnn_layers = config.model.get('dnn_layers', [512, 128])
        dropout_rate = config.model.get('dropout_rate', 0.2)
        dnn_input_size = encoder_hidden_size + encoder_hidden_size + 1 + 1
        layers = []
        input_dim = dnn_input_size
        for hidden_dim in dnn_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*layers)
        logger.info(f"Classifier DNN initialized with input size {dnn_input_size}, layers {dnn_layers}, output size 1.")
        logger.info(f"Model structure: {self}")
    def _get_embedding(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.pooler_output
        return embedding
    def forward(self, input_ids_flaky, attention_mask_flaky, input_ids_patch, attention_mask_patch):
        embedding_flaky = self._get_embedding(input_ids_flaky, attention_mask_flaky)
        embedding_patch = self._get_embedding(input_ids_patch, attention_mask_patch)
        feature_diff = embedding_flaky - embedding_patch
        feature_prod = embedding_flaky * embedding_patch
        feature_euclidean = torch.norm(feature_diff, p=2, dim=1, keepdim=True)
        feature_cosine = F.cosine_similarity(embedding_flaky, embedding_patch, dim=1).unsqueeze(-1)
        combined_features = torch.cat((feature_diff, feature_prod, feature_euclidean, feature_cosine), dim=1)
        logits = self.classifier(combined_features)
        return logits