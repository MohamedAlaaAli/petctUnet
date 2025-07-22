import torch.nn as nn
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

class TextEmbedder(nn.Module):
    """
    Model to extract text embeddings.
    """
    def __init__(self, model_name: str = 'zzxslp/RadBERT-RoBERTa-4m'):
        """
        Initializes the text embedder model.

        Args:
            model_name (str): Name of the model on Hugging Face Hub.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config, use_safetensors=True)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad
    def forward(self, texts: list[str] | str):
        """
        Args:
            texts (List[str] or str): Input text or list of texts.
        Returns:
            last_hidden_state: Tensor of shape (B, T, H)
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # Move inputs to the same device as model
        encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}

        res = self.model(**encoded_input)
        return res.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
