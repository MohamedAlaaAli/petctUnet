import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel


class TextEmbedder(nn.Module):
    """
        Model to exact text embedding.
    """
    def __init__(self, model_name:str = 'zzxslp/RadBERT-RoBERTa-4m'):
        """
            initializes the text embedder model.

            Args:
                model_name (str): Name of the model as found on Hugging Face Hub.

        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
    
    def forward(self, text:str):
        """
            Args:
                text (str): Input text.
            Returns:
                Text Embedding as tensors.
        """
        encoded_input = self.tokenizer(text, return_tensors='pt')
        _, emb = self.model(**encoded_input)
        return emb