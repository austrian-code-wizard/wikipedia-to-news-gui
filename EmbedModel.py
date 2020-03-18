import numpy as np
import torch


class DataLoader:
    def __init__(self, model_class, tokenizer_class, pretrained_weights, device):
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_model = model_class.from_pretrained(pretrained_weights)
        self.device = device
        self.bert_model.to(self.device)

    def tokenize_text(self, df, max_seq):
        return [
            self.tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.sentence.values
        ]

    @staticmethod
    def pad_text(tokenized_text, max_seq):
        return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])

    def tokenize_and_pad_text(self, df, max_seq):
        tokenized_text = self.tokenize_text(df, max_seq)
        padded_text = self.pad_text(tokenized_text, max_seq)
        return torch.tensor(padded_text)

    def get_values(self, data, max_seq):
        indices = self.tokenize_and_pad_text(data, max_seq)
        self.bert_model.eval()
        x_batch = self.bert_model(indices.to(self.device))[0]
        return x_batch
