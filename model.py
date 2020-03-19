import torch
import pandas as pd
from EmbedModel import DataLoader
import transformers


# setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Analyzer:
    def __init__(self, embeddings="bert-base-uncased", model='2020-03-16T05:54:11--acc--0.938.pt'):
        self.model = torch.load(f'static/{model}', map_location=torch.device(device))
        self.embeddings = DataLoader(transformers.BertModel, transformers.BertTokenizer, embeddings, device)

    def analyze(self, text):
        text = text.split('. ')
        text = [t for t in text if len(t) > 0]
        text = [t.strip() for t in text]
        df = pd.DataFrame(text)
        df.columns = ["sentence"]
        result = self.embeddings.get_values(df, 40)
        torch.set_grad_enabled(False)
        result = self.model(result)
        _, y_pred = torch.max(result, 1)
        return text, list(y_pred.numpy())
