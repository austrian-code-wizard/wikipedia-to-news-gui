from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class HighwayCNN(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(HighwayCNN, self).__init__()

        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes

        self.name = "HighwayCNN"
        self.static = static
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(len(Ks) * Co, len(Ks) * Co)
        self.gate = nn.Linear(len(Ks) * Co, len(Ks) * Co)
        self.linear = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x_gate = F.relu(self.gate(x))
        x_proj = self.sigmoid(self.projection(x))
        x = x_gate * x_proj + (1 - x_gate) * x
        x = self.linear(x)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        output = self.sigmoid(x)
        return output