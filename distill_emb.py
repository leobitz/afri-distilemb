import torch
import torch.nn as nn
from config import DistillEmbConfig
import numpy as np
from transformers import PreTrainedModel
class DistillEmbSmall(nn.Module):
    
    def __init__(self, config):
        super(DistillEmbSmall, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.char_vocab_size, 64)
        self.conv1 = nn.Conv1d(self.config.num_input_chars, 128, 5, stride=1)
        self.conv2 = nn.Conv1d(128, 256, 5, stride=1)
        self.conv3 = nn.Conv1d(256, 384, 5, stride=1)
        self.conv4 = nn.Conv1d(384, 512, 4, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.output_layer = nn.Linear(512, 512)

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.norm0 = nn.LayerNorm([self.config.num_input_chars, 64])
        self.norm1 = nn.LayerNorm([128, 30])
        self.norm2 = nn.LayerNorm([256, 13])
        self.norm3 = nn.LayerNorm([384, 4])
        self.norm4 = nn.LayerNorm(512)
        self.output_norm = nn.LayerNorm(512)

        self.dropout = nn.Dropout(config.distill_dropout)
    
    def embed(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.norm0(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm3(x)
        # print(x.shape)
        x = self.conv4(x).squeeze()
        _x = x
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.output_layer(x) + _x

        return x
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = input_ids
        assert len(x.shape) in [2, 3], "Input tensor must be of shape (B, S) or (B, S, N)"
        if len(x.shape) == 2:
            return self.tanh(self.embed(x))
        
        b, s, n = x.shape
        x = x.view(b* s, n)
        x = self.tanh(self.embed(x))
        x = x.view((b, s, -1))
        return x


class DistillEmbBase(nn.Module):
    
    def __init__(self, config):
        super(DistillEmbBase, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.char_vocab_size, 128)
        self.conv1 = nn.Conv1d(self.config.num_input_chars, 128, 5, stride=1)
        self.conv2 = nn.Conv1d(128, 256, 5, stride=1)
        self.conv3 = nn.Conv1d(256, 384, 5, stride=1)
        self.conv4 = nn.Conv1d(384, 448, 3, stride=1)
        self.conv5 = nn.Conv1d(448, 512, 3, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.output_layer = nn.Linear(512, 512)

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.norm0 = nn.LayerNorm([self.config.num_input_chars, 128])
        self.norm1 = nn.LayerNorm([128, 62])
        self.norm2 = nn.LayerNorm([256, 29])
        self.norm3 = nn.LayerNorm([384, 12])
        self.norm4 = nn.LayerNorm([448, 5])
        self.norm5 = nn.LayerNorm([512, 1])
        self.norm6 = nn.LayerNorm(512)
        self.output_norm = nn.LayerNorm(512)

        self.dropout = nn.Dropout(config.distill_dropout)
    
    def embed(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.norm0(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm4(x)

        x = self.conv5(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm5(x).flatten(1)
        _x = x

        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm6(x)
        x = self.output_layer(x) + _x

        return x
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = input_ids
        assert len(x.shape) in [2, 3], "Input tensor must be of shape (B, S) or (B, S, N)"
        if len(x.shape) == 2:
            return self.tanh(self.embed(x))
        
        b, s, n = x.shape
        x = x.view(b* s, n)
        x = self.tanh(self.embed(x))
        x = x.view((b, s, -1))
        return x

class DistillEmb(PreTrainedModel):
    def __init__(self, config: DistillEmbConfig):
        super(DistillEmb, self).__init__(config)
        if config.size == "small":
            self.encoder = DistillEmbSmall(config)
        else:
            self.encoder = DistillEmbBase(config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.encoder(input_ids, **kwargs)


if __name__ == "__main__":
    config = DistillEmbConfig(
        char_vocab_size=400,
        num_input_chars=12,
        distill_dropout=0.1,
        size="base"
    )
    m = DistillEmb(config)
    
    x = np.random.randint(0, 400, (10, 12))
    x = torch.tensor(x, dtype=torch.int64)
    y = m(x)
    print(y.shape)