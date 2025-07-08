import torch
import torch.nn as nn
from config import DistilEmbConfig

class DistillEmbSmall(nn.Module):
    
    def __init__(self, config):
        super(DistillEmbSmall, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.char_vocab_size, self.config.char_emb_size)
        self.conv1 = nn.Conv1d(self.config.num_input_chars, 128, config.kernel, stride=1)
        self.conv2 = nn.Conv1d(128, 256, config.kernel, stride=1)
        self.conv3 = nn.Conv1d(256, 384, config.kernel, stride=1)
        self.conv4 = nn.Conv1d(384, 512, 4, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.output_layer = nn.Linear(512, self.config.output_emb_size)

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.norm0 = nn.LayerNorm([self.config.num_input_chars, self.config.char_emb_size])
        self.norm1 = nn.LayerNorm([128, 30])
        self.norm2 = nn.LayerNorm([256, 13])
        self.norm3 = nn.LayerNorm([384, 4])
        self.norm4 = nn.LayerNorm(self.config.output_emb_size)
        self.output_norm = nn.LayerNorm(self.config.output_emb_size)

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
        # print(x.shape)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.output_layer(x)

        return x
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = input_ids
        assert len(x.shape) in [2, 3], "Input tensor must be of shape (B, S) or (B, S, N)"
        if len(x.shape) == 2:
            return self.output_norm(self.tanh(self.embed(x)))
        
        b, s, n = x.shape
        x = x.view(b* s, n)
        x = self.output_norm(self.tanh(self.embed(x)))
        x = x.view((b, s, -1))
        return x


class DistillEmbBase(nn.Module):
    
    def __init__(self, config):
        super(DistillEmbBase, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.char_vocab_size, self.config.char_emb_size)
        self.conv1 = nn.Conv1d(self.config.num_input_chars, 128, config.kernel, stride=1)
        self.conv2 = nn.Conv1d(128, 192, config.kernel, stride=1)
        self.conv3 = nn.Conv1d(192, 256, config.kernel, stride=1)
        self.conv4 = nn.Conv1d(256, 384, 3, stride=1)
        self.conv5 = nn.Conv1d(384, 512, 3, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.output_layer = nn.Linear(512, self.config.output_emb_size)

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.norm0 = nn.LayerNorm([self.config.num_input_chars, self.config.char_emb_size])
        self.norm1 = nn.LayerNorm([128, 30])
        self.norm2 = nn.LayerNorm([192, 13])
        self.norm3 = nn.LayerNorm([256, 4])
        self.norm4 = nn.LayerNorm([384, 4])
        self.norm5 = nn.LayerNorm(self.config.output_emb_size)
        self.output_norm = nn.LayerNorm(self.config.output_emb_size)

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

        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.output_layer(x)

        return x
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = input_ids
        assert len(x.shape) in [2, 3], "Input tensor must be of shape (B, S) or (B, S, N)"
        if len(x.shape) == 2:
            return self.output_norm(self.tanh(self.embed(x)))
        
        b, s, n = x.shape
        x = x.view(b* s, n)
        x = self.output_norm(self.tanh(self.embed(x)))
        x = x.view((b, s, -1))
        return x
    
if __name__ == "__main__":
    config = DistilEmbConfig(
        num_chars=400,
        output_emb_size=512,
        char_emb_size=64,
        num_input_chars=12,
        kernel_size=5,
        dropout=0.1
    )
    m = DistillEmbBase(config)
    import numpy as np
    x = np.random.randint(0, 400, (10, 12))
    x = torch.tensor(x, dtype=torch.int64)
    y = m(x)
    print(y.shape)