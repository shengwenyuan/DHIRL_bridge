
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IntentionNet(nn.Module):
    def __init__(self, phi_dim, num_latents, hidden_dim=128):
        super(IntentionNet, self).__init__()
        self.fc1 = nn.Linear(phi_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_latents)
        self._non_linearity = F.relu

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class StatesRNN(nn.Module):
    def __init__(self, num_states, num_actions, num_latents, hidden_dim=128, rnn_hidden_dim=128, num_layers=1, dropout=0.1):
        super(StatesRNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers

        self.state_embed = nn.Embedding(num_states, hidden_dim)
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # self.rnn = nn.LSTM(
        #     input_size=hidden_dim,
        #     hidden_size=rnn_hidden_dim,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )
        
        self.output_proj = nn.Linear(rnn_hidden_dim, num_latents)

    def forward(self, bs, ba, mask=None, total_length=None):
        state_embeds = self.state_embed(bs)   # (B, T, hidden_dim)
        action_embeds = self.action_embed(ba) # (B, T, hidden_dim)
        x = state_embeds + action_embeds       # (B, T, hidden_dim)
        if mask is not None:
            lengths = mask.sum(dim=1)
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn_out_packed, _ = self.rnn(x_packed)
            rnn_out, _ = pad_packed_sequence(rnn_out_packed, batch_first=True, total_length=total_length)  # (B, T_max, rnn_hidden_dim)
        else:
            rnn_out, _ = self.rnn(x)
        logits = self.output_proj(rnn_out)           # (B, T, num_latents)

        return logits
    

class IntentionTransformer(nn.Module):
    def __init__(self,
                 num_states,
                 num_actions,
                 num_latents,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.state_embed = nn.Embedding(num_states, d_model)
        self.action_embed = nn.Embedding(num_actions, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_latents)

    def forward(self, bs, ba, mask=None, total_length=None):
        # bs: (B, T), ba: (B, T)
        state_embeds = self.state_embed(bs)   # (B, T, d_model)
        action_embeds = self.action_embed(ba) # (B, T, d_model)
        x = state_embeds + action_embeds       # (B, T, d_model)
        x = self.pos_encoding(x)          # add positional encoding
        if mask is not None:
            padding_mask = ~mask
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)           # (B, T, d_model)

        logits = self.fc_out(x)           # (B, T, num_latents)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)