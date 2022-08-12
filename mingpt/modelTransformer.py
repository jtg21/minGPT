import math
import logging
from turtle import forward

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt import model

class CasualSelfAttentionNoMask(model.CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class DecoderBlock(model.Block):
    def __init__(self, config):
        super().__init__(config)
        self.attn_nm = CasualSelfAttentionNoMask(config)
    
    def forward(self, x, enc_x):
        x = x + self.attn(self.ln1(x))
        x = x + self.attn_nm(enc_x)
        x = x + self.mlp(self.ln2(x))
        return x

class EncoderBlock(model.Block):
    def __init__(self, config):
        super().__init__(config)
        self.attn = CasualSelfAttentionNoMask(config)

class Encoder(model.GPT):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Override GPT masked attention blocks with normal attention blocks
        self.blocks = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_layer)])
     
class Decoder(model.GPT):
    def __init__(self, config) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)])
    
    def forward(self, idx, enc_x, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x, enc_x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, idx, targets=None):
        enc_x = self.encoder(idx, targets)
        x = self.decoder(idx, enc_x, targets)
        return x

