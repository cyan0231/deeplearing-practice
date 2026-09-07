"""学习版 Transformer：可学习位置编码，编码器—解码器架构。

仅包含编码器、解码器及其基础模块，依赖 PyTorch。
src_valid_len、tgt_valid_len 表示各自输入的非 padding 长度，均须大于 0。
解码器接收已经准备好的输入编号，例如 <bos> 后接目标词前缀。
输入为右侧 padding 的词元编号，序列长度不能超过 max_len。
"""

import math
import torch
from torch import nn


def mask_softmax(scores, valid_len):
    # scores: (batch*head_num, query_num, key_num)
    # valid_len: (batch*head_num,)
    positions = torch.arange(scores.shape[-1], device=scores.device)
    mask = positions.reshape(1, 1, -1) >= valid_len.reshape(-1, 1, 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


class attention_module(nn.Module):
    def __init__(self, query_size, key_size, value_size,
                 attention_hidden, head_num):
        super().__init__()
        assert attention_hidden % head_num == 0
        self.head_num = head_num
        self.head_size = attention_hidden // head_num
        self.w_q = nn.Linear(query_size, attention_hidden)
        self.w_k = nn.Linear(key_size, attention_hidden)
        self.w_v = nn.Linear(value_size, attention_hidden)
        self.w_o = nn.Linear(attention_hidden, attention_hidden)

    def split_head(self, X):
        batch, step, _ = X.shape
        X = X.reshape(batch, step, self.head_num, self.head_size)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, step, self.head_size)

    def merge_head(self, X):
        batch = X.shape[0] // self.head_num
        step = X.shape[1]
        X = X.reshape(batch, self.head_num, step, self.head_size)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(batch, step, -1)

    def forward(self, query, key, value, valid_len, causal=False):
        Q = self.split_head(self.w_q(query))
        K = self.split_head(self.w_k(key))
        V = self.split_head(self.w_v(value))

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_size)

        if causal:
            # Q、K 来自同一个完整目标前缀，不使用 KV 缓存。
            assert Q.shape[1] == K.shape[1]
            causal_mask = torch.triu(
                torch.ones(Q.shape[1], K.shape[1],
                           dtype=torch.bool, device=Q.device),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        valid_len = valid_len.repeat_interleave(self.head_num)
        weights = mask_softmax(scores, valid_len)
        output = torch.bmm(weights, V)
        return self.w_o(self.merge_head(output))


class PositionalEncoding(nn.Module):
    """位置编号查表，得到可学习的位置向量。"""

    def __init__(self, d_model, dropout, max_len=1000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        positions = torch.arange(X.shape[1], device=X.device)
        P = self.embedding(positions)  # (step, d_model)
        return self.dropout(X + P)


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, head_num, dropout):
        super().__init__()
        self.attention = attention_module(
            d_model, d_model, d_model, d_model, head_num
        )
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, X, valid_len):
        A = self.attention(X, X, X, valid_len)
        Y = self.addnorm1(X, A)
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, head_num,
                 num_layers, dropout, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, ffn_hidden, head_num, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids, valid_len):
        X = self.embedding(token_ids) * math.sqrt(self.d_model)
        X = self.pos_encoding(X)
        for block in self.blocks:
            X = block(X, valid_len)
        return X  # (batch, src_step, d_model)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, head_num, dropout):
        super().__init__()
        self.self_attention = attention_module(
            d_model, d_model, d_model, d_model, head_num
        )
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attention = attention_module(
            d_model, d_model, d_model, d_model, head_num
        )
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, X, enc_output, src_valid_len, tgt_valid_len):
        A = self.self_attention(X, X, X, tgt_valid_len, causal=True)
        Y = self.addnorm1(X, A)
        B = self.cross_attention(Y, enc_output, enc_output, src_valid_len)
        Z = self.addnorm2(Y, B)
        return self.addnorm3(Z, self.ffn(Z))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, head_num,
                 num_layers, dropout, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, ffn_hidden, head_num, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, enc_output, src_valid_len, tgt_valid_len):
        X = self.embedding(token_ids) * math.sqrt(self.d_model)
        X = self.pos_encoding(X)
        for block in self.blocks:
            X = block(X, enc_output, src_valid_len, tgt_valid_len)
        return self.linear(X)  # (batch, tgt_step, tgt_vocab_size)

