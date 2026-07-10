import torch
import torch.nn as nn

import bpe_tokenizer

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "gkv_bias": False,
}


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.dropout_embedding = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["num_layers"])])
        self.final_norm = LayerNorm(config["embedding_dim"])
        self.output_head = nn.Linear(config["embedding_dim"], config["vocab_size"])

    def forward(self, in_ids):
        barch_size, seq_length = in_ids.shape
        token_embeddings = self.token_embedding(in_ids)
        position_embeddings = self.position_embedding(torch.arange(seq_length, device=in_ids.device))
        x = token_embeddings + position_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (var + self.eps).sqrt()
        return normalized


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"] * 4, config["embedding_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
            gkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut
        return x


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", output.shape)
