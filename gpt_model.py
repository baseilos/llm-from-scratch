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

    def forward(self, x):
        return x

if __name__ == "__main__":
    tokenizer = bpe_tokenizer.BPETokenizer()
    batch = []
    text1 = "Every effort moves out"
    text2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim = 0)
    print(batch)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

