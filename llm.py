import os
import sys

venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
venv_dir = os.path.dirname(os.path.dirname(venv_python))
if os.path.exists(venv_python) and os.path.realpath(sys.prefix) != os.path.realpath(venv_dir):
    os.execv(venv_python, [venv_python, __file__, *sys.argv[1:]])

import torch
from gpt_model import GPTModel
from bpe_tokenizer import BPETokenizer

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decoded = tokenizer.decode(flat.tolist())
    return decoded

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "gkv_bias": False,
}

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = BPETokenizer()
    start_context = "Every effort moves you"

    token_ids = generate_text_simple(model, text_to_token(start_context, tokenizer), 10, GPT_CONFIG_124M["context_length"])
    print("Output text: ", token_ids_to_text(token_ids, tokenizer))
