import torch
from torch._functorch._aot_autograd import input_output_analysis

from gpt_model import GPTModel
from bpe_tokenizer import BPETokenizer
from data_loader import CustomDataset


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

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer()
    model = GPTModel(GPT_CONFIG_124M)

    file_path = "data/the-verdict.txt"
    with open(file_path, "r", encoding="utf_8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Total characters:", total_characters)
    print("Total tokens:", total_tokens)

    train_ration = 0.90
    split_idx = int(train_ration * total_tokens)
    train_data = text_data[split_idx:]
    val_data = text_data[:split_idx]

    train_loader = CustomDataset.create_dataloader(tokenizer,
                                                   train_data,
                                                   batch_size=2,
                                                   max_length=GPT_CONFIG_124M["context_length"],
                                                   stride=GPT_CONFIG_124M["context_length"],
                                                   drop_last=True,
                                                   shuffle=True,
                                                   run_workers=0)

    val_loader = CustomDataset.create_dataloader(tokenizer,
                                                 val_data,
                                                 batch_size=2,
                                                 max_length=GPT_CONFIG_124M["context_length"],
                                                 stride=GPT_CONFIG_124M["context_length"],
                                                 drop_last=True,
                                                 shuffle=True,
                                                 run_workers=0)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Train loss:", train_loss)
    print("Val loss:", val_loss)