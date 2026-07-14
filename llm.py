import torch
from plotter import plot_loss

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


def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, 50, context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.shape.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch + 1}, Step {global_step:06d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen


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
    model.to(device)

    file_path = "data/the-verdict.txt"
    with open(file_path, "r", encoding="utf_8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Total characters:", total_characters)
    print("Total tokens:", total_tokens)

    train_ration = 0.90
    split_idx = int(train_ration * total_characters)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    print("Data sizes: ", len(train_data), len(val_data))

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
                                                 drop_last=False,
                                                 shuffle=False,
                                                 run_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,
                                                               num_epochs, 5, 5, "Every effort moves you", tokenizer)
    epoch_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_loss(epoch_tensor, tokens_seen, train_losses, val_losses)
