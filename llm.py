import torch.nn

from data_loader import CustomDataset

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
data_loader, tokenizer = CustomDataset.create_dataloader(raw_text, batch_size=8, nax_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(data_loader)
inputs, target = next(data_iter)
print("Token IDs:\n", inputs)
print("Input shape:\n", inputs.shape)

# Create token embeddings (from inputs)
output_dim = 256
token_embedding_layer = torch.nn.Embedding(tokenizer.vocabulary_size(), output_dim)
token_embeddings = token_embedding_layer(inputs)

# Create position embeddings
position_embedding_layer = torch.nn.Embedding(max_length, output_dim)
position_embeddings = position_embedding_layer(torch.arange(max_length))
print(position_embeddings.shape)

# Combine token embeddings and position embeddings
input_embeddings = token_embeddings + position_embeddings
print(input_embeddings.shape)
