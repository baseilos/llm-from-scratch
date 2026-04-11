from data_loader import CustomDataset

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

data_loader = CustomDataset.create_dataloader(raw_text, batch_size=8, nax_length=4, stride=4, shuffle=False)
data_iter = iter(data_loader)
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)