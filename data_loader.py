import torch
from torch.utils.data import Dataset, DataLoader, dataloader
from bpe_tokenizer import BPETokenizer


class CustomDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # Generate chunks of input and target tokens by sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.target_ids[item]

    @staticmethod
    def create_dataloader(tokenizer, txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, run_workers=0):
        dataset = CustomDataset(txt, tokenizer, max_length, stride)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=run_workers)
        return data_loader
