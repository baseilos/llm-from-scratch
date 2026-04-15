import tiktoken

class BPETokenizer:
    END_OF_TEXT_TOKEN = '<|endoftext|>'

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={BPETokenizer.END_OF_TEXT_TOKEN})

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocabulary_size(self):
        return self.tokenizer.n_vocab
