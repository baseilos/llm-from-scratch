import re

class WordTokenizer:
    UNKNOWN_TOKEN = '<|unk|>'
    END_OF_TEXT_TOKEN = '<|endoftext|>'

    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {v: k for k, v in vocabulary.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else WordTokenizer.UNKNOWN_TOKEN for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text

    @staticmethod
    def from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_unique_tokens = sorted(set(preprocessed))
        all_unique_tokens.extend([WordTokenizer.END_OF_TEXT_TOKEN, WordTokenizer.UNKNOWN_TOKEN])
        vocabulary = {token: integer for integer, token in enumerate(all_unique_tokens)}
        return WordTokenizer(vocabulary)

