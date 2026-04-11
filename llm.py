from bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer()
text1 = "Hello Gisburn "
text2 = " the course"
text = BPETokenizer.END_OF_TEXT_TOKEN.join([text1, text2])
print(text)
encoded = tokenizer.encode(text)
print("Encoded: ", encoded)
print("Decoded: ", tokenizer.decode(encoded))

