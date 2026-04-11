from tokenizer import Tokenizer

tokenizer = Tokenizer.from_file("data/the-verdict.txt")
text1 = "Hello Gisburn "
text2 = " the course"
text = Tokenizer.END_OF_TEXT_TOKEN.join([text1, text2])
print(text)
encoded = tokenizer.encode(text)
print("Encoded: ", encoded)
print("Decoded: ", tokenizer.decode(encoded))