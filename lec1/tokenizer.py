import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
print(enc.n_vocab) # almost 200k vocab size

text = "Hello, world!"
tokens = enc.encode(text)
print(tokens)

encoding_array =[13225, 11, 2375, 0]
decoded_text = enc.decode(encoding_array)
print(decoded_text)

