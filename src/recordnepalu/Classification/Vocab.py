from nepalikit.tokenization import Tokenizer

tokenizer = Tokenizer()
max_length = 256 

class CustomTokenizer:

    def __init__(self, tokenizer, vocab=None):
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab else {}

    def build_vocab(self, texts):
        index = 2 
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        for text in texts:
            tokens = self.tokenizer.tokenize(text, level='word')
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1

    def encode(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        token_ids = token_ids[:max_length] + [self.vocab['<PAD>']] * (max_length - len(token_ids))
        return token_ids

if __name__ == "__main__":
    import pandas as pd
    initial_dataset = pd.read_csv("/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/NepaliText.csv")

    texts = initial_dataset['data'].tolist()
    
    custom_tokenizer = CustomTokenizer(tokenizer)
    custom_tokenizer.build_vocab(texts)
    
    encoded_texts = [custom_tokenizer.encode(text) for text in texts]
    print(f"Vocabulary size: {len(custom_tokenizer.vocab)}")