import re
import torch
import torch.nn as nn
import json
from nepalikit.tokenization import Tokenizer
from nepalikit.preprocessing import TextProcessor
from nepalikit.manage_stopwords import load_stopwords

class CustomLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super(CustomLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        pooled_output = torch.mean(lstm_out, dim=1)
        dropped_out = self.dropout(pooled_output)
        return self.fc(dropped_out)

def load_tokenizer(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as vocab_file:
        vocab = json.load(vocab_file)
    return Tokenizer(), vocab

def preprocess_text(text, processor, stopwords):
    text = processor.remove_html_tags(text)
    text = processor.remove_special_characters(text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r'[\d०१२३४५६७८९]', '', text)
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    filtered_words = [word for word in filtered_words if len(word) > 2]
    return ' '.join(filtered_words)

def predict(text, tokenizer, model, max_length, device, processor, stopwords):
    text = preprocess_text(text, processor, stopwords)
    model.eval()
    with torch.no_grad():
        encoded_input = tokenizer.encode(text, max_length)
        input_ids = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = (input_ids != tokenizer.vocab['<PAD>']).long().to(device)

        outputs = model(input_ids, attention_mask)
        _, predicted_class = torch.max(outputs, 1)

        return predicted_class.item()