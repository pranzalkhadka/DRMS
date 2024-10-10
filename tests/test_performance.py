import sys
import os
import time
import unittest
import json
from PIL import Image
import pytesseract
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.classification import CustomLSTMClassifier, load_tokenizer, predict
from nepalikit.preprocessing import TextProcessor
from nepalikit.manage_stopwords import load_stopwords
from nepalikit.tokenization import Tokenizer

with open('test_images.json', 'r') as f:  
    test_images = json.load(f)

vocab_file = '../app/tokenizer_vocab.json'  
model_path = '../app/nepali_text_classification_model.pth'  
stopwords_path = '../app/NepaliStopWords' 
num_classes = 4
vocab_size = 31159
max_length = 256

tokenizer = Tokenizer()

def ocr_exec(model, filepath):
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image, lang=model)
    return text 


class CustomTokenizer:
    def __init__(self, tokenizer, vocab=None):
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab else {}

    def encode(self, text, max_length):
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        token_ids = token_ids[:max_length] + [self.vocab['<PAD>']] * (max_length - len(token_ids))
        return token_ids

tokenizer, vocab = load_tokenizer(vocab_file)
custom_tokenizer = CustomTokenizer(tokenizer, vocab=vocab)

model = CustomLSTMClassifier(vocab_size=vocab_size, embedding_dim=25, hidden_dim=8, output_dim=num_classes)
model.load_state_dict(torch.load(model_path, weights_only=True))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

stopwords = load_stopwords(stopwords_path)
processor = TextProcessor()

class TestOCRPerformance(unittest.TestCase):

    def test_total_process_time(self):
        for test_image in test_images:
            image_path = test_image["path"]
            expected_time_limit = 5.0 
            
            start_time = time.time()
            
            # Extract text using OCR
            extracted_text = ocr_exec('nep-fuse-2', image_path)
            
            # Predict the label using the model
            predicted_label = predict(extracted_text, custom_tokenizer, model, max_length, device, processor, stopwords)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.assertLessEqual(total_time, expected_time_limit, f"Total process took too long for {image_path}")

if __name__ == '__main__':
    unittest.main()