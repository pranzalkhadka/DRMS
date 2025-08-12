import torch
# import sys
# import os
# sys.path.append(os.path.join(os.getcwd(), 'Classification'))

from classification import CustomLSTMClassifier, load_tokenizer, predict

from nepalikit.preprocessing import TextProcessor
from nepalikit.manage_stopwords import load_stopwords
from nepalikit.tokenization import Tokenizer


# extracted_text = "राजश्व न्यायाधिकरणमा रिक्त पदहरूका कार्यालयका कर्मचारीहरू पुनरावेदन अदालत अदालतको क्षेत्र जिल्ला जिल्ला अदालतमा रिक्त राजपत्र अनड्रित प्रथम श्रणी त्राहेक कर्मचारीहरू कार्यरत कर्मचारीहरू नियमहरू प्रारम्भ हुँदाका बखत सेवामा कार्यरत राजपत्र अन्रित उपनियम जुनसुकै लेखिएको पूर्ति प्रतिशत"
#Construct the paths relative to the current working directory
# vocab_file = os.path.join(os.getcwd(), 'Classification\Resources', 'tokenizer_vocab.json')
# model_path = os.path.join(os.getcwd(), 'Classification\Resources', 'nepali_text_classification_model.pth')
# stopwords_path = os.path.join(os.getcwd(), 'Classification\Resources', 'NepaliStopWords')

# vocab_file = "app/tokenizer_vocab.json"
# model_path = "app/nepali_text_classification_model.pth"
# stopwords_path = "app/NepaliStopWords"


vocab_file = "tokenizer_vocab.json"
model_path = "nepali_text_classification_model.pth"
stopwords_path = "NepaliStopWords"

# vocab_file = os.path.join('app', 'tokenizer_vocab.json')
# model_path = os.path.join('app', 'nepali_text_classification_model.pth')

num_classes = 4
vocab_size = 31362
max_length = 128

tokenizer = Tokenizer()

class CustomTokenizer:
    def __init__(self, tokenizer, vocab=None):
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab else {}

    def build_vocab(self, texts):
        index = 2
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        for text in texts:
            tokens = self.tokenizer.tokenize(text, level = 'word')
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1

    def encode(self, text, max_length):
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        token_ids = token_ids[:max_length] + [self.vocab['<PAD>']] * (max_length - len(token_ids))
        return token_ids




def predict_category(input_text):
   
    tokenizer, vocab = load_tokenizer(vocab_file)
    custom_tokenizer = CustomTokenizer(tokenizer, vocab=vocab)

    model = CustomLSTMClassifier(vocab_size=vocab_size, embedding_dim=25, hidden_dim=4, output_dim=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    stopwords = load_stopwords(stopwords_path)
    processor = TextProcessor()
    predicted_label = predict(input_text, custom_tokenizer, model, max_length, device, processor, stopwords)
    class_names = ['Education', 'ID', 'Policy', 'Press_Release']
    predicted_class_name = class_names[predicted_label]
    return predicted_class_name