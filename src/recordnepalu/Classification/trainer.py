from Data_ingestion import download_and_extract_data
from Data_transformation import Preprocessor
from Vocab import CustomTokenizer
from dataset import CustomTextDataset
from train import ModelTrainer
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from model import CustomLSTMClassifier
from nepalikit.tokenization import Tokenizer

tokenizer = Tokenizer()
num_epochs = 15
max_length = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_pipeline():
    # Downloading and Extracting data
    download_and_extract_data()

    # Pre processing
    csv_path = "/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/NepaliText.csv"
    stopwords_path = '/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/NepaliStopWords'
    output_dir = "/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification"

    initial_dataset = pd.read_csv(csv_path)
    if 'paras' in initial_dataset.columns:
        initial_dataset = initial_dataset.rename(columns={'paras': 'data'})

    texts = initial_dataset['data'].tolist()

    preprocessor = Preprocessor() 
    num_classes = preprocessor.transform_data(csv_path, stopwords_path, output_dir)

    # Building vocabulary
    custom_tokenizer = CustomTokenizer(tokenizer)
    custom_tokenizer.build_vocab(texts)

    vocab_size = len(custom_tokenizer.vocab)

    # Creating dataloaders
    train_dataset = CustomTextDataset(pd.read_csv("/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/preprocessed_train.csv"), tokenizer, custom_tokenizer, max_length)
    validation_dataset = CustomTextDataset(pd.read_csv("/home/pranjal/Downloads/Document-Management-of-Nepali-Papers/dataset/Classification/preprocessed_val.csv"), tokenizer, custom_tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32)

    # Defining the model
    model = CustomLSTMClassifier(vocab_size=vocab_size, embedding_dim=50, hidden_dim=32, output_dim=num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = ModelTrainer(model, train_dataloader, validation_dataloader, criterion, optimizer, device)

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = trainer.evaluate(model, validation_dataloader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs} - '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} - '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

if __name__ == "__main__":
    run_pipeline()
    