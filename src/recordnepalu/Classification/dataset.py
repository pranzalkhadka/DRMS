import torch
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, custom_tokenizer, max_length=256):
        self.texts = dataframe['data'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.custom_tokenizer = custom_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        input_ids = torch.tensor(self.custom_tokenizer.encode(text), dtype=torch.long)
        attention_mask = (input_ids != self.custom_tokenizer.vocab['<PAD>']).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }