import torch
from sklearn.metrics import accuracy_score

class ModelTrainer:

    def __init__(self, model, train_dataloader, validation_dataloader, criterion, optimizer, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_acc += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

        return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

    def evaluate(self, model, dataloader, criterion, device):
        model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                epoch_acc += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

        return epoch_loss / len(dataloader), epoch_acc / len(dataloader)