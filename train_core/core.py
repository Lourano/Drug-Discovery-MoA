import torch.nn as nn
import torch

class Dataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.features[item, :], dtype = torch.float),
            "y": torch.tensor(self.targets[item, :], dtype = torch.float)}
    

class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    @staticmethod
    def loss_(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)
    
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        
        for data in data_loader:
            self.optimizer.zero_grad()
            features = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(features)
            loss = self.loss_(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)
    
    
    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        
        for data in data_loader:
            self.optimizer.zero_grad()
            features = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(features)
            loss = self.loss_(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)
        

class FullyConnectedModel(nn.Module):
    def __init__(self, num_features, num_targets, num_layers, hidden_size, dropout):
        super().__init__()
        layers = []
        
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(num_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                nn.PReLU()
                
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                nn.PReLU()
                
        layers.append(nn.Linear(hidden_size, num_targets))
        self.model = nn.Sequential(*layers)
        
    
    def forward(self, x):
        x = self.model(x)
        return x
    