import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import librosa

import numpy as np

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Train a snoring classifier.')
parser.add_argument('--n-features', dest='n_features', type=int, default=40,
                    help='Number of features to extract for classifier.')

args = parser.parse_args()

# Data information
DATA_ROOT = 'data/'
DATA_SPLIT = 0.8
N_FEATURES = args.n_features
N_CLASSES = 2

# Training settings
N_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

wandb.init()
wandb.config.update({ 'n_features': N_FEATURES, 'learning_rate': LEARNING_RATE })

def extract_features(filepath, n_features=40):
    signal, sr = librosa.load(filepath)
    return librosa.feature.mfcc(signal, sr, n_mfcc=n_features).T

def load_dataset(data_root, n_classes, n_features=40):
    data = []
    
    for i in range(n_classes):
        class_files = os.listdir(data_root + str(i))
        
        for filename in class_files:
            filepath = data_root + str(i) + '/' + filename

            x = extract_features(filepath, n_features=n_features)
            y = np.array([i]) # Force correct input size

            data.append((x, y))

    np.random.shuffle(data)
    return data

data = load_dataset(DATA_ROOT, N_CLASSES, n_features=N_FEATURES)

# Split train and test data and create dataloaders
split_index = int(len(data) * DATA_SPLIT)
train_data = data[split_index:]
test_data = data[:split_index]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, n_features=40):
        super(Classifier, self).__init__()
        
        self.lin1 = nn.Linear(n_features, 50)
        self.lin2 = nn.Linear(50, 60)
        self.lstm = nn.LSTM(60, 30, bidirectional=True, batch_first=True)
        self.lin3 = nn.Linear(60, 1)
    
    def forward(self, x, device='cpu'):
        b_len, s_len, _ = x.shape
        
        # Initialize hidden and cell state
        hidden = (torch.zeros(2, b_len, 30).to(device),
                  torch.zeros(2, b_len, 30).to(device))
        
        # Create mind vector and run trough LSTM
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y, _ = self.lstm(y, hidden)
        
        # Get last prediction from LSTM
        y = y[:, -1, :]
        
        # Classify
        return torch.sigmoid(self.lin3(y))

def accuracy(model, test_data):
    n_cor, n_tot = 0, 0 # Track total and correct predictions
    with torch.no_grad():
        for x, y in test_data:
            # Make prediction
            y_hat = torch.round(model(x))
            
            n_tot += y.size(0)
            n_cor += (y_hat == y).sum().item()
    
    return round(100 * (n_cor / n_tot), 2)

model = Classifier(n_features=N_FEATURES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, N_EPOCHS + 1):
    for x, y in train_loader:
        y = y.float()
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Make prediction
        y_hat = model(x)
        
        # Calculate loss and gradients w.r.t. loss
        loss = F.binary_cross_entropy(y_hat, y)
        loss.backward()
        
        # Step on gradients
        optimizer.step()
        
    train_acc, test_acc = accuracy(model, train_loader), accuracy(model, test_loader)
    print(f'\rEpoch [{epoch}/{N_EPOCHS}]\t | Train Accuracy: {train_acc}%\t | Test Accuracy: {test_acc}%\t', end="")
    wandb.log({ 'epoch': epoch, 'train_acc': train_acc, 'test_acc': test_acc })
