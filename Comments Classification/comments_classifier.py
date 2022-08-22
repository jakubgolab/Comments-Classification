import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split

# Data

data = pd.read_csv("./data/cleaned_data.csv")

all_words = []
unique = []
bag_of_words = []

stemmer = LancasterStemmer()
for comment in data["Text"].values:
    comment = str(comment)
    words = comment.split()
    for word in words:
        w = stemmer.stem(word)
        all_words.append(w)
        
unique = np.unique(all_words)
        
# Creating bag of words

for comment in data["Text"].values:
    comment = str(comment)
    comment = [stemmer.stem(w) for w in comment.split()]
    bag = []
    for i, word in enumerate(unique):
        if word in comment:
            bag.append(1)
        else:
            bag.append(0)
    bag_of_words.append(bag)
    
# creating datasets

X = np.array(bag_of_words, dtype=np.uint8)
y = np.array(data["Labels"].values, dtype=np.uint8).reshape(-1, 1)

# print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

class CommentsDataSet(Dataset):
    def __init__(self, X, y, transformations = None):
        super().__init__()
        self.X = X
        self.y = y
        self.transformations = transformations
        self.n_samples = X.shape[0]
        
    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        
        if self.transformations:
            sample = self.transformations(sample)
            
        return sample
    
    def __len__(self):
        return self.n_samples
    

class ToTensor:
    def __call__(self, sample):
        var, target = sample
        return torch.from_numpy(var), torch.from_numpy(target)
    

dataset_train = CommentsDataSet(X_train, y_train, transformations=ToTensor())
dataset_test = CommentsDataSet(X_test, y_test, transformations=ToTensor())

train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=4, shuffle=False)


# Model creation

class ANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[64, 128]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        out = self.fc1(X.float())
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return self.sigmoid(out)
    
    
input_size = X_train.shape[1]
output_size = 1
learning_rate = 0.1
epochs = 200

model = ANN(input_size, output_size)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
loss_statistics = []

for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        
        #forward
        y_pred = model(x)
        loss = criterion(y_pred.float(), y.float())
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%4 == 0:
            print(f'Epoch: {epoch+1} / {epochs}   loss = {loss.item():.4f}')
            loss_statistics.append(loss.item())
            
# Testing

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
    for i, (x, y) in enumerate(test_loader):
        output = model(x)
        n_samples += len(y)
        n_correct += (torch.round(output) == y).sum().item()
    

    print("Testing Accuracy: ", 100.0 * n_correct / n_samples)