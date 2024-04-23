import pandas as pd
import cleaning as c
import torch.nn as nn   
import torch as pt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy
import tqdm



PATH = 'data'

# class BubbleClassifier():
#     def __init__(self, path):
#         super().__init__()
#         self.path = path
#         self.df = c.read_and_clean(self.path)
#         self.df = self.df.dropna()
#         self.X, self.y = conv_to_tensor(self.df.iloc[:, 4:-6], self.df.iloc[:, -1])
#         self.relu = nn.ReLU()
#         self.hidden = nn.Linear(self.X.shape[1], 1)
#         self.sigmoid = nn.Sigmoid() 
#         self.output = nn.Linear()
#         self.model = None

#     def forward(self, X): 
#         X = self.relu(self.hidden(X))
#         X = self.sigmoid(X)
#         return X

class BubbleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(16, 48)
        self.relu = nn.ReLU()
        self.output = nn.Linear(48, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 16)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 16)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(16, 16)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    
    
def conv_to_tensor(x, y):

    x = pt.tensor(x.values, dtype=pt.float32)
    y = pt.tensor(y.values, dtype=pt.float32).reshape(-1, 1)

    return x, y

import torch.optim as optim
 
def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
    n_epochs = 250   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = pt.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf  
    best_weights = None
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc



from sklearn.model_selection import StratifiedKFold, train_test_split
 

def main():

    bc = BubbleClassifier()
    deepbc = Deep()

    df = c.read_and_clean(PATH)
    df = df.dropna()
    X, y = conv_to_tensor(df.iloc[:, 4:-6], df.iloc[:, -1])


if __name__ == '__main__':
    main()