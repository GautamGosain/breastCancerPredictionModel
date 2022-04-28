#1 ) Design model (input,output size, forward pass)
#2) Construct loss and optimizer
#3) Training loop
#   - forward pass : compute prediction
#   -backward pass : gradients
#   -update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0)data
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)  # 20% test sets   #type of X_,Y_ double

#scale
sc = StandardScaler()   # features will have zero mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32)) # convert to tensors
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)

#1) model
#f=wx+b , sigmoid(f)

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)             # 1 output features

    def  forward(self, X):
        Y_pred = torch.sigmoid(self.linear(X))
        return Y_pred

model = LogisticRegression(n_features)

#2)loss and optimizer
learningRate = 0.01
nIters = 1000
loss = nn.BCELoss()        #binary cross entropy loss/ log loss
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

#3) training loop
for iteration in range(nIters):
    #forward pass
    Ypred = model(X_train)
    #loss calc
    Loss = loss(Ypred, Y_train)
    #backward pass
    Loss.backward()
    #gradient update
    optimizer.step()
    # set gradient zero
    optimizer.zero_grad()
    [w,b] = model.parameters()
    print(f'iteration{iteration} : w = {w} , b = {b.item()}, loss = {Loss.item()}')     #printing all parameters and loss

#testing
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()        #number less than 0.5 to 0. more than 0.5 to 1
    acc = y_predicted_cls.eq(Y_test).sum()/float(Y_test.shape[0])                      # eq --- equal function       implementation of unit function
    print(f'accuracy = {acc}')