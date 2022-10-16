#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN, GRU and LSTM PyTorch Example - Classify MNIST digits


Reference-
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
"""


# Specify GPU to be used-
# %env CUDA_DEVICE_ORDER = PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES = 0


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


print(f"torch version: {torch.__version__}")

# Check if there are multiple devices (i.e., GPU cards)-
print(f"Number of GPU(s) available = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch does not have access to GPU")

# Device configuration-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device is {device}')


# Hyper-parameters-
num_epochs = 10
batch_size = 256
learning_rate = 0.001


# Load MNIST dataset-
train_dataset = torchvision.datasets.MNIST(
    root = '../Downloads/.data/', train = True,
    transform = transforms.ToTensor(), download = True
)

test_dataset = torchvision.datasets.MNIST(
    root = '../Downloads/.data/', train = False,
    transform = transforms.ToTensor()
)

# Sanity check-
print(f"train dataset shape: {list(train_dataset.data.size())}"
      f" & test dataset shape: {list(test_dataset.data.size())}"
      )

# Create training and testing dataloaders-
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset, batch_size = batch_size,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset, batch_size = batch_size,
    shuffle = False
)


def count_trainable_params(model):
    # Count number of layer-wise parameters and total parameters-
    tot_params = 0
    for param in model.parameters():
        # print(f"layer.shape = {param.shape} has {param.nelement()} parameters")
        tot_params += param.nelement()
    return tot_params




# Define RNN architecture
class RNN(nn.Module):
    def __init__(
            self, inp_size,
            hidden_size, num_layers,
            num_classes, type = 'rnn'
            ):
        super(RNN, self).__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.type = type
        
        if self.type == 'rnn':
            # Define an RNN layer-
            self.rnn = nn.RNN(
                input_size = inp_size, hidden_size = hidden_size,
                num_layers = num_layers, batch_first = True,
                dropout = 0, bidirectional = False
                )
        elif self.type == 'gru':
            # Define a GRU layer-
            self.gru = nn.GRU(
                input_size = inp_size, hidden_size = hidden_size,
                num_layers = num_layers, batch_first = True,
                dropout = 0, bidirectional = False
                )
        elif self.type == 'lstm':
            # Define an LSTM layer-
            self.lstm = nn.LSTM(
                input_size = inp_size, hidden_size = hidden_size,
                num_layers = num_layers, batch_first = True,
                dropout = 0, bidirectional = False
                )
        else:
            print("Invalid input! Valid inputs are: gru, lstm or rnn.")
        
        # Define a linear/fc layer-
        self.lin_layer = nn.Linear(
            in_features = hidden_size, out_features = num_classes
            )
    
    
    def forward(self, x):
        # RNN requires two inputs-
        # input: (batch_size, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch_size, hidden_size)
        
        # Initialize initial hidden state-
        h_0 = torch.zeros(
            self.num_layers, x.size(0),
            self.hidden_size
            ).to(device)
        
        if self.type == 'rnn':
            out = self.rnn(x, h_0)
        elif self.type == 'gru':
            out = self.gru(x, h_0)
        elif self.type == 'lstm':
            # Initialize initial cell state (only for LSTM)-
            c_0 = torch.zeros(
                self.num_layers, x.size(0),
                self.hidden_size
                ).to(device)
            
            # For LSTM-
            out, h_n = self.lstm(x, (h_0, c_0))
        
        # out, h_n = self.rnn(x, h_0)
        # out: (batch_size, seq_len, num_directions * hidden_size)
        # containing the output features (h_t) from the last layer of the
        # RNN, for each t.
        # out: (batch_size, 28, 128)
        
        # h_n: (num_directions * num_layers, batch_soze, hidden_size)
        # containing the final hidden state for each element in the batch.
        
        # Decode hidden state only of last time step 't'-
        out = out[:, -1, :]
        # -1 is for the last time step
        
        out = self.lin_layer(out)
        return out


# Define hyper-parameters-
input_size = 28
sequence_length = 28
hidden_size = 128

# We are stacking two RNNs together. The 2nd RNN accepts as input the
# output from 1st RNN-
num_layers = 2

num_classes = 10

# Initiaize an instance of RNN architecture-
model_RNN = RNN(
    inp_size = input_size, hidden_size = hidden_size,
    num_layers = num_layers, num_classes = num_classes,
    type = 'lstm'
    ).to(device)

print(f"RNN model has {count_trainable_params(model_RNN)} trainable parameters")

# Define cost function and optimizer-
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_RNN.parameters(), lr = learning_rate)

'''
# Get one batch of training data-
x, y = next(iter(train_loader))
x = x.squeeze().to(device)
y = y.to(device)

x.shape
# torch.Size([256, 28, 28])

# pred = model_RNN(x)

# pred.shape, y.shape
# (torch.Size([256, 10]), torch.Size([256]))

# criterion(pred, y)
'''


def train_one_epoch(model, dataloader, dataset):
    
    # Place model to device-
    model.to(device)
    
    # Enable training mode-
    model.train()
    
    # Initialize variables to keep track of 3 losses-
    running_final_loss = 0.0
    
    for i, data in tqdm(
        enumerate(dataloader),
        total = int(len(dataset) / dataloader.batch_size)
        ):
      
        x = data[0]
        y = data[1]
        
        # Push to 'device'-
        x = x.to(device)
        y = y.to(device)
        
        # Resize input to be 3D-
        x = x.squeeze()
        
        # Empty accumulated gradients-
        optimizer.zero_grad()
        
        # Perform forward propagation-
        pred = model(x)
        
        # Compute loss-
        final_loss = criterion(pred, y)
        
        # Update losses-
        running_final_loss += final_loss.item()
        
        # Compute gradients wrt total loss-
        final_loss.backward()
        
        # Perform gradient descent-
        optimizer.step()
    
    # Compute losses as float values-
    train_loss = running_final_loss / len(dataloader.dataset)
    
    return train_loss


def validate_one_epoch(model, dataloader, dataset):
    
    # Place model to device-
    model.to(device)
    
    # Enable evaluation mode-
    model.eval()
    
    running_final_loss = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader),
            total = int(len(test_dataset) / dataloader.batch_size)
        ):
            
            x_v = data[0]
            y_v = data[1]
            
            # Push data points to 'device'-
            x_v = x_v.to(device)
            y_v = y_v.to(device)
            
            # Resize input to be 3D-
            x_v = x_v.squeeze()
            
            # Perform forward propagation-
            pred = model(x_v)
            
            # Compute loss-
            final_loss = criterion(pred, y_v)
        
            # Update losses-
            running_final_loss += final_loss.item()
            
            
    val_loss = running_final_loss / len(dataloader.dataset)
    
    return val_loss


def compute_accuracy(model, dataloader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dataloader:
            # images = images.reshape(-1, sequence_length, input_size).to(device)
            images = images.squeeze().to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    return 100.0 * n_correct / n_samples


'''
train_loss = train_one_epoch(
    model = model_RNN, dataloader = train_loader,
    dataset = train_dataset
    )

test_loss = validate_one_epoch(
    model = model_RNN, dataloader = test_loader,
    dataset = test_dataset
    )

# print(f"train loss = {train_loss:.4f} & test loss = {test_loss:.4f}")
'''


# Python dict to contain training metrics-
train_history = {}

# Initialize parameters for Early Stopping manual implementation-
best_val_loss = 10000
# loc_patience = 0

# User input parameters for Early Stopping in manual implementation-
# minimum_delta = 0.001
# patience = 3


for epoch in range(1, num_epochs + 1):
    '''
    # Manual early stopping implementation-
    if loc_patience >= patience:
        print("\n'EarlyStopping' called!\n")
        break
    '''
    
    # Train model for 1 epoch-
    train_loss = train_one_epoch(
        model = model_RNN, dataloader = train_loader,
        dataset = train_dataset
        )

    test_loss = validate_one_epoch(
        model = model_RNN, dataloader = test_loader,
        dataset = test_dataset
        )
    
    test_acc = compute_accuracy(model = model_RNN, dataloader = test_loader)
    
    # Store model performance metrics in Python3 dict-
    train_history[epoch] = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    
    print(f"Epoch = {epoch}; train loss = {train_loss:.4f}",
          f", test loss = {test_loss:.4f} & test acc = "
          f"{test_acc:.2f}%"
          )
    
    
    # Code for manual Early Stopping:
    if (test_loss < best_val_loss):
    # (np.abs(val_epoch_loss - best_val_loss) >= minimum_delta):

        # update 'best_val_loss' variable to lowest loss encountered so far-
        best_val_loss = test_loss
        
        # reset 'loc_patience' variable-
        # loc_patience = 0

        print(f"Saving model with lowest test loss = {test_loss:.4f}\n")
        
        # Save trained model with 'best' validation accuracy-
        torch.save(model_RNN.state_dict(), "RNN_MNIST_best_model.pth")
        
    '''
    else:  # there is no improvement in monitored metric 'val_loss'
        loc_patience += 1  # number of epochs without any improvement
    '''


# Save training history as pickle file-
with open("RNN_MNIST_training_history.pkl", "wb") as file:
    pickle.dump(train_history, file)


'''
# Visualize trainig and testing losses-
plt.figure(figsize = (9, 8))
plt.plot([train_history[k]['train_loss'] for k in train_history.keys()], label = 'train loss')
plt.plot([train_history[k]['test_loss'] for k in train_history.keys()], label = 'test loss')
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("RNN-LSTM MNIST Training Visualization")
plt.show()

# Visualize test accuracy-
plt.figure(figsize = (9, 8))
plt.plot([train_history[k]['test_acc'] for k in train_history.keys()], label = 'test loss')
plt.xlabel("epochs")
plt.ylabel("%")
plt.title("RNN-LSTM MNIST Testing Accuracy % Visualization")
plt.show()
'''

del model_RNN


# Load 'best' trained weights from before-
trained_model = RNN(
    inp_size = input_size, hidden_size = hidden_size,
    num_layers = num_layers, num_classes = num_classes,
    type = 'lstm'
    ).to(device)

if device == 'cpu':
    trained_model.load_state_dict(torch.load('RNN_MNIST_best_model.pth'), map_location = torch.device('cpu'))
else:
    trained_model.load_state_dict(torch.load('RNN_MNIST_best_model.pth'))


# Compute test accuracy usng 'best' trained model-
test_acc = compute_accuracy(model = trained_model, dataloader = test_loader)

print(f"Test accuracy using 'best' (LSTM) RNN parameters = {test_acc:.2f}%")
# Test accuracy using 'best' (LSTM) RNN parameters = 98.68%

