import os
import cv2
import numpy as np
from tqdm import tqdm
import requests, gzip, hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model import Net
from fetch import fetchData

# ==================== LOADING DATA ================================================================================

# define path to store dataset
path = 'Datasets/mnist'

# load mnist dataset from yann.lecun.com, train data is of shape (60000, 28, 28) and targets are of shape (60000)
train_data = fetchData("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
train_targets = fetchData("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
test_data = fetchData("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
test_targets = fetchData("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# show images from dataset using OpenCV
for train_image, train_target in zip(train_data, train_targets):
    train_image = cv2.resize(train_image, (400, 400))
    cv2.imshow("Image", train_image)
    # if Q button break this loop
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

batch_size_train = 64  # how many batches to split the train set into
batch_size_test = 64  # how many batches to split the test set into

# change the color of every pixel to a number between 0 and 1
train_data = np.expand_dims(train_data, axis=1) / 255.0
test_data = np.expand_dims(test_data, axis=1) / 255.0

# split data into batches of size [(batch_size, 1, 28, 28) ...] (64 batches)
train_batches = [np.array(train_data[i:i+batch_size_train]) for i in range(0, len(train_data), batch_size_train)]

# split targets into batches of size [(batch_size) ...] (64 batches)
train_target_batches = [np.array(train_targets[i:i+batch_size_train]) for i in range(0, len(train_targets), batch_size_train)]

test_batches = [np.array(test_data[i:i+batch_size_test]) for i in range(0, len(test_data), batch_size_test)]
test_target_batches = [np.array(test_targets[i:i+batch_size_test]) for i in range(0, len(test_targets), batch_size_test)]


# ==================== NETWORK STUFF ================================================================================

n_epochs = 5  # number of epochs
learning_rate = 0.001  # how quickly the network changes itself

# create network
network = Net()

# print network summary
summary(network, (1, 28, 28), device="cpu")

# optimizer: adjusts the weights of the network according to the learning rate
# higher learning rate => bigger adjustments
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# loss function: evaluates how well the network is doing
loss_function = nn.CrossEntropyLoss()


# ==================== TRAINING ================================================================================

def train(epoch):

    # set network to training mode
    network.train()

    loss_sum = 0
    # create a progress bar
    train_pbar = tqdm(zip(train_batches, train_target_batches), total=len(train_batches))
    for index, (data, target) in enumerate(train_pbar, start=1):

        # convert data to torch.FloatTensor
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        # update progress bar with loss value
        loss_sum += loss.item()
        train_pbar.set_description(f"Epoch {epoch}, loss: {loss_sum / index:.4f}")
