import os
import cv2
import numpy as np
from tqdm import tqdm
import requests, gzip, os, hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model import Net

# ===== LOADING DATA =====

# define path to store dataset
path = 'Datasets/mnist'


def fetch(url):
    if os.path.exists(path) is False:
        os.makedirs(path)

    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


# load mnist dataset from yann.lecun.com, train data is of shape (60000, 28, 28) and targets are of shape (60000)
train_data = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
train_targets = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
test_data = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
test_targets = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# show images from dataset using OpenCV
for train_image, train_target in zip(train_data, train_targets):
    train_image = cv2.resize(train_image, (400, 400))
    cv2.imshow("Image", train_image)
    # if Q button break this loop
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


# define training hyperparameters
# we will need these later

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

# ===== NETWORK STUFF =====

n_epochs = 5  # number of epochs
learning_rate = 0.001  # how quickly the network changes itself

# create network
network = Net()

# uncomment to print network summary
summary(network, (1, 28, 28), device="cpu")

# define loss function and optimizer
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
