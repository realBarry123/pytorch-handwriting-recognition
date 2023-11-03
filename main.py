import os
import cv2
import numpy as np
from tqdm import tqdm
import requests, gzip, os, hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

# from model import Net

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