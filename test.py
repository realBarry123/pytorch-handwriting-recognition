import os
import cv2
import torch
import numpy as np
from random import randint

from model import Net
from fetch import fetchData

test_data = fetchData("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
test_targets = fetchData("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# output path
model_path = 'Model/06_pytorch_introduction'

# construct network and load weights
network = Net()
network.load_state_dict(torch.load("Models/network.pkl"))
network.eval()  # set to evaluation mode


index = randint(0, 9999)

test_image = test_data[index]
test_target = test_targets[index]

# normalize image and convert to tensor
inference_image = torch.from_numpy(test_image).float() / 255.0
inference_image = inference_image.unsqueeze(0).unsqueeze(0)

# predict
output = network(inference_image)
pred = output.argmax(dim=1, keepdim=True)
prediction = str(pred.item())

test_image = cv2.resize(test_image, (400, 400))
cv2.imshow(prediction, test_image)
cv2.waitKey()
