import os
import cv2
import torch
import numpy as np
import requests, gzip, os, hashlib

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


# loop over test images
for test_image, test_target in zip(test_data, test_targets):

    # normalize image and convert to tensor
    inference_image = torch.from_numpy(test_image).float() / 255.0
    inference_image = inference_image.unsqueeze(0).unsqueeze(0)

    # predict
    output = network(inference_image)
    pred = output.argmax(dim=1, keepdim=True)
    prediction = str(pred.item())

    test_image = cv2.resize(test_image, (400, 400))
    cv2.imshow(prediction, test_image)
    key = cv2.waitKey(0)
    if key == ord('q'):  # break on q key
        break

    cv2.destroyAllWindows()