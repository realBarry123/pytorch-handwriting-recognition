import os
import cv2
import torch
import numpy as np
import requests, gzip, os, hashlib

from model import Net
from fetch import fetchData

test_data = fetchData("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
test_targets = fetchData("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
