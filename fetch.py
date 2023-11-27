
import os
import numpy as np
import requests  # for fetching url
import gzip  # zip files
import hashlib  # not sure what this does but something about imports

path = "Datasets/mnist"  # Path where to save the downloaded mnist dataset


def fetch_data(url):
    """
    Fetches a dataset and saves it in Datasets/mnist
    :param url: the url of the dataset
    :return: data formatted in a numpy array
    """
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
