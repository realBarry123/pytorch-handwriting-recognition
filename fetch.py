import os, requests, gzip, hashlib
import numpy as np

path = 'Datasets/mnist' # Path where to save the downloaded mnist dataset


def fetchData(url):
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