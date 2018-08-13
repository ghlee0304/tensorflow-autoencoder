import os
import sys
import urllib.request
import gzip
import numpy as np


def print_download_progress(count, block_size, total_size):
    decimals = 1
    format_str = "{0:." + str(decimals) + "f}"
    bar_length = 100
    pct_complete = format_str.format((float(count * block_size) / total_size) * 100)
    total = int(total_size / block_size) + 1
    filled_length = int(round(bar_length * count / total))
    if float(pct_complete) > 100.:
        pct_complete = "100"
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r |%s| %s%s ' % (bar, pct_complete, '%')),
    if pct_complete == 1.0:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_and_load_mnist(save_path, as_image=False, seed =0, fraction_of_validation=0.2):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        if not os.path.exists(save_path + file_name):
            print("\n>>> Download " + file_name + " : ")
            file_path, _ = urllib.request.urlretrieve(url=data_url + file_name, filename=save_path + file_name,
                                                      reporthook=print_download_progress)
        else:
            print(">>> {} data has apparently already been downloaded".format(file_name))

    with gzip.open(save_path + 'train-images-idx3-ubyte.gz') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * 60000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data
        if as_image == True:
            x_train = data.reshape(60000, 28, 28, 1)
        else:
            x_train = data.reshape(60000, 784)

    with gzip.open(save_path + 'train-labels-idx1-ubyte.gz') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(60000)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data
        y_train = np.expand_dims(data, 1)

    np.random.seed(seed)
    mask = np.random.permutation(len(x_train))
    x_train = x_train[mask]
    y_train = y_train[mask]

    ntrain = int(len(x_train) * (1-fraction_of_validation))
    nvalidation = int(len(x_train) - ntrain)
    x_validation = x_train[:nvalidation]
    y_validation = y_train[:nvalidation]
    x_train = x_train[nvalidation:]
    y_train = y_train[nvalidation:]

    with gzip.open(save_path + 't10k-images-idx3-ubyte.gz') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * 10000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data
        if as_image == True:
            x_test = data.reshape(10000, 28, 28, 1)
        else:
            x_test = data.reshape(10000, 784)

    with gzip.open(save_path + 't10k-labels-idx1-ubyte.gz') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(10000)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data
        y_test = np.expand_dims(data, 1)

    return {"train_data":x_train/255., "train_target":y_train,
            "validation_data":x_validation/255., "validation_target":y_validation,
            "test_data":x_test/255., "test_target":y_test}