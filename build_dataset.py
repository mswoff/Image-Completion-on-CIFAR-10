

import random
import os
import numpy as np
import tensorflow as tf

# from PIL import Image
from cifar_to_images import load_test_data
from cifar_to_images import load_training_data



mask_size = 4
SIZE = 32


def get_inner_mask():
    mask = np.zeros((SIZE, SIZE, 3))
    mask_start = int(SIZE/2 - mask_size)
    mask_stop = int(SIZE/2 + mask_size)
    mask[mask_start:mask_stop, mask_start:mask_stop, :] = 1
    return mask, mask_start, mask_stop
    

if __name__ == '__main__':
    
    mask, mask_start, mask_stop = get_inner_mask()

    test_imgs = load_test_data()

    Y_test = test_imgs * mask
    Y_test = Y_test[:, mask_start:mask_stop, mask_start:mask_stop, :]
    X_test = test_imgs * (1 - mask)

    perm = np.random.permutation(10000)
    Y_valid = Y_test[perm[:5000]]
    X_valid = X_test[perm[:5000]]

    np.save("data/32x32_CIFAR/valid_imgs/Y_valid", Y_valid)
    np.save("data/32x32_CIFAR/valid_imgs/X_valid", X_valid)

    Y_test = Y_test[perm[5000:]]
    X_test = X_test[perm[5000:]]

    np.save("data/32x32_CIFAR/test_imgs/Y_test", Y_test)
    np.save("data/32x32_CIFAR/test_imgs/X_test", X_test)



    train_imgs = load_training_data()

    Y_train = train_imgs * mask
    Y_train = Y_train[:, mask_start:mask_stop, mask_start:mask_stop, :]

    X_train = train_imgs * (1 - mask)

    np.save("data/32x32_CIFAR/train_imgs/Y_train", Y_train)
    np.save("data/32x32_CIFAR/train_imgs/X_train", X_train)



