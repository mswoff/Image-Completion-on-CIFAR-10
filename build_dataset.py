"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from model.cifar_to_images import load_test_data
from model.cifar_to_images import load_training_data
# import matplotlib.pyplot as plt



mask_size = 4
SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/CIFAR-10', help="Directory with the dataset")
parser.add_argument('--output_dir', default='data/32x32_CIFAR', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    

    mask = np.zeros((SIZE, SIZE, 3))
    mask_start = int(SIZE/2 - mask_size)
    mask_stop = int(SIZE/2 + mask_size)
    mask[mask_start:mask_stop, mask_start:mask_stop, :] = 1


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



    # img.save('my.png')
    # img.show()

    # with open(file, 'rb') as fo:
    #     dict = pickle.load(fo, encoding='bytes')


    # Define the data directories
    # train_data_dir = os.path.join(args.data_dir, 'train_imgs')
    # test_data_dir = os.path.join(args.data_dir, 'test_imgs')

    # # Get the filenames in each directory (train and test)
    # filenames = os.listdir(train_data_dir)
    # filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    # test_filenames = os.listdir(test_data_dir)
    # test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # print(filenames)
    # print(test_filenames)

    # # Split the images in 'train_signs' into 80% train and 20% dev
    # # Make sure to always shuffle with a fixed seed so that the split is reproducible
    # random.seed(230)
    # filenames.sort()
    # random.shuffle(filenames)

    # split = int(0.8 * len(filenames))
    # train_filenames = filenames[:split]
    # dev_filenames = filenames[split:]

    # filenames = {'train': train_filenames,
    #              'dev': dev_filenames,
    #              'test': test_filenames}

    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    # else:
    #     print("Warning: output dir {} already exists".format(args.output_dir))

    # # Preprocess train, dev and test
    # for split in ['train', 'dev', 'test']:
    #     output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
    #     if not os.path.exists(output_dir_split):
    #         os.mkdir(output_dir_split)
    #     else:
    #         print("Warning: dir {} already exists".format(output_dir_split))

    #     print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
    #     for filename in tqdm(filenames[split]):
    #         resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
