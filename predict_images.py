
import argparse
import os
import random
import cPickle as pickle

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from build_dataset import get_inner_mask


from matplotlib import pyplot as plt

from basic_model import init_params_basic, forward_prop_basic
from basic_no_clip_model import init_params_basic_no_clip, forward_prop_basic_no_clip
from basic_model_sigmoid import init_params_basic_sig, forward_prop_basic_sig
from deeper_basic_model import init_params_deep_basic, forward_prop_deep_basic
from fully_connected_model import init_params_basic_FC, forward_prop_basic_FC
from encoder_decoder import init_params_enc, forward_prop_enc
from super_deep import init_params_super_deep, forward_prop_super_deep
from deep_encoder_decoder import init_params_deep_enc, forward_prop_deep_enc



parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="basic")
parser.add_argument('-r', '--restore', default=None)
parser.add_argument('-n', '--num_images', type=int, default=10)
parser.add_argument('-s', '--seed', type=int, default=10)
parser.add_argument('-v', '--version', default="train")

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    
    return X, Y

    

def random_mini_batches(X_train, Y_train, minibatch_size, seed):
    np.random.seed(seed)
    perm = np.random.permutation(X_train.shape[0])

    batches = []

    for i in range(int(X_train.shape[0]/minibatch_size)):
        x_batch = X_train[minibatch_size*i: minibatch_size*(i+1)]
        y_batch = Y_train[minibatch_size*i: minibatch_size*(i+1)]
        batches.append((x_batch, y_batch))

    return batches


def recombine_images(X,  Y):
    mask, mask_start, mask_stop = get_inner_mask()
    height = mask_stop - mask_start
    inner = np.copy(Y)
    recombined = np.copy(X)
    recombined[:, mask_start:mask_stop, mask_start:mask_stop] = inner.reshape(Y.shape[0], height, height, 3)
    return recombined

def random_image_visualization(X, Y, predictions, epoch, num_images, model, version):

    chosen_Xs = X
    chosen_Ys = Y
    chosen_preds = predictions

    orig_images = recombine_images(chosen_Xs, chosen_Ys)
    pred_images = recombine_images(chosen_Xs, chosen_preds)

    if not os.path.isdir("images/" + model + "/" + version):
        os.makedirs("images/" + model + "/" + version)

    if not os.path.isdir("images/" + model + "/" + version + "/" + str(epoch)):
        os.makedirs("images/" + model + "/" + version + "/" + str(epoch))

    for i in range(num_images):
        orig_image = orig_images[i]

        plt.imshow(orig_image, interpolation='nearest')
        plt.savefig("images/" + model + "/" + version + "/" + str(epoch) + "/orig_img_" + str(index) + '.png', bbox_inches='tight')
        plt.close()
           
        pred_image = pred_images[i] 
        plt.imshow(pred_image, interpolation='nearest')
        plt.savefig("images/" + model + "/" + version + "/" + str(epoch) + "/pred_img_" + str(index) + '.png', bbox_inches='tight')
        plt.close()



if __name__ == '__main__':


    args = parser.parse_args()


    X_train = np.load("data/32x32_CIFAR/train_imgs/X_train.npy")
    Y_train = np.load("data/32x32_CIFAR/train_imgs/Y_train.npy")
    Y_train = Y_train.reshape(Y_train.shape[0], 8*8*3)
    

    X_dev = np.load("data/32x32_CIFAR/valid_imgs/X_valid.npy")
    Y_dev = np.load("data/32x32_CIFAR/valid_imgs/Y_valid.npy")
    Y_dev = Y_dev.reshape(Y_dev.shape[0], 8*8*3)

    m, n_H0, n_W0, n_C0 = X_train.shape
    m, n_y = Y_train.shape


    ops.reset_default_graph()
    tf.set_random_seed(1)
    

    X, Y = create_placeholders(32, 32, 3, n_y)


    if args.model == 'deep':
        parameters = init_params_deep_basic()

        preds = forward_prop_deep_basic(X, parameters)

    elif args.model == 'basic':
        parameters = init_params_basic()

        preds = forward_prop_basic(X, parameters)

    elif args.model == "basicSigmoid":
        parameters = init_params_basic_sig()

        preds = forward_prop_basic_sig(X, parameters)

    elif args.model == "FC":
        parameters = init_params_basic_FC()

        preds = forward_prop_basic_FC(X, parameters)
    elif args.model == "encoder":
        parameters = init_params_enc()

        preds = forward_prop_enc(X, parameters)
    elif args.model == 'noClip':
        parameters = init_params_basic_no_clip()

        preds = forward_prop_basic_no_clip(X, parameters)
    elif args.model == 'superDeep':
        parameters = init_params_super_deep()

        preds = forward_prop_super_deep(X, parameters) 
    elif args.model == 'deepEnc':
        parameters = init_params_deep_enc()

        preds = forward_prop_deep_enc(X, parameters)            
    else:
        print("Invalid model")




    saver = tf.train.Saver()



    with tf.Session() as sess:

        saver.restore(sess, args.restore)
        epoch = int(args.restore.split("_")[1])

        
        if args.version == 'train':
            X_use = X_train
            Y_use = Y_train
        else:
            X_use = X_dev
            Y_use = Y_dev

        np.random.seed(args.seed)
        indices = np.random.choice(range(X_use.shape[0]),size=args.num_images, replace=False)

        X_use = X_use[indices]
        Y_use = Y_use[indices]

        preds = sess.run(preds, feed_dict={X:X_use, Y:Y_use})
           
        random_image_visualization(X_use, Y_use, preds, epoch, args.num_images,args.model, args.version)



