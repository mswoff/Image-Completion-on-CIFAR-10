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

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    
    return X, Y

def recombine_images(X,  Y):
    mask, mask_start, mask_stop = get_inner_mask()
    height = mask_stop - mask_start
    inner = np.copy(Y)
    recombined = np.copy(X)
    recombined[:, mask_start:mask_stop, mask_start:mask_stop] = inner.reshape(Y.shape[0], height, height, 3)
    return recombined

def random_image_visualization(X, Y, predictions, epoch, num_images, model, seed):
    np.random.seed(seed)
    indices = np.random.choice(range(X.shape[0]),size=num_images, replace=False)

    chosen_Xs = X[indices]
    chosen_Ys = Y[indices]
    chosen_preds = predictions[indices]

    orig_images = recombine_images(chosen_Xs, chosen_Ys)
    pred_images = recombine_images(chosen_Xs, chosen_preds)

    if not os.path.isdir("images/" + model + "/test"):
        os.makedirs("images/" + model + "/test")

    if not os.path.isdir("images/" + model + "/test/" + str(epoch)):
        os.makedirs("images/" + model + "/test/" + str(epoch))

    for i in range(num_images):
        orig_image = orig_images[i]
        index = indices[i]

        plt.imshow(orig_image, interpolation='nearest')
        plt.savefig("images/" + model + "/test/" + str(epoch) + "/orig_img_" + str(index) + '.png', bbox_inches='tight')
        plt.close()
           
        pred_image = pred_images[i] 
        plt.imshow(pred_image, interpolation='nearest')
        plt.savefig("images/" + model + "/test/" + str(epoch) + "/pred_img_" + str(index) + '.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':


    args = parser.parse_args()


    X_test = np.load("data/32x32_CIFAR/test_imgs/X_test.npy")
    Y_test = np.load("data/32x32_CIFAR/test_imgs/Y_test.npy")
    Y_test = Y_test.reshape(Y_test.shape[0], 8*8*3)
    

    m, n_H0, n_W0, n_C0 = X_test.shape
    m, n_y = Y_test.shape


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



    loss = tf.losses.mean_squared_error(labels=Y, predictions=preds)

    saver = tf.train.Saver()



    with tf.Session() as sess:

        saver.restore(sess, args.restore)
        epoch = int(args.restore.split("_")[1])



        loss, preds = sess.run([loss, preds], feed_dict={X:X_test, Y:Y_test})
        random_image_visualization(X_test, Y_test, preds, epoch, args.num_images, args.model, args.seed)
           
        print(loss)