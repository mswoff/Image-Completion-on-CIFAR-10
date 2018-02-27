
import argparse
import logging
import os
import random

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/32x32_CIFAR',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    
    return X, Y

n_C0 = 3
filters1 = 10
filters2 = 20
filters3 = 20
SIZE = 32

def init_params():
    W1 = tf.get_variable("W1", [5, 5, n_C0, filters1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, filters1, filters2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [5, 5, filters2, filters2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [1, 1, filters3, 3], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    return {"W1": W1, "W2": W2, "W3": W3, "W4": W4}

def forward_prop(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']


    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="VALID")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding="VALID")
    P2 = tf.nn.max_pool(Z2, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="VALID")
    A2 = tf.nn.relu(P2)


    Z3 = tf.nn.conv2d(A2, W3, strides=[1, 1, 1, 1], padding="VALID")
    A3 = tf.nn.relu(Z3)

    Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding="VALID")
    A4 = tf.nn.relu(Z4)
    F = tf.contrib.layers.flatten(A4)

    return F

    

def random_mini_batches(X_train, Y_train, minibatch_size, seed):
    np.random.seed(seed)
    perm = np.random.permutation(X_train.shape[0])

    batches = []

    for i in range(int(X_train.shape[0]/minibatch_size)):
        x_batch = X_train[minibatch_size*i: minibatch_size*(i+1)]
        y_batch = Y_train[minibatch_size*i: minibatch_size*(i+1)]
        batches.append((x_batch, y_batch))

    return batches


if __name__ == '__main__':


    X_train = np.load("data/32x32_CIFAR/train_imgs/X_train.npy")
    Y_train = np.load("data/32x32_CIFAR/train_imgs/Y_train.npy")

    X_dev = np.load("data/32x32_CIFAR/valid_imgs/X_valid.npy")
    Y_dev = np.load("data/32x32_CIFAR/valid_imgs/Y_valid.npy")

    m, n_H0, n_W0, n_C0 = X_train.shape
    m, n_yH, n_yW, n_C0 = Y_train.shape


    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3

    X, Y = create_placeholders(32, 32, 3, 64 * 3)

    parameters = init_params()

    preds = forward_prop(X, parameters)

    loss = tf.losses.mean_squared_error(labels=Y, predictions=preds)

    optimizer = tf.train.AdamOptimizer(learning_rate= .01).minimize(loss)

    init = tf.global_variables_initializer()



    num_epochs = 100
    minibatch_size = 1000
 

    with tf.Session() as sess:
        sess.run(init)
        

        for epoch in range(num_epochs):

            minibatch_cost = 0.0

            num_minibatches = int(m / minibatch_size)
            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                minibatch_Y = minibatch_Y.reshape(minibatch_Y.shape[0], 8*8*3)

                _ , temp_cost = sess.run([optimizer, loss], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches

            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))








