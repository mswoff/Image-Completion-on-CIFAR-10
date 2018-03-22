
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
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/32x32_CIFAR',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

parser.add_argument('-m', '--model', default="basic")
parser.add_argument('-r', '--restore', default=None)
parser.add_argument('-n', '--num_epochs', type=int, default=100)
parser.add_argument('-l', '--learning_rate', type=float, default=.0001)

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

def random_dev_image_visualization(X, Y, predictions, num_images, seed, epoch, model):
    np.random.seed(seed)
    indices = np.random.choice(range(X.shape[0]),size=num_images, replace=False)

    chosen_Xs = X[indices]
    chosen_Ys = Y[indices]
    chosen_preds = predictions[indices]

    orig_images = recombine_images(chosen_Xs, chosen_Ys)
    pred_images = recombine_images(chosen_Xs, chosen_preds)

    if not os.path.isdir("images/" + model + "/epoch" + str(epoch)):
        os.makedirs("images/" + model + "/epoch" + str(epoch))

    for i in range(num_images):
        orig_image = orig_images[i]
        index = indices[i]

        plt.imshow(orig_image, interpolation='nearest')
        plt.savefig("images/" + model + "/epoch" + str(epoch) +"/orig_img_" + str(index) + '.png', bbox_inches='tight')
        plt.close()
           
        pred_image = pred_images[i] 
        plt.imshow(pred_image, interpolation='nearest')
        plt.savefig("images/" + model + "/epoch" + str(epoch) +"/pred_img_" + str(index) + '.png', bbox_inches='tight')
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


    # Which model to train on
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

    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss, global_step=global_step)

    if args.restore == None:
        init = tf.global_variables_initializer()

    saver = tf.train.Saver()


    num_epochs = args.num_epochs
    minibatch_size = 1000



    with tf.Session() as sess:

        if args.restore != None:
            saver.restore(sess, args.restore)
            start = int(args.restore.split("_")[1])
            with open('images/' + args.model + "/vects_" + str(start) + ".p", 'rb') as file:
                train_losses, dev_losses, epochs = pickle.load(file)

            epochs = epochs + range(start, start + num_epochs)
        else:
            sess.run(init)
            start = 0

            train_losses = []
            dev_losses = []
            epochs = range(start, start + num_epochs)

        seed = 0



        for epoch in range(start, start + num_epochs):

            minibatch_cost = 0.0

            num_minibatches = int(m / minibatch_size)
            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)


            for minibatch in minibatches[:num_minibatches]:

                (minibatch_X, minibatch_Y) = minibatch

                _ , temp_cost = sess.run([optimizer, loss], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches

            dev_loss, dev_preds = sess.run([loss, preds], feed_dict={X:X_dev, Y:Y_dev})
            print("Cost after epoch %i: %f, %f" % (epoch, minibatch_cost, dev_loss))

            train_losses.append(minibatch_cost)
            dev_losses.append(dev_loss)

            if epoch % 100 == 0:
                random_dev_image_visualization(X_dev, Y_dev, dev_preds, 10, 10, epoch, args.model)
                # path = "models/" + args.model + "_" + str(epoch+1) + "_" + ".ckpt"
                # saver.save(sess, path)
                # print("Saved to " + path)

                # with open('images/' + args.model + "/vects_" + str(start + num_epochs) + ".p", 'wb') as file:
                #     pickle.dump((train_losses, dev_losses, epochs), file)


        random_dev_image_visualization(X_dev, Y_dev, dev_preds, 10, 10, epoch, args.model)
        path = "models/" + args.model + "_" + str(epoch+1) + "_" + ".ckpt"
        saver.save(sess, path)
        print("Saved to " + path)

        with open('images/' + args.model + "/vects_" + str(start + num_epochs) + ".p", 'wb') as file:
            pickle.dump((train_losses, dev_losses, epochs), file)

        

        plt.plot(epochs[5:], train_losses[5:], label="train loss")
        plt.plot(epochs[5:], dev_losses[5:], label="dev loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig("images/" + args.model + "/loss_graph" + str(start + num_epochs) + ".png")
        plt.close()




