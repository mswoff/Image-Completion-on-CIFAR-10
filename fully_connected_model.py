import tensorflow as tf
n_C0 = 3
filters1 = 10
filters2 = 20
filters3 = 30
filters4 = 40
FC_size = 4*8*8*3
SIZE = 32

def init_params_basic_FC():
    W1 = tf.get_variable("W1", [5, 5, n_C0, filters1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, filters1, filters2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [5, 5, filters2, filters3], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3, 3, filters3, filters4], initializer=tf.contrib.layers.xavier_initializer(seed=0))


    return {"W1": W1, "W2": W2, "W3": W3, "W4": W4}

def forward_prop_basic_FC(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']


    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="VALID")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding="VALID")
    P2 = tf.nn.max_pool(Z2, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="VALID")
    A2 = tf.nn.relu(P2)


    Z3 = tf.nn.conv2d(A2, W3, strides=[1, 1, 1, 1], padding="SAME")
    A3 = tf.nn.relu(Z3)

    Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding="SAME")
    A4 = tf.nn.relu(Z4)

    F = tf.contrib.layers.flatten(A4)
    FC1 = tf.layers.dense(F, FC_size, activation=tf.nn.relu)
    # FC2 = tf.layers.dense(FC1, 8*8*3, activation=tf.nn.sigmoid)
    FC2 = tf.layers.dense(FC1, 8*8*3, activation=tf.nn.relu)
    clip = tf.clip_by_value(FC2, 0.0, 1.0)


    return clip