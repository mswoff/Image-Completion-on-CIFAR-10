import tensorflow as tf
n_C0 = 3
filters1 = 40
filters2 = 40
filters3 = 40
filters4 = 60
filters5 = 60
filters6 = 60
SIZE = 32

def init_params_super_deep():
    W1 = tf.get_variable("W1", [7, 7, n_C0, filters1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [7, 7, filters1, filters2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [7, 7, filters2, filters3], initializer=tf.contrib.layers.xavier_initializer(seed=2))
    W4 = tf.get_variable("W4", [7, 7, filters3, filters4], initializer=tf.contrib.layers.xavier_initializer(seed=3))
    W5 = tf.get_variable("W5", [5, 5, filters4, filters5], initializer=tf.contrib.layers.xavier_initializer(seed=4))

    W6 = tf.get_variable("W6", [5, 5, filters5, filters6], initializer=tf.contrib.layers.xavier_initializer(seed=5))
    W7 = tf.get_variable("W7", [4, 4, filters6, 8*8*3], initializer=tf.contrib.layers.xavier_initializer(seed=6 ))

    return {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6, "W7": W7} #, "W8": W8, "W9": W9, "W10": W10

def forward_prop_super_deep(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    W7 = parameters['W7']


    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)


    Z3 = tf.nn.conv2d(A2, W3, strides=[1, 1, 1, 1], padding="VALID")
    A3 = tf.nn.relu(Z3)

    Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding="VALID")
    A4 = tf.nn.relu(Z4)

    Z5 = tf.nn.conv2d(A4, W5, strides=[1, 1, 1, 1], padding="VALID")
    P5 = tf.nn.max_pool(Z5, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="VALID")
    A5 = tf.nn.relu(P5)

    Z6 = tf.nn.conv2d(A5, W6, strides=[1, 1, 1, 1], padding="VALID")
    A6 = tf.nn.relu(Z6)

    Z7 = tf.nn.conv2d(A6, W7, strides=[1, 1, 1, 1], padding="VALID")
    A7 = tf.nn.relu(Z7)


    clip = tf.clip_by_value(A7, 0.0, 1.0)
    F = tf.contrib.layers.flatten(clip)

    return F