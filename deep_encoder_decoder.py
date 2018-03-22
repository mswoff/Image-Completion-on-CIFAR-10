import tensorflow as tf
n_C0 = 3
filters1 = 30
filters2 = 40
filters3 = 40
filters4 = 40
encoding_size = 4*8*8*3
filters5 = 40
filters6 = 40

SIZE = 32

def init_params_deep_enc():
    W1 = tf.get_variable("W1", [5, 5, n_C0, filters1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, filters1, filters2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [5, 5, filters2, filters3], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [5, 5, filters3, filters4], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    return {"W1": W1, "W2": W2, "W3": W3, "W4": W4}

def forward_prop_deep_enc(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']


    # 32 -> 28 -> 24 -> 12 -> 6 -> 2 -> 1
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="VALID")
    A1 = tf.nn.relu(Z1)

    Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding="VALID")
    P2 = tf.nn.max_pool(Z2, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="VALID")
    A2 = tf.nn.relu(P2)


    Z3 = tf.nn.conv2d(A2, W3, strides=[1, 2, 2, 1], padding="SAME")
    A3 = tf.nn.relu(Z3)

    Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding="VALID")
    A4 = tf.nn.relu(Z4)


    flatten = tf.contrib.layers.flatten(A4)

    enc = tf.layers.dense(flatten, encoding_size, activation=tf.nn.relu)

    enc = tf.reshape(enc, [-1, 1, 1, encoding_size])


    Z5 = tf.layers.conv2d_transpose(enc, filters5, 3, strides=2, padding="SAME")
    A5 = tf.nn.relu(Z5)

    Z6 = tf.layers.conv2d_transpose(A5, filters6, 3, strides=2, padding="SAME")
    A6 = tf.nn.relu(Z6)

    Z7 = tf.layers.conv2d_transpose(A6, 3, 3, strides=2, padding="SAME")
    # A7 = tf.nn.sigmoid(Z7)
    A7 = tf.nn.relu(Z7)



    clip = tf.clip_by_value(A7, 0.0, 1.0)
    F = tf.contrib.layers.flatten(clip)

    return F