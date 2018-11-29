import tensorflow as tf


def network(feature_input, labels, mode):
    """
    Creates a simple multi-layer convolutional neural network
    
    :param feature_input: 
    :param labels: 
    :param mode: 
    :return: 
    """
    filters = [32, 64, 128]
    dropout_rates = [0.2, 0.4, 0.7]
    conv_layer = feature_input

    for filter_num, dropout_rate in zip(filters, dropout_rates):
        conv_layer = conv_block(conv_layer, mode, filters=filter_num, dropout=dropout_rate)

    # Dense Layer
    pool4_flat = tf.layers.flatten(conv_layer)
    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Age Head
    age_dense = tf.layers.dense(inputs=dropout, units=1024)
    age_logits = tf.layers.dense(inputs=age_dense, units=101)

    # Gender head
    gender_dense = tf.layers.dense(inputs=dropout, units=1024)
    gender_logits = tf.layers.dense(inputs=gender_dense, units=2)

    return age_logits, gender_logits


def conv_block(input_layer, mode, filters=64, dropout=0.0):
    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

    dropout_layer = tf.layers.dropout(
        inputs=pool, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    return dropout_layer
