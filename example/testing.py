#!/usr/bin/python
"""Example usage of `testing`."""
import tensorflow as tf
from tf_toolbox.testing import report_train_val_changes
from tf_toolbox.testing import report_update_ops_run


def get_inputs():
    """Dummy inputs."""
    batch_size = 8
    features = tf.random_normal(shape=(batch_size, 16), dtype=tf.float32)
    labels = tf.random_normal(shape=(batch_size,), dtype=tf.float32)
    return features, labels


def train_op_fn():
    """Spot the bug."""
    features, labels = get_inputs()
    n_hidden = 16
    x = tf.layers.dense(features, n_hidden)
    x = tf.layers.batch_normalization(
        x, training=True, scale=False)
    x = tf.nn.relu(x)
    x = tf.layers.dense(features, 1)
    x = tf.squeeze(x, axis=-1)
    loss = tf.nn.l2_loss(x - labels)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return train_op


print('train_op_fn')
report_train_val_changes(train_op_fn)


def train_op_fn_v2():
    """Better yet?"""
    features, labels = get_inputs()
    n_hidden = 16
    x = tf.layers.dense(features, n_hidden)
    x = tf.layers.batch_normalization(
        x, training=True, scale=False)
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, 1)
    x = tf.squeeze(x, axis=-1)
    loss = tf.nn.l2_loss(x - labels)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return train_op


print('train_op_fn_v2`')
report_train_val_changes(train_op_fn_v2)
report_update_ops_run(train_op_fn_v2)


def train_op_fn_v3():
    train_op = train_op_fn_v2()
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ops.append(train_op)
    with tf.control_dependencies(ops):
        train_op = tf.no_op()
    return train_op


print('train_op_fn_v3')
report_train_val_changes(train_op_fn_v3)
report_update_ops_run(train_op_fn_v3)
