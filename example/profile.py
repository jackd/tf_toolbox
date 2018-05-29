import os
import tensorflow as tf
from tf_toolbox.profile import create_profile


def get_inputs():
    """Dummy inputs."""
    batch_size = 8
    features = tf.random_normal(shape=(batch_size, 16), dtype=tf.float32)
    labels = tf.random_normal(shape=(batch_size,), dtype=tf.float32)
    return features, labels


def train_op_fn():
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


filename = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'example_profile.json')

create_profile(train_op_fn, filename)
