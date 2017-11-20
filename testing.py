import numpy as np
import tensorflow as tf


def train_val_changes(train_op_fn, n_steps=5):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        train_op = train_op_fn()
        trainable_vars = tf.trainable_variables()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        vals = sess.run(trainable_vars)
        for i in range(n_steps):
            sess.run(train_op)
        new_vals = sess.run(trainable_vars)
        names = [v.name for v in trainable_vars]
        unchanged_names = []
        changed_names = []
        for name, val, new_val in zip(names, vals, new_vals):
            if np.all(val == new_val):
                unchanged_names.append(name)
            else:
                changed_names.append(name)
        return unchanged_names, changed_names


def report_train_val_changes(train_op_fn, steps=5):
    unchanged_names, changed_names = train_val_changes(train_op_fn, steps)
    n_unchanged = len(unchanged_names)
    n_changed = len(changed_names)
    n_total = n_unchanged + n_changed
    if len(unchanged_names) == 0:
        print('All trainable variables changed :)')
    else:
        print('%d / %d training variables unchanged'
              % (n_unchanged, n_total))
        print('Changed vars:')
        for name in changed_names:
            print(name)
        print('Unchanged vars:')
        for name in unchanged_names:
            print(name)


def do_update_ops_run(train_op_fn):
    graph = tf.Graph()
    with graph.as_default():
        step = tf.Variable(initial_value=0, dtype=tf.int32, name='test_step')
        update_step = tf.assign_add(step, 1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_step)
        train_op = train_op_fn()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_op)
        s = sess.run(step)

    return s == 1
