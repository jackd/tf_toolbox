import numpy as np
import tensorflow as tf


def train_val_changes(train_op_fn, n_steps=5):
    """
    Determine which variables change, and which don't.

    Args:
        `train_op_fn`: function mapping no inputs to a train operation.
        `n_steps`: number of times to run the resulting `train_op`.

    Returns:
        unchanged_names: list of names of tensors which did not change value.
        changed_names: list of names of tensors which did change value.

    If `unchanged_names` is not empty, there are likely unused variables which
    could possibly be removed, or your gradients are not flowing correctly.
    """
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
    """This wrapper around train_val_changes with printed output."""
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
    """
    Determine whether all update ops are running by default.

    This is helpful to determine whether other automatically created update_ops
    will automatically be run, e.g. moving average updates in batch
    normalization.

    Implemented by creating an update op, calling `train_op_fn`, then running
    the resulting `train_op` and checking if the initially created update op
    is run.

    Args:
        `train_op_fn`: function mapping no inputs to a train op

    Returns:
        None if no update ops created by `train_op_fn`
        True if the initially created update_op is run
        False if the initially created update_op is not run.

    If the initially created update_op is not run (indicated by this function
    returning `False`), consider wrapping your `train_op` with
    `tf.control_dependencies`.
    ```
    def new_train_op_fn():
        train_op = old_train_op_fn()
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ops.append(train_op)
        with tf.control_dependencies(ops):
            fixed_train_op = tf.no_op()
        return fixed_train_op
    ```
    """
    graph = tf.Graph()
    with graph.as_default():
        step = tf.Variable(
            initial_value=0, dtype=tf.int32, name='test_step', trainable=False)
        update_step = tf.assign_add(step, 1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_step)
        train_op = train_op_fn()
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if ops[0] == update_step and len(ops) == 1:
            return None

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_op)
        s = sess.run(step)

    return s == 1


def report_update_ops_run(train_op_fn):
    """This wrapper around do_update_ops_run with printing."""
    s = do_update_ops_run(train_op_fn)
    if s is None:
        print('No UPDATE_OPS created by `train_op_fn`')
    elif s:
        print('UPDATE_OPS run successfully :)')
    else:
        print('UPDATE_OPS not automatically run :(')
