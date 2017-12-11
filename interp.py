import tensorflow as tf


def bilinear_interp(grid_vals, coords, name='bilinear_interp'):
    """
    Perform bilinear interpolation to approximate grid_vals between indices.

    Args:
        grid_vals: values at grid coordinates, (n_x, n_y, ...) (n_dims of them)
            or (batch_size, n_x, n_y, ...).
        coords: coordinate values to be interpolated at, (n_points, n_dims) or
            (batch_size, n_points, n_dims).

    Returns:
        (batch_size, n_points) or (n_points,) tensor of interpolated grid
            values.
    """
    with tf.name_scope(name):
        grid_vals = tf.convert_to_tensor(grid_vals, tf.float32)
        coords = tf.convert_to_tensor(coords, tf.float32)

        shape = coords.shape.as_list()
        batched = len(shape) == 3
        n_dims = shape[-1]
        shape = tf.shape(coords)

        if batched:
            batch_size = shape[0]
            n_points = shape[1]
        else:
            n_points = tf.shape(coords)[0]
        n_dims = coords.shape.as_list()[-1]
        if len(grid_vals.shape) != (n_dims + batched):
            raise ValueError(
                'Inconsistent shapes for interpolation. \n'
                'grid_vals: %s, coords: %s'
                % (str(grid_vals.shape), str(coords.shape)))

        di = tf.stack(tf.meshgrid(*(
            [tf.range(2)]*n_dims), indexing='ij'), axis=-1)
        di = tf.reshape(di, (-1, n_dims))

        floor_coords = tf.floor(coords)
        delta = coords - floor_coords
        floor_coords = tf.cast(floor_coords, tf.int32)

        all_coords = tf.expand_dims(floor_coords, -2) + di

        if batched:
            batch_index = tf.tile(
                tf.reshape(
                    tf.range(batch_size, dtype=tf.int32), (-1, 1, 1)),
                (1, n_points, 2**n_dims))
            batch_index = tf.expand_dims(batch_index, axis=-1)
            all_coords = tf.concat([batch_index, all_coords], axis=-1)

        neg_delta = 1 - delta
        delta = tf.stack([neg_delta, delta], axis=-1)
        deltas = tf.unstack(delta, axis=-2)
        # batch-wise meshgrid - maybe map_fn instead?
        # n_dim explicit loops here, as opposed to presumably n_points?
        for i, delta in enumerate(deltas):
            deltas[i] = tf.reshape(
                delta, (-1,) + (1,)*i + (2,) + (1,)*(n_dims - i - 1))

        delta_prod = deltas[0]
        for d in deltas[1:]:
            delta_prod = delta_prod*d

        if batched:
            delta_prod = tf.reshape(
                delta_prod, (batch_size, n_points, 2**n_dims))
        else:
            delta_prod = tf.reshape(delta_prod, (n_points, 2**n_dims))

        corner_vals = tf.gather_nd(grid_vals, all_coords)
        vals = tf.reduce_sum(corner_vals*delta_prod, axis=-1)
    return vals
