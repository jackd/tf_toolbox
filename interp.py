from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_bilinear_coords_and_factors(
        coords, name='bilinear_coords_and_factors'):
    """
    Get coordinates and factors used in bilinear interpolation.

    Usage:
    ```python
    corner_coords, factors = get_bilinear_coords_and_factors(coords)
    interped_values = gather_scale_sum(values, corner_coords, factors)
    ```
    """
    with tf.name_scope(name):
        coords = tf.convert_to_tensor(coords, tf.float32)

        shape = coords.shape.as_list()
        batched = len(shape) == 3
        n_dims = shape[-1]
        shape = tf.shape(coords)

        if batched:
            batch_size = shape[0]
            n_points = shape[1]
        else:
            n_points = shape[0]
        n_dims = coords.shape.as_list()[-1]

        di = tf.stack(tf.meshgrid(*(
            [tf.range(2)]*n_dims), indexing='ij'), axis=-1)
        di = tf.reshape(di, (-1, n_dims))

        floor_coords = tf.floor(coords)
        delta = coords - floor_coords
        floor_coords = tf.cast(floor_coords, tf.int32)

        floor_coords = tf.expand_dims(floor_coords, -2)
        corner_coords = floor_coords + di

        neg_delta = 1 - delta
        delta = tf.stack([neg_delta, delta], axis=-1)
        deltas = tf.unstack(delta, axis=-2)
        # batch-wise meshgrid - maybe map_fn instead?
        # n_dim explicit loops here, as opposed to presumably n_points?
        for i, delta in enumerate(deltas):
            deltas[i] = tf.reshape(
                delta, (-1,) + (1,)*i + (2,) + (1,)*(n_dims - i - 1))

        factors = deltas[0]
        for d in deltas[1:]:
            factors = factors*d

        if batched:
            factors = tf.reshape(
                factors, (batch_size, n_points, 2**n_dims))
        else:
            factors = tf.reshape(factors, (n_points, 2**n_dims))

    return corner_coords, factors


def fix_coords_for_gather(coords, value_rank):
    n_dim = coords.shape[-1].value
    if n_dim == value_rank:
        return coords
    elif n_dim == value_rank - 1:
        shape = tf.shape(coords)
        batch_size = shape[0]
        n_points = shape[1]
        n_dims = shape[3]
        batch_index = tf.tile(
            tf.reshape(
                tf.range(batch_size, dtype=tf.int32), (-1, 1, 1)),
            (1, n_points, 2**n_dims))
        batch_index = tf.expand_dims(batch_index, axis=-1)
        return tf.concat([batch_index, coords], axis=-1)
    else:
        raise ValueError(
            'coords must have rank %d or %d for value_rank %d, got %d'
            % (value_rank-1, value_rank, value_rank, n_dim))


def gather_scale_sum(values, coords, factors):
    """
    Gather values at coords, scale by factors and sum.

    Second stage of `bilinear_interp`.
    """
    with tf.name_scope('gather_scale_sum'):
        n_dims = len(values.shape)
        coords = fix_coords_for_gather(coords, n_dims)
        corner_vals = tf.gather_nd(values, coords)
        interped_vals = tf.reduce_sum(corner_vals*factors, axis=-1)
    return interped_vals


def _assert_shapes_consistent(grid_vals, coords):
    n_dims = coords.shape.as_list()[-1]
    batched = len(coords.shape) == 3
    if len(grid_vals.shape) != (n_dims + batched):
        raise ValueError(
            'Inconsistent shapes for interpolation. \n'
            'grid_vals: %s, coords: %s'
            % (str(grid_vals.shape), str(coords.shape)))


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

    See also:
        `get_bilinear_coords_and_factors`, `gather_scale_sum`
    """
    with tf.name_scope(name):
        grid_vals = tf.convert_to_tensor(grid_vals, tf.float32)
        coords = tf.convert_to_tensor(coords, tf.float32)
        _assert_shapes_consistent(grid_vals, coords)
        corner_coords, factors = get_bilinear_coords_and_factors(coords)
        interped_vals = gather_scale_sum(grid_vals, corner_coords, factors)

    return interped_vals
