#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_toolbox.interp import bilinear_interp

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()


def example1d():
    grid = [10, 11, 14]
    coords = [[0], [0.5], [1.2]]
    interp_vals = bilinear_interp(grid, coords)
    print('1D example:')
    print(interp_vals)


def example3d():
    coords = tf.constant([
        # [0.99, 0.2, 0.3],
        # [1.01, 0.2, 0.3],
        # [0.5, 0.6, 0.7],
        # [0.5, 0.4, 1.5],
        # [1, 1, 1],
        # [1, 0.99, 1],
        # [1, 1, 0.99],
        # [1, 0.99, 0.99],
        # [0.99, 0.99, 0.99],
        [1, 1, 1],
        [0, 1, 1],
        ], dtype=tf.float32)
    grid = tf.constant(
        [[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=tf.float32)
    grid = tf.stack([grid, grid+100, grid+200], axis=0)
    interp_vals = bilinear_interp(grid, coords)
    print('3D example:')
    print(interp_vals)


def example3d_batched():
    grid = tf.constant(
        [[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=tf.float32)
    grid = tf.stack([grid, grid+100, grid+200, grid+300])
    coords = tf.constant([
        [1, 1, 1],
        [0, 1, 1],
        ], dtype=tf.float32)
    grid = tf.stack([grid, grid + 1000])
    coords = tf.stack([coords, coords])

    interp_vals = bilinear_interp(grid, coords)
    print('3D batched example:')
    print(interp_vals)


def test_batched(n_dims=3):
    import numpy as np
    from scipy.ndimage import map_coordinates
    batch_size = 7
    n_samples = 13
    grid_size = np.random.randint(5, 10, size=n_dims)
    grid = np.random.randn(batch_size, *grid_size).astype(np.float32)
    coords = np.random.rand(batch_size, n_samples, n_dims)
    coords *= grid_size
    coords = coords.astype(np.int32)
    gt = []
    for (i, d) in zip(coords, grid):
        gt.append(map_coordinates(d, i.T, order=2))
    gt = np.stack(gt, axis=0)
    actual = np.array(bilinear_interp(grid, coords))
    print(np.max(np.abs(actual - gt)))
    print(np.max(np.abs(gt)))


# example1d()
# example3d()
# example3d_batched()
test_batched(3)
test_batched(4)
test_batched(5)
