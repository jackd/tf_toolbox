Various utilities for tensorflow including profiling, testing, record writing and more.

### Setup
Clone this repository and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_toolbox.git
export PYTHON=$PYTHONPATH:/path/to/parent_dir
```

For profiling, you'll also need to modify your `LD_LIBRARY_PATH`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64/
```

### Usage
#### Profiling
Ensure you've modified your `LD_LIBRARY_PATH` as above, then:
```
import tensorflow as tf
from tf_toolbox.profile import create_profile

def graph_fn():
    # basic classifier
    features, labels = get_inputs()
    logits = get_logits(features)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return train_op


create_profile(graph_fn, './profiles/example.json')
```
Open chrome, navigate to `chrome://tracing`, hit `load` and navigate to `./profiles/example.json`.

#### Testing
We provide two basic tests: `train_val_changes` and `do_update_ops_run`. They come with printing version `report_train_val_changes` and `report_update_ops_run`, but you'll probably want to write wrappers for your own testing framework. See `report`ing mechanism in the [example](example/testing_example.py).

#### Interpolation
Bilinear interpolation. Not exactly testing/profiling... but see the [example](example/inter_example.py).

#### Data
Utilities for writing `tfrecords`.
