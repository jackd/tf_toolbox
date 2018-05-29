"""
Script for creating profile trace.

Based on this blog post
https://medium.com/towards-data-science/howto-profile-tensorflow-1a49fb18073d

To use, ensure libcupti is on LD_LIBRARY_PATH:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64/
```
"""

import os
import tensorflow as tf
from tensorflow.python.client import timeline
from progress.bar import IncrementalBar


def create_profile(graph_fn, filename, skip_runs=10):
    """
    Creates a profile of running the graph defined in `graph_fn`.

    Example usage:
    ```
    def graph_fn():
        features, labels = get_inputs()
        train_op = get_train_op(features, labels)
        return train_op

    create_profile(graph_fn, 'my_profile.json')
    ```

    Open chrome, navigate to `chrome://tracing` and load `my_profile.json`.

    Note this requires the CUPTI/lib to be on your `LD_LIBRARY_PATH`, e.g.
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64/
    """
    print('Building graph...')
    graph = tf.Graph()
    with graph.as_default():
        output = graph_fn()
    path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), filename)
    print('Starting session...')
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        print('Warming up...')
        if skip_runs > 0:
            bar = IncrementalBar(max=skip_runs)
            for i in range(skip_runs):
                sess.run(output)
                bar.next()
            bar.finish()

        print('Performing profiled session run...')
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(output, options=options, run_metadata=run_metadata)

        print('Fetching timeline...')
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        print('Generating chrome trace...')
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        print('Writing trace...')
        with open(path, 'w') as f:
            f.write(chrome_trace)
    print('Stacktrace successfully written.')
    print('To view, go to `chrome://tracing` > load')
    print(path)
