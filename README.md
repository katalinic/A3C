# A3C
TF A3C, more of a prototypical nature.

The repository contains: `a2c.py`, which is A2C using [`tf.scan()`][tf-scan] for rollouts.

`a3c.py` extends `a2c.py` to multiple workers, and is A3C based on threading, intended for a single machine.

Both `a2c.py` and `a3c.py` use a slightly modified version of PyProcess (see `pyprocess.py`). The environments
are placed into subprocesses with communication done via pipes, and embedded into the TensorFlow graph
via [`tf.py_func()`][tf-pyfunc]. The style of the rollouts (worker-environment) interaction was largely
inspired by DeepMind's [IMPALA](https://arxiv.org/abs/1802.01561).

Dist folder contains the original, non-optimised, placeholder based distributed TF version of the code.

[tf-scan]: https://www.tensorflow.org/api_docs/python/tf/scan
[tf-pyfunc]: https://www.tensorflow.org/api_docs/python/tf/py_func
