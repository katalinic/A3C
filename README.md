# A3C
TF A3C.

The repository contains an implementation of A3C.

The environments are placed into subprocesses with communication done via pipes, and embedded into the TensorFlow graph
via [`tf.py_func()`][tf-pyfunc]. The style of the rollouts (agent-environment interaction using [`tf.scan()`][tf-scan])
was largely inspired by DeepMind's [IMPALA](https://arxiv.org/abs/1802.01561).

[tf-scan]: https://www.tensorflow.org/api_docs/python/tf/scan
[tf-pyfunc]: https://www.tensorflow.org/api_docs/python/tf/py_func
