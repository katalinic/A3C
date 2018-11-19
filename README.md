# A3C
TF A3C, more of a prototypical nature.

Dist folder contains the original, distributed TF version of the code.

The repository also contains: `a2c.py`, which is A2C using [`tf.scan()`][tf-scan] for rollouts with `tfenv.py` wrapping the environments in TensorFlow using [`tf.py_func()`][tf-pyfunc].

`a3c_threaded.py` extends `a2c.py` to multiple workers, and is A3C based on threading, intended for a single machine.

[tf-scan]: https://www.tensorflow.org/api_docs/python/tf/scan
[tf-pyfunc]: https://www.tensorflow.org/api_docs/python/tf/py_func 
