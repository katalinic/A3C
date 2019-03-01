"""PyProcess with minor modifications.

Adaptation of PyProcess as available at:
https://github.com/deepmind/scalable_agent/blob/master/py_process.py

Changes mostly intended for use with OpenAI Gym.
"""

import multiprocessing

import tensorflow as tf

nest = tf.contrib.framework.nest


class _TFProxy:
    def __init__(self, type_):
        self._type = type_

    def __getattr__(self, name):
        def call(*args):
            # We assume here that the env being wrapped has a dict of specs
            # for each method that might be called here. Each spec should
            # be a tuple of dtypes and shapes.
            dtypes, shapes = self._type.specs[name]

            def py_call(*args):
                try:
                    self._out.send(args)
                    result = self._out.recv()
                    if isinstance(result, Exception):
                        raise result
                    if result is not None:
                        return result
                except Exception as e:
                    if isinstance(e, IOError):
                        raise StopIteration()  # Clean exit.
                    else:
                        raise

            result = tf.py_func(py_call, (name,) + tuple(args), dtypes,
                                name=name)

            if isinstance(result, tf.Tensor):  # This was Operation.
                result.set_shape(shapes)
                return result

            for t, shape in zip(result, shapes):
                t.set_shape(shape)
            return result
        return call

    def _start(self):
        self._out, in_ = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=self._worker_fn,
            args=(self._type, in_))
        self._process.start()

    def _close(self):
        try:
            self._out.send(None)
            self._out.close()
        except IOError:
            pass
        self._process.join()

    def _worker_fn(self, type_, in_):
        try:
            while True:
                # Receive request.
                serialized = in_.recv()

                if serialized is None:
                    if hasattr(type_, 'close'):
                        type_.close()
                    in_.close()
                    return

                method_name = serialized[0].decode()
                inputs = serialized[1:]

                # Compute result.
                results = getattr(type_, method_name)(*inputs)
                if results is not None:
                    results = nest.flatten(results)

                # Respond.
                in_.send(results)
        except Exception as e:
            if 'type_' in locals() and hasattr(type_, 'close'):
                try:
                    type_.close()
                except:
                    pass
            in_.send(e)


class PyProcess(object):
    def __init__(self, type_):
        self._proxy = _TFProxy(type_)

    @property
    def proxy(self):
        return self._proxy

    def close(self):
        self._proxy._close()

    def start(self):
        self._proxy._start()
