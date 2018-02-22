import tensorflow as tf
from train import train
from collections import defaultdict

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.app.flags.DEFINE_integer("num_workers", 1, "Number of agents")
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("train", "inference", "Either 'train' or 'inference'") #make inference default to prevent accidental training
tf.app.flags.DEFINE_string("record", "False", "Either 'True' or 'False'") #whether to record - should only be used with inference
FLAGS = tf.app.flags.FLAGS


def build_cluster_def(num_workers, num_ps, port=2222):
  cluster = defaultdict(list)

  host = 'localhost'
  for _ in range(num_ps):
    cluster['ps'].append('{}:{}'.format(host, port))
    port += 1

  for _ in range(num_workers):
    cluster['worker'].append('{}:{}'.format(host, port))
    port += 1

  return tf.train.ClusterSpec(cluster).as_cluster_def()

def main(_):

    #temporarily testing single agent
    cluster = build_cluster_def(FLAGS.num_workers, 1)

    if FLAGS.job_name == 'worker':
        server = tf.train.Server(
            cluster, job_name="worker", task_index=FLAGS.task,
            config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        train(FLAGS, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=FLAGS.task,
                                     config=tf.ConfigProto(device_filters=["/job:ps"]))
        server.join() #added to test

if __name__ == '__main__':
    tf.app.run()
