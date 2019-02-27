import threading

import tensorflow as tf
import numpy as np

from envs import create_env
from agents import A3CAgent
from worker import Worker


nest = tf.contrib.framework.nest
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_string("env", 'BreakoutDeterministic-v4', "Gym environment.")
flags.DEFINE_integer("unroll_length", 5, "Number of steps per rollout.")
flags.DEFINE_integer("train_steps", 40000000, "Training steps.")
flags.DEFINE_integer("test_every", 1000000, "Test every x training steps.")
flags.DEFINE_integer("test_eps", 30, "Number of test episodes.")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_boolean("eval_mode", True, "True if evaluating performance.")

# Optimisation.
flags.DEFINE_float("gamma", 0.99, "Discount factor.")
flags.DEFINE_float("beta_v", 0.5, "Value loss coefficient.")
flags.DEFINE_float("beta_e", 0.01, "Entropy loss coefficient.")
flags.DEFINE_float("init_lr", 7e-4, "Initial learning rate.")

# Model saving.
flags.DEFINE_string("model_directory", './models/', "Model directory.")
flags.DEFINE_boolean("save_model", False, "True if saving model.")
flags.DEFINE_boolean("load_model", False, "True if loading model.")

GLOBAL_SCOPE = 'Global_Net'


def set_random_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def train(constants):
    set_random_seeds(constants.seed)

    # Dummy worker to get shared variables into graph.
    dummy_env, action_space = create_env(constants.env)
    dummy_agent = A3CAgent(action_space)
    dummy_worker = Worker(dummy_env.proxy, dummy_agent, GLOBAL_SCOPE,
                          constants)
    dummy_worker.build_rollout()
    dummy_worker.build_loss()

    with tf.variable_scope(GLOBAL_SCOPE):
        global_step = tf.get_variable(
            "global_step", [], tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)
    # Optimiser.
    opt = tf.train.AdamOptimizer(learning_rate=constants.init_lr)
    shared_optimisers = [opt]

    # Create envs.
    print('Creating environments.')
    envs = []
    for i in range(constants.num_workers):
        env, action_space = create_env(constants.env)
        envs.append(env)

    # Create agents.
    print('Creating agents.')
    agents = []
    for i in range(constants.num_workers):
        agent = A3CAgent(action_space)
        agents.append(agent)

    # Create workers.
    print('Creating workers.')
    workers = []
    for i in range(constants.num_workers):
        worker = Worker(envs[i].proxy, agents[i], 'W_{}'.format(i), constants)
        worker.build_rollout()
        worker.build_loss()
        worker.build_optimisation(
            global_step,
            optimisers=shared_optimisers,
            from_scopes=None,
            to_scopes=[GLOBAL_SCOPE])
        workers.append(worker)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    vars_to_save_load = tf.trainable_variables(GLOBAL_SCOPE) + [global_step]
    if constants.load_model or constants.save_model:
        saver = tf.train.Saver(vars_to_save_load)
    if constants.load_model:
        workers[0].load_model(sess, saver, constants.model_directory)

    # Close dummy env since it seems to have to be open during variable init.
    dummy_env.close()

    print('Starting workers.')
    coord = tf.train.Coordinator()
    worker_threads = []
    for worker in workers:
        t = threading.Thread(
            target=lambda: worker.train(sess, global_step, coord))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

    for env in envs:
        env.close()

    # Save shared variables.
    if constants.save_model:
        workers[0].save_model(sess, saver, constants.model_directory)
    sess.close()


def evaluate(constants):
    env, action_space = create_env(constants.env)
    agent = A3CAgent(action_space)
    worker = Worker(env.proxy, agent, GLOBAL_SCOPE, constants)
    worker.build_rollout()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # TODO: Model loading for A2C.
    if FLAGS.load_model:
        vars_to_save_load = tf.trainable_variables(GLOBAL_SCOPE)
        saver = tf.train.Saver(vars_to_save_load)
        worker.load_model(sess, saver, constants.model_directory)
    worker.evaluate(sess)
    env.close()
    sess.close()


def main(_):
    if FLAGS.eval_mode:
        evaluate(FLAGS)
    else:
        train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
