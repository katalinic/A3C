import threading
import time

import tensorflow as tf
import numpy as np

import gym

from preprocessing import atari_preprocess

from pyprocess import PyProcess


nest = tf.contrib.framework.nest
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("unroll_length", 5, "Number of steps per rollout.")
flags.DEFINE_string("env", 'BreakoutDeterministic-v4', "Gym environment.")
flags.DEFINE_float("learning_rate", 7e-4, "Learning rate.")
flags.DEFINE_float("gamma", 0.99, "Discount rate.")
flags.DEFINE_float("beta_v", 0.5, "Value loss coefficient.")
flags.DEFINE_float("beta_e", 0.01, "Entropy loss coefficient.")
flags.DEFINE_float("rms_decay", 0.99, "RMS decay.")
flags.DEFINE_float("rms_epsilon", 1e-1, "RMS epsilon.")
flags.DEFINE_float("grad_clip", 40., "Gradient clipping norm.")
flags.DEFINE_integer("train_steps", 40000000, "Training steps.")
flags.DEFINE_integer("test_every", 1000000, "Test every x training steps.")
flags.DEFINE_integer("max_steps", 80000000, "Max steps for LR decay.")
flags.DEFINE_integer("test_eps", 30, "Number of test episodes.")
flags.DEFINE_integer("seed", 1, "Random seed.")

GLOBAL_NET_SCOPE = 'Global_Net'
SPECS = {
    'reset': (tf.float32, [84, 84, 4]),
    'step': ([tf.float32, tf.float32, tf.bool, tf.bool],
             [[84, 84, 4], [], [], []])}


def torso(input_, num_actions):
    initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=1.0, mode='FAN_IN', uniform=True)
    with tf.variable_scope("torso", reuse=tf.AUTO_REUSE):
        x = tf.expand_dims(input_, 0)
        x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4,
            activation=tf.nn.relu, kernel_initializer=initializer)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2,
            activation=tf.nn.relu, kernel_initializer=initializer)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1,
            activation=tf.nn.relu, kernel_initializer=initializer)
        x = tf.reshape(x, [-1, 7 * 7 * 64])
        x = tf.layers.dense(x, 512, activation=tf.nn.relu,
            kernel_initializer=initializer)
        logits = tf.layers.dense(x, num_actions, activation=None,
            kernel_initializer=initializer)
        action = tf.multinomial(
            logits, num_samples=1, output_dtype=tf.int32)
        logits = tf.squeeze(logits)
        action = tf.squeeze(action)
        value = tf.layers.dense(x, 1, activation=None,
            kernel_initializer=initializer)
        value = tf.squeeze(value)
    return action, logits, value


def rollout(env, num_actions):
    init_obs = env.reset()
    init_a, init_logit, init_v = torso(init_obs, num_actions)
    init_r = tf.zeros([], dtype=tf.float32)
    init_d = tf.constant(False, dtype=tf.bool)
    init_true_d = tf.constant(False, dtype=tf.bool)
    # Dummy that should technically be an env step at start of episode.
    next_obs = tf.identity(init_obs)

    def create_state(t):
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(
                t.op.name, initializer=t, use_resource=True)

    # Persistent variables.
    persistent_state = nest.map_structure(create_state,
        (next_obs, init_a, init_logit, init_r, init_v, init_d, init_true_d))

    first_values = nest.map_structure(
        lambda v: v.read_value(), persistent_state)

    def step(input_, unused_i):
        obs = input_[0]
        action_, logit_, value_ = torso(obs, num_actions)
        obs_, r_, d_, td_ = env.step(action_)
        return obs_, action_, logit_, r_, value_, d_, td_

    outputs = tf.scan(
        step,
        tf.range(FLAGS.unroll_length),
        initializer=first_values,
        parallel_iterations=1)

    # Update persistent state with last element of each output.
    update_persistent_state = nest.map_structure(
        lambda v, t: v.assign(t[-1]), persistent_state, outputs)

    with tf.control_dependencies(nest.flatten(update_persistent_state)):
        # Append first states to the outputs.
        full_outputs = nest.map_structure(
            lambda first, rest: tf.concat([[first], rest], axis=0),
            first_values, outputs)

    return full_outputs


def loss_function(rollout_outputs):
    # Subset inputs.
    actions, logits, rewards = nest.map_structure(
        lambda t: t[:-1], rollout_outputs[:-2])
    values, dones = rollout_outputs[-2:]

    # Discounted reward calculation.
    def discount(rewards, gamma, values, dones):
        tf_gamma = tf.constant(gamma, tf.float32)
        processed_rewards = tf.squeeze(rewards)
        processed_rewards = tf.clip_by_value(processed_rewards, -1, 1)
        processed_rewards = tf.reverse(processed_rewards, axis=[0])
        reversed_dones = tf.reverse(dones, axis=[0])
        bootstrap_value = values[-1]
        discounted_reward = tf.scan(
            lambda R, v: v[0] + tf_gamma * R * tf.to_float(~v[1]),
            [processed_rewards, reversed_dones],
            initializer=bootstrap_value,
            back_prop=False,
            parallel_iterations=1)
        discounted_reward = tf.reverse(discounted_reward, axis=[0])
        return tf.stop_gradient(discounted_reward)

    dones = dones[:-1]
    discounted_targets = discount(rewards, FLAGS.gamma, values, dones)
    values = values[:-1]
    advantages = discounted_targets - values

    def policy_gradient_loss(logits, actions, advantages):
        cross_entropy_per_timestep = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
        policy_gradient_per_timestep = \
            cross_entropy_per_timestep * tf.stop_gradient(advantages)
        return tf.reduce_sum(policy_gradient_per_timestep)

    def advantage_loss(advantages):
        advantage_loss_per_timestep = 0.5 * tf.square(advantages)
        return tf.reduce_sum(advantage_loss_per_timestep)

    def entropy_loss(logits):
        policy = tf.nn.softmax(logits)
        log_policy = tf.nn.log_softmax(logits)
        entropy_per_timestep = - tf.reduce_sum(policy * log_policy, axis=1)
        return tf.reduce_sum(-entropy_per_timestep)

    loss = policy_gradient_loss(logits, actions, advantages)
    loss += FLAGS.beta_v * advantage_loss(advantages)
    loss += FLAGS.beta_e * entropy_loss(logits)

    return loss


def gradient_exchange(loss, agent_vars, shared_vars, optimiser):
    # Get worker gradients.
    gradients = tf.gradients(loss, agent_vars)
    if FLAGS.grad_clip > 0:
        gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.grad_clip)
    # Synchronisation of parameters.
    sync_op = tf.group(
        *[v1.assign(v2) for v1, v2 in zip(agent_vars, shared_vars)])
    train_op = optimiser.apply_gradients(zip(gradients, shared_vars))
    return train_op, sync_op


class Worker():
    def __init__(self, env_, scope, num_actions,
                 global_scope=None, optimiser=None, global_step=None):
        self.scope = scope
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope)
        with tf.variable_scope(scope):
            self.rollout_outputs = rollout(env_.proxy, num_actions)
            agent_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=tf.get_variable_scope().name)
            loss = loss_function(self.rollout_outputs[1:-1])
            train_op, sync_op = gradient_exchange(
                loss, agent_vars, global_vars, optimiser)
            global_step_increment = tf.assign_add(
                global_step, tf.constant(FLAGS.unroll_length, tf.int32))
            with tf.control_dependencies([sync_op]):
                self.train_op = tf.group(train_op, global_step_increment)

    def work(self, sess, global_step, coord=None):
        start = time.time()
        t, T, test_epoch = 0, 0, 0
        while not coord.should_stop() and T < FLAGS.train_steps:
            _, T = sess.run([self.train_op, global_step])
            t += 1
            # Test every 1000000 training steps.
            if T // FLAGS.test_every > test_epoch and FLAGS.test_eps:
                self._test(sess)
                test_epoch = T // FLAGS.test_every
        print(time.time() - start)

    def _test(self, sess):
        print('Testing.')
        eps = 0
        total_rewards = 0
        commenced = False
        # Instead of resetting environment, we wait until current episode
        # ends, and then commence tracking as normal.
        while eps < FLAGS.test_eps + 1:
            outp = sess.run(self.rollout_outputs)
            r, d = outp[-4], outp[-1]
            if commenced:
                total_rewards += np.sum(r[:-1])
            if np.sum(d[:-1]) > 0:
                commenced = True
                eps += 1
        print("Average Reward: {:.2f}".format(total_rewards / FLAGS.test_eps))


def train(seed=0):
    # Set random seeds.
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env_ = gym.make(FLAGS.env)
    num_actions = env_.action_space.n
    with tf.variable_scope(GLOBAL_NET_SCOPE):
        dummy_torso = torso(tf.zeros([84, 84, 4], tf.float32),
                            num_actions)
    env_.close()

    # Shared optimiser.
    global_step = tf.get_variable(
        "global_step", [], tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32),
        trainable=False)
    lr = tf.train.polynomial_decay(FLAGS.learning_rate, global_step,
                                   FLAGS.max_steps, 0, power=1.)
    optimiser = tf.train.RMSPropOptimizer(
        learning_rate=lr, decay=FLAGS.rms_decay, epsilon=FLAGS.rms_epsilon)

    # Create envs.
    envs = []
    for i in range(FLAGS.num_workers):
        env_ = gym.make(FLAGS.env)
        env_.seed(seed + i)
        # Assuming use of an environment with built-in frame skip of 4,
        # this serves as an approximation to the noop of 30.
        env_ = atari_preprocess(env_, noop=30 // 4 + 1)
        env_.specs = SPECS
        proxy_env = PyProcess(env_)
        proxy_env.start()
        envs.append(proxy_env)

    # Create workers.
    workers = []
    for i in range(FLAGS.num_workers):
        worker = Worker(envs[i], 'W_{}'.format(i), num_actions,
                        GLOBAL_NET_SCOPE, optimiser, global_step)
        workers.append(worker)

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=FLAGS.num_workers)
    sess = tf.Session(config=config)
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    worker_threads = []
    for worker in workers:
        t = threading.Thread(
            target=lambda: worker.work(sess, global_step, coord))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

    for env in envs:
        env.close()


def main(_):
    train(FLAGS.seed)


if __name__ == '__main__':
    tf.app.run()
