import threading
import time

import tensorflow as tf
import numpy as np
from tfenv import TFEnv
import gym

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
flags.DEFINE_integer("train_steps", 40000000, "Number of steps.")
flags.DEFINE_integer("max_steps", 80000000, "Max number of steps for LR decay.")
flags.DEFINE_integer("test_eps", 30, "Number of test episodes.")
GLOBAL_NET_SCOPE = 'Global_Net'

def torso(input_, num_actions):
    with tf.variable_scope("torso", reuse=tf.AUTO_REUSE):
        x = tf.expand_dims(input_, 0)
        x = tf.layers.conv2d(x, filters=16, kernel_size=[8, 8],
                             strides=(4, 4), activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=32, kernel_size=[4, 4],
                             strides=(2, 2), activation=tf.nn.relu)
        x = tf.reshape(x, [-1, 9 * 9 * 32])
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(x, num_actions, activation_fn=None)
        action = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        action = tf.squeeze(action)
        value = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        value = tf.squeeze(value)
    return action, logits, value

def rollout(env, num_actions):
    init_obs = env.obs.read_value()
    init_a, init_logit, init_v = torso(init_obs, num_actions)
    init_r = tf.zeros([], dtype=tf.float32)
    init_d = tf.constant(False, dtype=tf.bool)

    def create_state(t):
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    # Persistent variables.
    persistent_state = nest.map_structure(
        create_state, (init_obs, init_a, init_logit, init_r, init_v, init_d)
        )

    reset_persistent_state = nest.map_structure(
        lambda p, i: p.assign(i), persistent_state,
        (init_obs, init_a, init_logit, init_r, init_v, init_d)
        )

    first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)

    def step(input_, unused_i):
        obs = input_[0]
        action_, logit_, value_ = torso(obs, num_actions)
        env_step = env.step(action_)
        with tf.control_dependencies([env_step]):
            obs_ = env.obs.read_value()
            r_ = env.reward.read_value()
            d_ = env.done.read_value()
        return obs_, action_, logit_, r_, value_, d_

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

    return full_outputs, reset_persistent_state

def loss_function(rollout_outputs):
    # All inputs are to be subset, but need last elements of values
    # and done for discounted reward calculation.
    actions, logits, rewards = nest.map_structure(
        lambda t: t[:-1], rollout_outputs[:-2])
    values, dones = rollout_outputs[-2:]
    logits = tf.squeeze(logits)

    # Discounted reward calculation.
    def discount(rewards, gamma, values, dones):
        tf_gamma = tf.constant(gamma, tf.float32)
        processed_rewards = tf.squeeze(rewards)
        processed_rewards = tf.clip_by_value(processed_rewards, -1, 1)
        processed_rewards = tf.reverse(processed_rewards, axis=[0])
        reversed_dones = tf.reverse(dones, axis=[0])
        bootstrap_value = values[-1] * tf.to_float(~dones[-1])
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
        cross_entropy_per_timestep = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=actions)
        policy_gradient_per_timestep = cross_entropy_per_timestep * tf.stop_gradient(advantages)
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
    gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.grad_clip)
    # Synchronisation of parameters.
    sync_op = tf.group(
        *[v1.assign(v2) for v1, v2 in zip(agent_vars, shared_vars)])
    train_op = optimiser.apply_gradients(zip(gradients, shared_vars))
    return train_op, sync_op

class Worker():
    def __init__(self, env_, scope, global_scope=None, optimiser=None, global_step=None):
        num_actions = env_.action_space.n
        self.scope = scope
        if scope == global_scope:
            with tf.variable_scope(scope):
                env = TFEnv(env_)
                self.env_reset = env.reset()
                self.rollout_outputs, _ = rollout(env, num_actions)
        else:
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope)
            with tf.variable_scope(scope):
                env = TFEnv(env_)
                self.rollout_outputs, reset_persistent_state = rollout(env, num_actions)
                with tf.control_dependencies(nest.flatten(reset_persistent_state)):
                    self.env_reset = env.reset()
                agent_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
                loss = loss_function(self.rollout_outputs[1:])
                train_op, sync_op = gradient_exchange(
                    loss, agent_vars, global_vars, optimiser)
                global_step_increment = tf.assign_add(
                    global_step, tf.constant(FLAGS.unroll_length, tf.int32))
                with tf.control_dependencies([sync_op]):
                    self.train_op = tf.group(train_op, global_step_increment)

    def work(self, sess, global_step, coord=None):
        start = time.time()
        t, T = 0, 0
        num_rollouts = FLAGS.train_steps // FLAGS.unroll_length
        rollouts_per_worker = num_rollouts // FLAGS.num_workers
        sess.run(self.env_reset)
        while not coord.should_stop() and T < FLAGS.train_steps:
            _, T = sess.run([self.train_op, global_step])
            # if self.scope[-1] == '0' and t % (rollouts_per_worker // 100) == 0:
            #     print(self.scope, t)
            t += 1
            # Test every 10% of training progress.
            if t > 0 and t % (rollouts_per_worker // 10) == 0 and FLAGS.test_eps:
                self.test(sess)
        print(time.time() - start)

    def test(self, sess):
        print('Testing.')
        eps = 0
        total_rewards = 0
        sess.run(self.env_reset)
        while eps < FLAGS.test_eps:
            outp = sess.run(self.rollout_outputs)
            r, d = outp[-3], outp[-1]
            total_rewards += np.sum(r[:-1])
            if np.sum(d[:-1]) > 0:
                eps += 1
        print("Average Reward: {:.2f}".format(total_rewards / FLAGS.test_eps))

def train():
    env_ = gym.make(FLAGS.env)
    global_worker = Worker(env_, GLOBAL_NET_SCOPE, GLOBAL_NET_SCOPE)
    global_step = tf.get_variable("global_step", [], tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32),
        trainable=False)
    lr = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, FLAGS.max_steps, 0, power=1.)
    optimiser = tf.train.RMSPropOptimizer(
        learning_rate=lr, decay=FLAGS.rms_decay, epsilon=FLAGS.rms_epsilon)
    workers = []
    for i in range(FLAGS.num_workers):
        env_ = gym.make(FLAGS.env)
        worker = Worker(env_, 'W_{}'.format(i), GLOBAL_NET_SCOPE, optimiser, global_step)
        workers.append(worker)

    config = tf.ConfigProto(intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=2 * FLAGS.num_workers)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(sess, global_step, coord)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
