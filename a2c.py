import os
import time

import tensorflow as tf
import numpy as np

from tfenv import TFEnv

import gym

nest = tf.contrib.framework.nest
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("learning_rate", 7e-4, "Learning rate.")
flags.DEFINE_float("gamma", 0.99, "Discount rate.")
flags.DEFINE_float("beta_v", 0.5, "Value loss coefficient.")
flags.DEFINE_float("beta_e", 0.01, "Entropy loss coefficient.")
flags.DEFINE_float("rms_decay", 0.99, "RMS decay.")
flags.DEFINE_float("rms_epsilon", 1e-1, "RMS epsilon.")
flags.DEFINE_float("grad_clip", 40., "Gradient clipping norm-.")
flags.DEFINE_integer("unroll_length", 5, "Number of steps per rollout.")
flags.DEFINE_integer("train_steps", 1000000, "Number of train episodes.")
flags.DEFINE_integer("test_eps", 30, "Number of test episodes.")
flags.DEFINE_boolean("training", False, "Boolean for training. Testing if False.")
flags.DEFINE_string("env", 'BreakoutDeterministic-v4', "Gym environment.")
flags.DEFINE_string("model_directory", './models/', "Model directory.")

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
        (init_obs, init_a, init_logit, init_v, init_r, init_d)
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

def optimisation(rollout_outputs):
    # All inputs are to be subset, but need last elements of values
    # and done for discounted reward calculation.
    actions, logits, rewards = nest.map_structure(
        lambda t: t[:-1], rollout_outputs[:-2])
    values, dones = rollout_outputs[-2:]
    logits = tf.squeeze(logits)

    # Discounted reward calculation.
    def discount(rewards, gamma):
        tf_gamma = tf.constant(gamma, tf.float32)
        processed_rewards = tf.squeeze(rewards)
        processed_rewards = tf.clip_by_value(processed_rewards, -1, 1)
        processed_rewards = tf.reverse(processed_rewards, axis=[0])
        bootstrap_value = values[-1] * tf.to_float(~dones[-1])
        discounted_reward = tf.scan(
            lambda R, r: r + tf_gamma * R,
            processed_rewards,
            initializer=bootstrap_value,
            back_prop=False,
            parallel_iterations=1)
        discounted_reward = tf.reverse(discounted_reward, axis=[0])
        return discounted_reward

    discounted_targets = discount(rewards, FLAGS.gamma)
    values = values[:-1]
    advantages = discounted_targets - values

    def policy_gradient_loss(logits, actions, advantages):
        cross_entropy_per_timestep = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=actions)
        policy_gradient_per_timestep = cross_entropy_per_timestep * tf.stop_gradient(advantages)
        return policy_gradient_per_timestep

    def advantage_loss(advantages):
        advantage_loss_per_timestep = 0.5 * tf.square(values)
        return advantage_loss_per_timestep

    def entropy_loss(logits):
        policy = tf.nn.softmax(logits)
        log_policy = tf.nn.log_softmax(logits)
        entropy_per_timestep = - tf.reduce_sum(policy * log_policy, axis=1)
        return -entropy_per_timestep

    loss = policy_gradient_loss(logits, actions, advantages)
    loss += FLAGS.beta_v * advantage_loss(advantages)
    loss += FLAGS.beta_e * entropy_loss(logits)

    dones = dones[:-1]
    # Masking sets loss to 0 after first done is encountered.
    mask = tf.Variable(tf.ones([FLAGS.unroll_length], tf.float32), name='mask')
    reset_mask = mask.assign(tf.ones([FLAGS.unroll_length], tf.float32))

    idx = tf.argmax(tf.cast(dones, tf.int32), output_type=tf.int32)
    clipped_idx = tf.cast(tf.clip_by_value(idx, 0, 1), tf.float32)
    done_not_present = tf.cast(tf.equal(dones[0], False), tf.float32)
    with tf.control_dependencies([reset_mask]):
        apply_mask = mask[idx + 1:].assign(
            (1. - clipped_idx) * (done_not_present -
            tf.zeros(FLAGS.unroll_length - (idx + 1), tf.float32))
            + clipped_idx * tf.zeros(FLAGS.unroll_length - (idx + 1), tf.float32))

    with tf.control_dependencies([apply_mask]):
        masked_loss = tf.reduce_sum(loss * mask)

    optimiser = tf.train.RMSPropOptimizer(
        learning_rate=FLAGS.learning_rate, decay=FLAGS.rms_decay, epsilon=FLAGS.rms_epsilon)

    gradients = tf.gradients(loss, tf.trainable_variables())
    if FLAGS.grad_clip > 0:
        gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.grad_clip)

    train_op = optimiser.apply_gradients(zip(gradients, tf.trainable_variables()))

    return train_op, dones


def train(env, num_actions):
    rollout_outputs, reset_persistent_state = rollout(env, num_actions)
    train_op, dones = optimisation(rollout_outputs[1:])
    with tf.control_dependencies(nest.flatten(reset_persistent_state)):
        env_reset = env.reset()
    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    start_time = time.time()
    sess.run(env_reset)
    num_rollouts = FLAGS.train_steps//FLAGS.unroll_length
    for i in range(num_rollouts + 1):
        _, d = sess.run([train_op, dones])
        if np.sum(d) > 0:
            sess.run(env_reset)
        if i > 0 and i % (num_rollouts / 10) == 0:
            eps = 0
            total_rewards = 0
            sess.run(env_reset)
            while eps < FLAGS.test_eps:
                outp = sess.run(rollout_outputs)
                r, d = outp[-3], outp[-1]
                total_rewards += np.sum(r[:-1])
                if np.sum(d) > 0:
                    sess.run(env_reset)
                    eps += 1
            print("Average Reward: {:.2f}".format(total_rewards/FLAGS.test_eps))

    print("Training completed. Time taken: {:.2f}".format(time.time() - start_time))

    if not os.path.exists(FLAGS.model_directory):
        os.mkdir(FLAGS.model_directory)

    saver = tf.train.Saver(tf.trainable_variables())
    saver.save(sess, FLAGS.model_directory + 'model.checkpoint')

def test(env, num_actions):
    rollout_outputs, reset_persistent_state = rollout(env, num_actions)
    with tf.control_dependencies(nest.flatten(reset_persistent_state)):
        env_reset = env.reset()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Load trained model.
    saver = tf.train.Saver(tf.trainable_variables())
    chckpoint = tf.train.get_checkpoint_state(FLAGS.model_directory)
    saver.restore(sess, chckpoint.model_checkpoint_path)

    eps = 0
    total_rewards = 0
    sess.run(env_reset)
    while eps < FLAGS.test_eps:
        outp = sess.run(rollout_outputs)
        r, d = outp[-3], outp[-1]
        total_rewards += np.sum(r[:-1])
        if np.sum(d) > 0:
            sess.run(env_reset)
            eps += 1
    print("Average Reward: {:.2f}".format(total_rewards/FLAGS.test_eps))

def main(_):
    env_ = gym.make(FLAGS.env)
    num_actions = env_.action_space.n
    env = TFEnv(env_)

    if FLAGS.training:
        train(env, num_actions)
    else:
        if not os.path.exists(FLAGS.model_directory):
            raise ValueError('Model directory does not exist.')
        test(env, num_actions)

if __name__ == '__main__':
    tf.app.run()
