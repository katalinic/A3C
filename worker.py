import os
import time

import tensorflow as tf
import numpy as np

from rollouts import rollout
from optimisation import loss_calculation, gradient_exchange


class Worker:
    """Handles all agent-environment interaction.

    This includes rollouts, optimisation, model loading and saving,
    performance tracking.
    """
    def __init__(self, env, agent, scope='Worker',
                 constants=None):
        self.env = env
        self.agent = agent
        self.action_space = self.agent.action_space
        self.scope = scope
        self.constants = constants

    def build_rollout(self):
        with tf.variable_scope(self.scope):
            self.rollout_outputs = rollout(self.env, self.agent,
                                           self.constants.unroll_length)

    def build_loss(self):
        with tf.variable_scope(self.scope):
            self.loss = loss_calculation(
                self.rollout_outputs, self.action_space, self.constants)

    def build_optimisation(self, global_step, optimiser, from_scope,
                           to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope=to_scope)
        if len(from_vars) != len(to_vars):
            print("""Warning: number of variables in source and target scopes
                  do not match. Ignore if A2C.""")
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=self.scope)
        if len(from_vars) != len(all_vars):
            print("Warning: source scope does not cover agent's variables.")
        train_op = gradient_exchange(self.loss, from_vars, to_vars, optimiser,
                                     self.constants)

        global_step_increment = tf.assign_add(
            global_step, tf.constant(self.constants.unroll_length, tf.int32))
        # Parameter synchronisation if target scope differs from source scope.
        if from_scope != to_scope:
            sync_op = tf.group(
                *[v1.assign(v2) for v1, v2 in zip(from_vars, to_vars)])
            with tf.control_dependencies([sync_op]):
                self.train_op = tf.group(train_op, global_step_increment)
        else:
            self.train_op = tf.group(train_op, global_step_increment)

    def train(self, sess, global_step, coord=None):
        start = time.time()
        t, T = 0, sess.run(global_step)
        test_epoch = T // self.constants.test_every
        # Acts as env reset.
        sess.run(self.rollout_outputs)
        while not coord.should_stop() and T < self.constants.train_steps:
            _, T = sess.run([self.train_op, global_step])
            t += 1
            if (self.constants.test_eps and
                    T // self.constants.test_every > test_epoch):
                self.evaluate(sess)
                test_epoch = T // self.constants.test_every
        print(time.time() - start)

    def evaluate(self, sess):
        print('Testing.')
        eps = 0
        total_rewards = 0
        commenced = False
        # Instead of resetting environment, we wait until current episode
        # ends, and then commence tracking as normal.
        # TODO: Fix this.
        while eps < self.constants.test_eps + 1:
            env_outputs, _, agent_outputs = sess.run(self.rollout_outputs)
            r, d = env_outputs.reward, env_outputs.done
            if commenced:
                total_rewards += np.sum(r[:-1])
            if np.sum(d[:-1]) > 0:
                commenced = True
                eps += 1
        average_reward = total_rewards / self.constants.test_eps
        print("Average Reward: {:.2f}".format(average_reward))

    def save_model(self, sess, saver, model_directory):
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        saver.save(sess, model_directory + 'model.checkpoint')

    def load_model(self, sess, saver, model_directory):
        chckpoint = tf.train.get_checkpoint_state(model_directory)
        if chckpoint is not None:
            saver.restore(sess, chckpoint.model_checkpoint_path)
