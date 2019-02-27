import os
import time
from collections import Counter

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

    def build_optimisation(self, global_step, optimisers=None,
                           from_scopes=None, to_scopes=None,
                           learning_rates=None):
        # Arguments must be sequences.
        if not hasattr(self, 'loss'):
            raise ValueError("Loss graph not built yet.")
        if from_scopes is None:
            from_scopes = ['']
        from_scopes = [self.scope + '/' + scope for scope in from_scopes]
        if to_scopes is None:
            to_scopes = from_scopes
        # Ensure that number of variables matches. Only raise warning if so.
        from_vars = [var for scope in from_scopes for var in
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)]
        to_vars = [var for scope in to_scopes for var in
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope)]
        if len(from_vars) != len(to_vars):
            print("""Warning: number of variables in source and target scopes
                  do not match. Setting target scopes to source scopes.
                  Ignore if A2C.""")
            to_scopes = from_scopes
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=self.scope)
        if len(from_vars) != len(all_vars):
            print("Warning: source scope does not cover all variables.")
        if optimisers is not None:
            if len(optimisers) != len(from_scopes):
                raise ValueError("Please provide an optimiser for each scope.")
        else:
            if learning_rates is None:
                raise ValueError("""Learning rate sequence cannot be empty if
                                 no optimisers are supplied.""")
            if len(learning_rates) != len(from_scopes):
                raise ValueError("""Please provide a learning rate for each
                                 scope.""")
            optimisers = [tf.train.AdamOptimizer(learning_rate=lr) for
                          lr in learning_rates]
        grad_apply_ops = [gradient_exchange(self.loss, f, t, o) for f, t, o in
                          zip(from_scopes, to_scopes, optimisers)]
        train_op = tf.group(*grad_apply_ops)

        global_step_increment = tf.assign_add(
            global_step, tf.constant(self.constants.unroll_length, tf.int32))
        # Parameter synchronisation if target scope differs from source scope.
        if from_scopes != to_scopes:
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
        actions_taken = []
        while eps < self.constants.test_eps + 1:
            env_outputs, _, agent_outputs = sess.run(self.rollout_outputs)
            actions_taken += agent_outputs.action.tolist()
            r, d = env_outputs.reward, env_outputs.done
            if commenced:
                total_rewards += np.sum(r[:-1])
            if np.sum(d[:-1]) > 0:
                commenced = True
                eps += 1
        average_reward = total_rewards / self.constants.test_eps
        print("Average Reward: {:.2f}".format(average_reward))
        print(Counter(actions_taken))

    def save_model(self, sess, saver, model_directory):
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        saver.save(sess, model_directory + 'model.checkpoint')

    def load_model(self, sess, saver, model_directory):
        chckpoint = tf.train.get_checkpoint_state(model_directory)
        if chckpoint is not None:
            saver.restore(sess, chckpoint.model_checkpoint_path)
