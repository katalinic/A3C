import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from models import SimpleModel, CNNModel

class Worker(object):
    def __init__(self, task, obs_size, action_size, env, learning_rate = 1e-3, gamma=0.99, beta=0.01):
        self.update_every_k_steps = 5 #initial hardcoding
        self.t = 1
        self.episodes = 0
        self.env = env
        self.gamma = gamma #discount factor
        self.done = False
        self.task = task
        self.obs_size = obs_size
        self.action_size = action_size
        self.beta = beta #entropy coefficient

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                # self.global_agent = SimpleModel(obs_size, action_size)
                self.global_agent = CNNModel(obs_size, action_size)
                #any variable under same scope must be added after agent definition, otherwise its instantiation will pull it under itself
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
                self.global_step_increment = tf.assign_add(self.global_step, tf.constant(1, tf.int32))

        starter_learning_rate = self._logUniformSample()
        self.lr = tf.train.polynomial_decay(starter_learning_rate, self.global_step,
                                                  80000000, 0, power = 1.)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.agent = CNNModel(obs_size, action_size)
                self.a = tf.placeholder(tf.int32, [None])
                self.r = tf.placeholder(tf.float32, [None])
                self._build_loss()
                self._gradient_exchange()

    def _build_loss(self):
        adv = self.r - self.agent.value
        # logp = -tf.log(tf.reduce_sum(self.agent.probs*self.a,axis=1))
        logp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.agent.probs,labels=self.a)
        pg = tf.reduce_sum(logp*tf.stop_gradient(adv))
        sq = 0.5*tf.reduce_sum(tf.square(adv))
        #entropy
        entropy = -tf.reduce_sum(self.agent.probs * tf.log(self.agent.probs), 1)
        self.loss = pg+0.5*sq-self.beta*entropy

    def interaction(self, sess):
        sess.run(self.sync_op)
        X, R, A = self._nstep_rollout(sess)
        sess.run(self.train_op, feed_dict = {self.agent.x : X, self.r : R, self.a : A})

    def _gradient_exchange(self):
        #get worker gradients
        gradients = tf.gradients(self.loss, self.agent.vars)
        self.gradients, _ = tf.clip_by_global_norm(gradients, 40)
        #synchronisation of parameters
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.agent.vars, self.global_agent.vars)])
        optimiser = tf.train.RMSPropOptimizer(learning_rate = self.lr, decay=0.99, epsilon=1e-1)
        self.train_op = optimiser.apply_gradients(zip(self.gradients,self.global_agent.vars))

    def _nstep_rollout(self, sess):
        states = []
        true_rewards = []
        actions_taken = []
        t_start = self.t
        if self.done == True or self.t==1:
            self.episodes += 1
            obs = self.env.reset()
            self.done = False
        else: obs = self.last_state

        while (not self.done and self.t-t_start!=self.update_every_k_steps):
            obs = obs.reshape(1,*self.obs_size)
            pr = self.agent.policy_and_value(sess, obs, 'prob')
            action = np.random.choice(self.action_size, p=pr.ravel())
            states.append(obs)
            actions_taken.append(action)
            obs, reward, done, info = self.env.step(action)
            sess.run(self.global_step_increment)
            self.t += 1
            true_rewards.append(reward)
            self.done = done

        exp = (states, actions_taken, true_rewards)
        if self.done != True:
            self.last_state = obs
            self.last_value = self.agent.policy_and_value(sess, self.last_state.reshape(1,*self.obs_size), 'value')[0]
        else: self.last_value = 0

        return self._process_experience(exp)

    def _process_experience(self, experience):
        states, actions_taken, true_rewards = experience
        R = self.last_value
        discounted_reward = []
        for r in true_rewards[::-1]:
            R=r+self.gamma*R
            discounted_reward.append(R)
        return np.concatenate(states), discounted_reward[::-1], actions_taken

#     @staticmethod
#     def _logUniformSample():
#         #restricted as larger initial learning rate seems to break rmsprop
#         return np.power(10,np.random.rand()*0.5-3.5)
