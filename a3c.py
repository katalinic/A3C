import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from models import SimpleModel, CNNModel

class Worker(object):
    def __init__(self, task, obs_size, action_size, env, learning_rate = 1e-3, gamma=0.99, eps=0.1, beta=0.01):
        #location of some of these parameters to be decided
        self.update_every_k_steps = 5 #initial hardcoding
        self.t = 1
        self.episodes = 0
        self.eps = eps #epsilon greedy
        self.env = env
        self.gamma = gamma #discount factor
        self.done = False
        self.task = task
        self.obs_size = obs_size
        self.action_size = action_size
        # self.lr = learning_rate
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
                                                  20000000, 0, power = 1.)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.a = tf.placeholder(tf.float32, [None, action_size])
                self.r = tf.placeholder(tf.float32, [None])
                # self.agent = SimpleModel(obs_size, action_size)
                self.agent = CNNModel(obs_size, action_size)
                # self.agent.global_step = self.global_step
                self._build_loss()
                self._gradient_exchange()

        end_eps = np.random.choice([0.1,0.01,0.5],p=[0.4,0.3,0.3])
        self.eps = tf.train.polynomial_decay(1., self.global_step,
                                                  4000000, end_eps, power = 1.)

    def _build_loss(self):
        adv = self.r - self.agent.value
        logp = -tf.log(tf.reduce_sum(self.agent.probs*self.a,axis=1))
        pg = tf.reduce_sum(logp*tf.stop_gradient(adv))
        sq = tf.reduce_sum(tf.square(adv))
        #entropy
        entropy = -tf.reduce_sum(self.agent.probs * tf.log(self.agent.probs), 1)
        self.loss = pg+sq-self.beta*entropy

    def interaction(self, sess):
        '''Only external method to be called
        First it synchronises gradients between the worker and the central unit by calling synchronise
        Then it performs the nstep_rollout, which yields the gradients
        Then it applies the gradients to the global network by calling update_global
        '''
        sess.run(self.sync_op)
        X, R, A = self._nstep_rollout(sess)
        sess.run(self.train_op, feed_dict = {self.agent.x : X, self.r : R, self.a : A})

    def _gradient_exchange(self):
        #get worker gradients
        gradients = tf.gradients(self.loss, self.agent.vars)
        self.gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
        #synchronisation of parameters
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.agent.vars, self.global_agent.vars)])
        # optimiser = tf.train.AdamOptimizer(self.lr) #the optimiser will have to be manually written
        optimiser = tf.train.RMSPropOptimizer(learning_rate = self.lr, decay=0.99)
        #apply updates to GLOBAL agent
        self.train_op = optimiser.apply_gradients(zip(self.gradients,self.global_agent.vars))

    def _nstep_rollout(self, sess):
        states = []
        true_rewards = []
        state_values = []
        actions_taken = []
        t_start = self.t
        if self.done == True or self.t==1:
            self.noop_counter = 0
            self.episodes += 1
            obs = self.env.reset()
            self.done = False
        else: obs = self.last_state

        while (not self.done and t_start-self.t!=self.update_every_k_steps):
            obs = obs.reshape(1,*self.obs_size)
            val, pr = self.agent.policy_and_value(sess, obs)
            eps = sess.run(self.eps) #changed to global decay
            if self.t==3000: print (pr)
            action = np.argmax(pr) if np.random.rand()<eps else np.random.choice(self.action_size,p=pr.ravel())
            if self.noop_counter>30:
                action = 1
                self.noop_counter = 0
            if action != 1:
                self.noop_counter += 1

            states.append(obs)
            actions_taken.append(action)
            state_values.append(val[0])
            obs, reward, done, info = self.env.step(action)
            sess.run(self.global_step_increment)
            self.t += 1
            true_rewards.append(reward)
            self.done = done

        exp = (states, state_values, actions_taken, true_rewards, done)
        if self.done != True: self.last_state = obs

        return self._process_experience(exp)

    def _process_experience(self, experience):
        states, state_values, actions_taken, true_rewards, done = experience
        R = state_values
        if self.done: R[-1] = 0
        for i in reversed(range(len(R)-1)):
            R[i]=self.gamma*R[i+1]+true_rewards[i]
        actions_taken = actions_taken[:-1]
        R = R[:-1]
        states = states[:-1]
        actions_taken_one_hot = np.zeros((len(actions_taken),self.action_size))
        actions_taken_one_hot[np.arange(len(actions_taken)),actions_taken]=1.
        return np.concatenate(states), R, actions_taken_one_hot

    @staticmethod
    def _logUniformSample():
        '''changed from -4 to -2 to -4 to -3 and its breaking rmsprop when too close to -2'''
        return np.power(10,np.random.rand()-4)
