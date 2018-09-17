import os
import time
from collections import defaultdict

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tfenv import TFEnv

import gym

tf.logging.set_verbosity(tf.logging.DEBUG)

nest = tf.contrib.framework.nest
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_integer("num_workers", 1, "Number of agents")
flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
flags.DEFINE_integer("task", 0, "Index of task within the job")
flags.DEFINE_boolean("train", False, "Boolean for training or inference.")

flags.DEFINE_float("learning_rate", 7e-4, "Learning rate.")
flags.DEFINE_float("gamma", 0.99, "Discount rate.")
flags.DEFINE_float("beta_v", 0.5, "Value loss coefficient.")
flags.DEFINE_float("beta_e", 0.01, "Entropy loss coefficient.")
flags.DEFINE_float("rms_decay", 0.99, "RMS decay.")
flags.DEFINE_float("rms_epsilon", 1e-1, "RMS epsilon.")
flags.DEFINE_integer("unroll_length", 5, "Number of steps per rollout.")
flags.DEFINE_integer("train_steps", 1000000, "Number of train episodes.")
flags.DEFINE_integer("test_eps", 30, "Number of test episodes.")
flags.DEFINE_boolean("training", False, "Boolean for training. Testing if False.")
flags.DEFINE_string("env", 'BreakoutDeterministic-v4', "Gym environment.")
# flags.DEFINE_string("env", 'PongDeterministic-v4', "Gym environment.")
flags.DEFINE_string("model_directory", './tmp/', "Model directory.")
flags.DEFINE_boolean("animate", False, "Animate test runs.")
flags.DEFINE_boolean("save_progress", False, "Save models during training.")

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
    # init_obs = env.obs#.read_value()
    init_obs = tf.zeros([84, 84, 4], tf.float32)
    # Testing.
    init_a, init_logit, init_v = torso(init_obs, num_actions)
    # init_a = tf.zeros([], tf.int32)
    # init_logit = tf.ones([1, num_actions], dtype=tf.float32)
    # init_v = tf.zeros([], tf.float32)

    init_r = tf.constant(0, dtype=tf.float32)
    init_d = tf.constant(False, dtype=tf.bool)

    def create_state(t):
        # with tf.variable_scope(None, default_name='state'):
            # return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)
        return tf.get_variable(t.op.name, initializer=t, use_resource=True)
        # return tf.get_variable(t.op.name, initializer=t)

    # Persistent variables.
    persistent_obs = create_state(init_obs)
    persistent_a = create_state(init_a)
    persistent_logit = create_state(init_logit)
    persistent_r = create_state(init_r)
    # persistent_v = create_state(init_v)
    persistent_v = tf.get_variable('test_v', shape=[], initializer=tf.zeros_initializer(), use_resource=True)
    persistent_d = create_state(init_d)

    # persistent_state = nest.map_structure(
    #     create_state, (init_obs, init_a, init_logit, init_r, init_v, init_d)
    #     )

    reset_obs = persistent_obs.assign(init_obs)
    reset_a = persistent_a.assign(init_a)
    reset_logit = persistent_logit.assign(init_logit)
    reset_r = persistent_r.assign(init_r)
    reset_v = persistent_v.assign(init_v)
    reset_d = persistent_d.assign(init_d)
    reset_persistent_state = tf.group([reset_obs, reset_a, reset_logit, reset_r, reset_v, reset_d])

    # reset_persistent_state = nest.map_structure(
    #     lambda p, i: p.assign(i), persistent_state,
    #     (init_obs, init_a, init_logit, init_r, init_v, init_d)
    #     )

    def step(input_, unused_i):
        obs = input_[0]
        action_, logit_, value_ = torso(obs, num_actions)
        env_step = env.step(action_)
        with tf.control_dependencies([env_step]):
            obs_ = env.obs.read_value()
            r_ = env.reward.read_value()
            d_ = env.done.read_value()
            return obs_, action_, logit_, r_, value_, d_

    # first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
    first_obs = persistent_obs.read_value()
    first_a = persistent_a.read_value()
    first_logit = persistent_logit.read_value()
    first_r = persistent_r.read_value()
    # first_v = persistent_v.read_value()
    first_v = tf.identity(persistent_v)
    first_d = persistent_d.read_value()
    first_values = (first_obs, first_a, first_logit, first_r, first_v, first_d)
    # This is not working for the desired values...
    # It works for a2c

    outputs = tf.scan(
        step,
        tf.range(FLAGS.unroll_length),
        initializer=first_values,
        parallel_iterations=1)

    # Update persistent state with last element of each output.
    # update_persistent_state = nest.map_structure(
    #     lambda v, t: v.assign(t[-1]), persistent_state, outputs)
    update_obs = persistent_obs.assign(outputs[0][-1])
    update_a = persistent_a.assign(outputs[1][-1])
    update_logit = persistent_logit.assign(outputs[2][-1])
    update_r = persistent_r.assign(outputs[3][-1])
    update_v = persistent_v.assign(outputs[4][-1])
    update_d = persistent_d.assign(outputs[5][-1])
    update_persistent_state = tf.group([update_obs, update_a, update_logit, update_r, update_v, update_d])
    # with tf.control_dependencies(nest.flatten(update_persistent_state)):
    with tf.control_dependencies([update_persistent_state]):
        # Append first states to the outputs.
        # full_outputs = nest.map_structure(
        #     lambda first, rest: tf.concat([[first], rest], axis=0),
        #     first_values, outputs)
        full_obs = tf.concat([[first_obs], outputs[0]], axis=0)
        full_a = tf.concat([[first_a], outputs[1]], axis=0)
        full_logit = tf.concat([[first_logit], outputs[2]], axis=0)
        full_r = tf.concat([[first_r], outputs[3]], axis=0)
        full_v = tf.concat([[first_v], outputs[4]], axis=0)
        full_d = tf.concat([[first_d], outputs[5]], axis=0)
        full_outputs = (full_obs, full_a, full_logit, full_r, full_v, full_d)

        return full_outputs, reset_persistent_state, first_values

def loss_function(rollout_outputs):
    # All inputs are to be subset, but need last elements of values
    # and done for discounted reward calculation.
    actions, logits, rewards  = nest.map_structure(
        lambda t: t[:-1], rollout_outputs[:-2])
    values, dones = rollout_outputs[-2:]
    logits = tf.squeeze(logits)

    # Discounted reward calculation.
    def discount(rewards, gamma, values):
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
        return tf.stop_gradient(discounted_reward)

    discounted_targets = discount(rewards, FLAGS.gamma, values)
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
    # masked_loss = loss * tf.to_float(~dones)
    # Proper masking sets loss to 0 after first done is encountered.
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

    return masked_loss, dones
    # return loss, dones

def gradient_exchange(loss, agent_vars, shared_vars, learning_rate):
    # Get worker gradients.
    gradients = tf.gradients(loss, agent_vars)
    gradients, _ = tf.clip_by_global_norm(gradients, 40)
    # Synchronisation of parameters.
    sync_op = tf.group(
        *[v1.assign(v2) for v1, v2 in zip(agent_vars, shared_vars)])
    optimiser = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate, decay=FLAGS.rms_decay, epsilon=FLAGS.rms_epsilon)
    train_op = optimiser.apply_gradients(zip(gradients, shared_vars))
    return train_op, sync_op, gradients

class Worker(object):
    def __init__(self, task, num_actions, env_):
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        # with tf.device("/job:ps/cpu:0"):
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                # g_env = TFEnv(env_)
                # g_rollout_outputs = rollout(g_env, num_actions)
                g_env = TFEnv(env_)
                init_obs = g_env.obs
                g_dummy_a, g_dummy_l, g_dummy_v = torso(init_obs, num_actions)

                self.shared_vars = shared_vars =  tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

                print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))
                print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=tf.get_variable_scope().name))
                self.global_step = tf.get_variable(
                    "global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)
                global_step_increment = tf.assign_add(self.global_step,
                                                      tf.constant(1, tf.int32))

        decayed_lr = tf.train.polynomial_decay(FLAGS.learning_rate, self.global_step,
                                                  80000000, 0, power = 1.)
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                env = TFEnv(env_)
                self.env_reset = env.reset()
                self.rollout_outputs, self.reset_pstate, self.fv = rollout(env, num_actions)
                print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))
                print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=tf.get_variable_scope().name))
                agent_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
                self.loss, self.dones = loss_function(self.rollout_outputs[1:])  #self.rol... is temporary
                train_op, sync_op, self.grads = gradient_exchange(self.loss, agent_vars, shared_vars, decayed_lr)  # self.loss is temporary
                with tf.control_dependencies([train_op]):
                    self.train_op = tf.group(sync_op, global_step_increment)

def train(server):
    env_ = gym.make(FLAGS.env)
    num_actions = env_.action_space.n

    with tf.Graph().as_default():  # Suspect not needed.
        worker = Worker(FLAGS.task, num_actions, env_)

        # Saving only done by "chief" task.
        if FLAGS.task == 0 and FLAGS.save_progress:
            if not os.path.exists(FLAGS.model_directory):
                os.mkdir(FLAGS.model_directory)

            variables_to_save = worker.shared_vars
            saver = tf.train.Saver(variables_to_save)

        with tf.Session(server.target) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            time.sleep(1)
            sess.run(worker.env_reset)
            sess.run(worker.reset_pstate)
            T = 0
            t = 0
            num_opt_steps = FLAGS.train_steps // FLAGS.unroll_length
            evaluate_every = num_opt_steps / (FLAGS.num_workers * 10)
            print("Commenced training.")
            start_time = time.time()
            eps = 0
            obs_store = []
            while T < num_opt_steps:
                # _, d, T = sess.run([worker.train_op, worker.dones, worker.global_step])
                allo, d, fvs = sess.run([worker.rollout_outputs, worker.dones, worker.fv])
                obs_store.append(allo[0][:FLAGS.unroll_length, :, :, -1])
                # d_from_allo = allo[-1]
                # r = allo[-3]
                # print('Loss: ', loss)
                # print('Dones: ', d)
                # print('Reward: ', r)
                print(allo[-2])
                print(fvs[-2])
                # FAILING FOR
                # VALUES
                # REWARDS
                # LOGITS
                # ACTIONS

                # print(np.sum(np.reshape(allo[0], [FLAGS.unroll_length+1,-1]),axis=1))
                # print(np.sum(fvs[0]))
                t += 1
                # if t==2: break
                if np.sum(d) > 0:
                    break
                    eps += 1
                    # print('EPISODE ENDED -------------------------')
                    if eps == 2: break
                    # # These can just be grouped...
                    sess.run(worker.env_reset)
                    sess.run(worker.reset_pstate)
                # obs_store.append(allo[0][:5, :, :, -1])
            all_obs = np.array(obs_store)
            all_obs = all_obs.reshape(-1, 84, 84)
            print(all_obs.shape)
            # animate
            def animate_image(image_arr):
                fig = plt.figure()
                all_frames = [[plt.imshow(m)] for m in image_arr]
                ani = animation.ArtistAnimation(fig, all_frames, interval=200, blit=False, repeat=True)
                plt.show()
            animate_image(all_obs)

                # Evaluate on 30 test episodes every 10% of training progress.
                # if t % evaluate_every == 0:
                #     eps = 0
                #     total_rewards = 0
                #     sess.run(worker.env_reset)
                #     sess.run(worker.reset_pstate)
                #     while eps < FLAGS.test_eps:
                #         outp = sess.run(worker.rollout_outputs)
                #         r, d = outp[-3], outp[-1]
                #         total_rewards += np.sum(r[:-1])
                #         if np.sum(d) > 0:
                #             sess.run(worker.env_reset)
                #             sess.run(worker.reset_pstate)
                #             eps += 1
                #     print("Average Reward: {:.2f}".format(total_rewards/FLAGS.test_eps))

            print('Time taken: {}'.format(time.time() - start_time))
            # Temporarily have saving done at end of training, otherwise should be done incrementally.
            if FLAGS.task == 0 and FLAGS.save_progress:
                saver.save(sess, FLAGS.model_directory + 'model.checkpoint', global_step=T)

def test():
    if not os.path.exists(FLAGS.model_directory):
        raise ValueError('Model directory does not exist.')

    env_ = gym.make(FLAGS.env)


    with tf.Graph().as_default():
        env = TFEnv(env_)
        env_reset = env.reset()
        num_actions = env_.action_space.n
        rollout_outputs = rollout(env, num_actions)
        sess = tf.Session()

        # To initialise local and env variables, we initialise all variables,
        # and then load the saved trainable variables.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Load model.
        saved_variables = {'global/' + var.op.name : var for var in tf.trainable_variables()}
        saver = tf.train.Saver(saved_variables)
        chckpoint = tf.train.get_checkpoint_state(FLAGS.model_directory)
        saver.restore(sess, chckpoint.model_checkpoint_path)

        if FLAGS.animate: all_obs = []
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
            if FLAGS.animate:
                all_obs.append(outp[0][:FLAGS.unroll_length, :, :, -1])
        print("Average Reward: {:.2f}".format(total_rewards/FLAGS.test_eps))
        if FLAGS.animate:
            all_obs = np.array(all_obs)
            all_obs = all_obs.reshape(-1, 84, 84)
            # Animate.
            def animate_image(image_arr):
                fig = plt.figure()
                all_frames = [[plt.imshow(m)] for m in image_arr]
                ani = animation.ArtistAnimation(fig, all_frames, interval=50, blit=False, repeat=True)
                plt.show()
            animate_image(all_obs)

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
    if FLAGS.training:
        cluster = build_cluster_def(FLAGS.num_workers, 1)

        if FLAGS.job_name == 'worker':
            server = tf.train.Server(
                cluster, job_name="worker", task_index=FLAGS.task,
                config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
            train(server)
        else:
            server = tf.train.Server(cluster, job_name="ps", task_index=FLAGS.task,
                                     config=tf.ConfigProto(device_filters=["/job:ps"]))
            server.join()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
