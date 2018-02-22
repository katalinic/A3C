import tensorflow as tf
from a3c import Worker
import gym
import preprocessing
import time
from gym import wrappers

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def train(args, server):

    T = 0
    max_global_steps = 100000000

    #breakout specific
    env = gym.make('BreakoutDeterministic-v4')
    #added for recording - this will record every episode; monitor default is perfect cubes, 0 1 8 27 etc.
    if args.record=="True":
        def video_callable_fn(ep):
            return True
        env = wrappers.Monitor(env, './rec/breakout', video_callable=video_callable_fn, force=True)
    env = preprocessing.EnvWrapper(env, True, 4)

    obs_size = (84,84,4)
    action_size = 4

    a3c = Worker(args.task, obs_size, action_size, env)

    saver = tf.train.Saver()
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    def init_fn(ses):
        ses.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                                 logdir='./tmp/',
                                 saver=saver,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 ready_op=tf.report_uninitialized_variables(variables_to_save),
                                 global_step=a3c.global_step,
                                 save_model_secs=7200)

    sess_config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    with sv.managed_session(server.target, config=sess_config) as sess, sess.as_default():

        if args.train!="inference":
            reward_tracker = []
            while (T<max_global_steps):
                T = sess.run(a3c.global_step)
                a3c.interaction(sess)
                eval_ind = 0
                '''every 1 million frames evaluate performance on 30 test episodes'''
                if a3c.t // 1000000 > eval_ind:
                    eval_ind = a3c.t // 1000000
                    cumulative_reward = 0
                    for j in range(30):
                        obs = env.reset()
                        episode_reward = 0
                        done = False
                        noop_counter = 0
                        while (not done):
                            action = a3c.agent.policy_and_value(sess, obs.reshape(1,*obs_size), 'act')
                            obs, reward, done, info = env.step(action, inference=True)
                            episode_reward+=reward
                        cumulative_reward += episode_reward
                    print ("Episode {}, Agent Time Step {}, Global Time Step {} - Rewards : {} ".format(a3c.episodes,a3c.t,T, cumulative_reward/30.))
                    reward_tracker.append(cumulative_reward/30.)
            np.save('reward_progress',np.array(reward_tracker))
        else:
            print ("Running inference.")
            print (sess.run(a3c.global_step))
            for test_ep in range(30):
                obs = env.reset()
                episode_reward = 0
                done = False
                noop_counter = 0
                while (not done):
                    env.render()
                    time.sleep(0.01)
                    action = a3c.agent.policy_and_value(sess, obs.reshape(1,*obs_size), 'act')
                    obs, reward, done, info = env.step(action, inference=True)
                    episode_reward+=reward
                print ("Test episode {}, Episode Reward: {}".format(test_ep, episode_reward))

    sv.stop()
