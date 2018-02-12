import tensorflow as tf
from a3c import Worker
import gym
import preprocessing

def train(args, server):

    T = 0
    max_global_steps = 100000000

    #simple cartpole env
    # env = gym.make('CartPole-v0')
    # obs_size = 4
    # action_size = 2

    #full scale breakout setting
    env = gym.make('BreakoutDeterministic-v4')
    env = preprocessing.EnvWrapper(env, True, 4)
    obs_size = (84,84,4)
    action_size = 4

    a3c = Worker(args.task, obs_size, action_size, env)

    saver = tf.train.Saver()
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    def init_fn(ses):
        # logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                                 logdir='./tmp/',
                                 saver=saver,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 ready_op=tf.report_uninitialized_variables(variables_to_save),
                                 global_step=a3c.global_step,
                                 save_model_secs=1800)

    sess_config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    with sv.managed_session(server.target, config=sess_config) as sess, sess.as_default():

        while (T<max_global_steps):
            T = sess.run(a3c.global_step)
            '''testing section'''
            # if args.task != 0:
            #     a3c.interaction(sess)
            # #we allow 0 syncing
            # else:
            #     sess.run(a3c.sync_op)
            a3c.interaction(sess)

            '''every n episodes evaluate performance on 10 test episodes'''
            # if args.task == 0:
            if a3c.episodes % 50 == 0:
                # print ("evaluating performance")
                cumulative_reward = 0
                for j in range(10):
                    obs = env.reset()
                    episode_reward = 0
                    done = False
                    noop_counter = 0
                    while (not done):
                        #changed to local agent for testing
                        action = a3c.agent.policy_and_value(sess, obs.reshape(1,*obs_size), 'act')
                        if action != 1: noop_counter += 1
                        if noop_counter > 30:
                            action = 1
                            noop_counter = 0
                        obs, reward, done, info = env.step(action)
                        # print (action, done, info['ale.lives'])
                        episode_reward+=reward


                    cumulative_reward += episode_reward


                print ("Episode {}, Agent Time Step {}, Global Time Step {} - Rewards : {} ".format(a3c.episodes,a3c.t,T, cumulative_reward/10.))

    print (a3c.t, a3c.episodes)
    sv.stop()
