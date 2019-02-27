import tensorflow as tf

from sub_networks import encoder, head


class A3CAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def init_state(self):
        return tf.zeros(1)

    def build(self, env_output, prev_agent_state):
        encoder_output = encoder(env_output.obs)
        policy_output = head(encoder_output, self.action_space)
        return policy_output

    def update_state(self, agent_output, prev_agent_state, done):
        return tf.zeros(1)
