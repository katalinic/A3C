from collections import namedtuple

import tensorflow as tf

nest = tf.contrib.framework.nest

EnvOutput = namedtuple('EnvOutput', 'obs reward done')


def rollout(env, agent, unroll_length):
    init_env_output = EnvOutput(*env.reset())
    init_agent_state = agent.init_state()
    init_agent_output = agent.build(init_env_output, init_agent_state)

    def create_state(t):
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(
                t.op.name, initializer=t, use_resource=True)

    # Persistent variables.
    persistent_state = nest.map_structure(create_state, (
        init_env_output, init_agent_state, init_agent_output))

    first_values = nest.map_structure(
        lambda v: v.read_value(), persistent_state)

    def step(input_, unused_i):
        prev_env_output, prev_agent_state, _ = input_

        agent_output = agent.build(prev_env_output, prev_agent_state)

        env_output = EnvOutput(*env.step(agent_output.action))

        agent_state = agent.update_state(
            agent_output, prev_agent_state, env_output.done)

        return env_output, agent_state, agent_output

    outputs = tf.scan(
        step,
        tf.range(unroll_length),
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

    return full_outputs
