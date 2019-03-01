import tensorflow as tf

from losses import value_loss, compute_return, policy_gradient, entropy

nest = tf.contrib.framework.nest


def advantage_calc(rewards, discounts, values):
    true_returns = compute_return(rewards, discounts, values)
    return true_returns - values[:-1]


def reward_fn(env_rewards, *args):
    rewards = env_rewards
    rewards = tf.clip_by_value(rewards, -1.0, 1.0)
    return rewards


def loss_calculation(rollout_outputs, action_space, constants):
    env_outputs, _, agent_outputs = rollout_outputs
    agent_outputs = nest.map_structure(lambda t: tf.squeeze(t), agent_outputs)

    values = agent_outputs.values

    # Subset all.
    env_outputs, agent_outputs = nest.map_structure(
        lambda t: t[:-1], (env_outputs, agent_outputs))
    dones = env_outputs.done

    env_rewards = env_outputs.reward
    rewards = reward_fn(env_rewards)

    gamma = tf.constant(constants.gamma, tf.float32)
    discounts = tf.to_float(~dones) * gamma

    advantages = advantage_calc(rewards, discounts, values)
    value_pred_loss = value_loss(advantages)

    pg_loss = policy_gradient(
        agent_outputs.logits, agent_outputs.action, advantages)
    entropy_loss = entropy(agent_outputs.logits)
    return (pg_loss
            - constants.beta_e * entropy_loss
            + constants.beta_v * value_pred_loss)


def gradient_exchange(loss, from_vars, to_vars, optimiser, constants):
    gradients = tf.gradients(loss, from_vars)
    if constants.grad_clip > 0:
        gradients, _ = tf.clip_by_global_norm(gradients, constants.grad_clip)
    train_op = optimiser.apply_gradients(zip(gradients, to_vars))
    return train_op
