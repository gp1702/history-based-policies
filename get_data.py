import torch
from dataclasses import dataclass

@dataclass
class TrajectorBatch:
    """
    Dataclass for storing data batch.
    Names of these tensors need to be same as that used in function gather_trajectories.
    """
    states: torch.tensor
    actions: torch.tensor
    a_ini: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    actor_z_hidden_states: torch.tensor
    critic_hidden_states: torch.tensor
    next_states: torch.tensor
    rewards: torch.tensor


def gather_trajectories(input_data):
    """
    Gather policy trajectories from gym environment.
    """

    # Unpack inputs.
    env = input_data["env"]
    actor = input_data["actor"]
    critic = input_data["critic"]
    hp = input_data["hp"]
    GATHER_DEVICE = input_data['device']
    _MIN_REWARD_VALUES = torch.full([hp.parallel_rollouts], hp.min_reward)

    # Initialise variables.
    obsv = env.reset()
    a_ini = torch.ones(hp.parallel_rollouts, env.action_space[0].shape[0])

    trajectory_data = {"states": [],
                       "actions": [],
                       "a_ini": [],
                       "action_probabilities": [],
                       "rewards": [],
                       "true_rewards": [],
                       "values": [],
                       "terminals": [],
                       "actor_z_hidden_states": [],
                       "critic_hidden_states": [],
                       "next_states": []}

    terminal = torch.ones(hp.parallel_rollouts)

    with torch.no_grad():

        '''initialise hidden states for z'''
        actor.get_z_init_state(hp.parallel_rollouts, GATHER_DEVICE)

        '''initialise hidden states for critic'''
        critic.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)

        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(hp.rollout_steps):

            '''store hidden states for z'''
            trajectory_data["actor_z_hidden_states"].append(actor.z_hidden_layer.squeeze(0).cpu())

            '''store hidden states for critic'''
            trajectory_data["critic_hidden_states"].append(critic.hidden_layer.squeeze(0).cpu())

            '''store initial action in buffer'''
            trajectory_data['a_ini'].append(a_ini)

            # Choose next action

            state = torch.tensor(obsv, dtype=torch.float32)
            trajectory_data["states"].append(state)

            value = critic(state=state.unsqueeze(0).to(GATHER_DEVICE),
                           a_ini=a_ini.unsqueeze(0).to(GATHER_DEVICE),
                           terminal=terminal.to(GATHER_DEVICE))

            trajectory_data["values"].append(value.squeeze(1).cpu())

            action_dist, _ = actor(state=state.unsqueeze(0).to(GATHER_DEVICE),
                                a_ini=a_ini.unsqueeze(0).to(GATHER_DEVICE),
                                terminal=terminal.to(GATHER_DEVICE))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not actor.continuous_action_space:
                action = action.squeeze(1)

            trajectory_data["actions"].append(action.cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

            # Step environment
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)

            next_state = torch.tensor(obsv, dtype=torch.float32)

            trajectory_data["next_states"].append(next_state)

            a_ini = action.cpu()

            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_MIN_REWARD_VALUES, torch.tensor(reward).float())

            trajectory_data["rewards"].append(transformed_reward)
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal)

        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        value = critic(state=state.unsqueeze(0).to(GATHER_DEVICE),
                       a_ini=a_ini.unsqueeze(0).to(GATHER_DEVICE),
                       terminal=terminal.to(GATHER_DEVICE))
        # Future value for terminal episodes is 0.
        trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors