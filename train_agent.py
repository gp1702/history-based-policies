from agent_1 import Actor
from agent_1 import Critic_Coordinator as Critic
from agent_1 import Psi
import hyperparams as h
import torch
import save_load_models
import get_data
import buffer
import utils
import gym
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
import torch.nn.functional as F
import re
import time
import os
from dataclasses import dataclass
import json

hp = h.HyperParameters()

ENV = hp.ENV
WORKSPACE_PATH = "/home/gandharv/scratch/common-info-ac/ais/Results/"
# WORKSPACE_PATH = "/home/gandharv/projects/rrg-bengioy-ad/gandharv/common-info-ac/ais/Results/"
EXPERIMENT_NAME = hp.exp_name
res_dir = hp.dir
container = 'run_'+str(hp.name)+'_'
check_point_container = str(hp.name)
_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")
log_dir = os.path.join(WORKSPACE_PATH, res_dir, EXPERIMENT_NAME, str(hp.name))
BASE_CHECKPOINT_PATH = os.path.join(WORKSPACE_PATH, 'checkpoints',
                                    res_dir, EXPERIMENT_NAME, check_point_container)
os.makedirs(log_dir, exist_ok=True)

ENV_MASK_VELOCITY = False

'''Save metrics for viewing with tensorboard.'''
SAVE_METRICS_TENSORBOARD = True

'''Save actor & critic parameters for viewing in tensorboard.'''
SAVE_PARAMETERS_TENSORBOARD = False

'''Save training state frequency in PPO iterations.'''
CHECKPOINT_FREQUENCY = hp.ckpt_freq

'''Step env asynchronously using multiprocess or synchronously.'''
ASYNCHRONOUS_ENVIRONMENT = False

'''Force using CPU for gathering trajectories.'''
FORCE_CPU_GATHER = True

batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
print(f"batch_count: {batch_count}")
assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"


def get_env_space(env):
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    env = gym.make(env)
    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    obsv_dim = env.observation_space.shape[0]
    return obsv_dim, action_dim, continuous_action_space


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """

    def __init__(self, env):
        super(MaskVelocityWrapper, self).__init__(env)
        if ENV == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif ENV == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif ENV == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1, ])
        elif ENV == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1, ])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return observation * self.mask

@dataclass
class StopConditions:
    """
    Store parameters and variables used to stop training.
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    if hp.ENV == "Ant-v2" or hp.ENV == "Humanoid-v2" or hp.ENV == "HumanoidStandup-v2":
        max_iterations = 2000
    else:
        max_iterations: int = hp.max_iterations


torch.set_num_threads(15)

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GATHER_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

writer = SummaryWriter(log_dir=log_dir)

'''Initialise Models'''

obsv_dim, action_dim, continuous_action_space = get_env_space(hp.ENV)

actor = Actor(obsv_dim,
              action_dim,
              continuous_action_space=continuous_action_space,
              trainable_std_dev=hp.trainable_std_dev,
              init_log_std_dev=hp.init_log_std_dev)

critic = Critic(obsv_dim, action_dim)

psi = Psi(ais_dim=hp.hidden_size)

actor_optimizer = optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)

critic_optimizer = optim.AdamW(critic.parameters(), lr=hp.critic_learning_rate)

"""optimiser for psi network"""
psi_optimizer = optim.AdamW(list(psi.parameters()) +
                            list(actor.parameters()), lr=hp.psi_learning_rate)

stop_conditions = StopConditions()

data = {'Actor': actor,
        'Critic': critic,
        'State_dim': obsv_dim,
        'Action_dim': action_dim,
        'continuous_action_space': continuous_action_space,
        'Actor_optimizer': actor_optimizer,
        'Critic_optimizer': critic_optimizer,
        'Checkpoint_path': BASE_CHECKPOINT_PATH,
        'device': TRAIN_DEVICE,
        'stop_conditions': stop_conditions,
        'hp': hp,
        'psi': psi,
        'psi_optimizer': psi_optimizer
        }

actor, critic, psi, actor_optimizer, critic_optimizer, psi_optimizer, iteration, stop_conditions \
                            = save_load_models.start_or_resume_from_checkpoint(data=data)

'''Main training loop'''


def train_model(actor, critic, psi, actor_optimizer, critic_optimizer, psi_optimizer,
                iteration, stop_conditions, target_kl=0.03):
    """Vector environment manages multiple instances of the environment.
    A key difference between this and the standard gym environment is it automatically resets.
    Therefore when the done flag is active in the done vector the corresponding state is the first new state."""

    all_kl_divs = []
    env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)

    """Normalise observations"""
    env = utils.VecNormalize(env)

    if ENV_MASK_VELOCITY:
        env = MaskVelocityWrapper(env)

    while iteration < stop_conditions.max_iterations+2:

        actor = actor.to(GATHER_DEVICE)
        critic = critic.to(GATHER_DEVICE)
        start_gather_time = time.time()

        """Sampling data from the environment"""

        input_data = {"env": env,
                      "actor": actor,
                      "critic": critic,
                      "discount": hp.discount,
                      "gae_lambda": hp.gae_lambda,
                      "hp": hp,
                      "device": GATHER_DEVICE}

        trajectory_tensors = get_data.gather_trajectories(input_data)
        trajectory_episodes, len_episodes = buffer.split_trajectories_episodes(trajectory_tensors,
                                                                               parallel_rollouts=hp.parallel_rollouts)
        trajectories = utils.pad_and_compute_returns(trajectory_episodes, len_episodes, hp=hp)

        """Calculate mean reward."""
        complete_episode_count = trajectories["terminals"].sum().item()
        terminal_episodes_rewards = (
                trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum()
        mean_reward = terminal_episodes_rewards / complete_episode_count

        """Calculating discounted return"""

        dis_ret = [utils.get_discounted_return(trajectories["true_rewards"][i], hp.discount) for i in
                   range(trajectories["true_rewards"].shape[0])]

        disc_return = np.mean(dis_ret)

        """Check stop conditions."""

        if mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = mean_reward
            stop_conditions.fail_to_improve_count = 0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

        trajectory_dataset = buffer.TrajectoryDataset(trajectories=trajectories,
                                                      data_fields= get_data.TrajectorBatch,
                                                      device=TRAIN_DEVICE,
                                                      hp=hp)
        end_gather_time = time.time()
        start_train_time = time.time()

        actor = actor.to(TRAIN_DEVICE)
        critic = critic.to(TRAIN_DEVICE)
        psi = psi.to(TRAIN_DEVICE)

        """Train actor and critic."""

        for epoch_idx in range(hp.ppo_epochs):
            for batch in trajectory_dataset:
                approx_kl_divs = []

                '''Fetch hidden cell for z'''

                actor.z_hidden_layer = batch.actor_z_hidden_states[:1].to(TRAIN_DEVICE)

                # Update actor
                actor_optimizer.zero_grad()
                action_dist, ais = actor(state=batch.states[-1, :].unsqueeze(dim=0).to(TRAIN_DEVICE),
                                         a_ini=batch.a_ini[-1, :].unsqueeze(dim=0).to(TRAIN_DEVICE))

                """Action dist runs on cpu as a workaround to CUDA illegal memory access"""

                action_probabilities = action_dist.log_prob(batch.actions[-1, :].to("cpu")).to(TRAIN_DEVICE)

                """Sampling the next ais"""
                _, next_ais = actor(state=batch.next_states[-1, :].unsqueeze(dim=0).to(TRAIN_DEVICE),
                                    a_ini=batch.actions[-1, :].unsqueeze(dim=0).to(TRAIN_DEVICE))
                target_ais = next_ais.detach().clone()

                """Sampling values from psi network"""
                predicted_reward, next_obs, weights = psi(ais)

                """Compute probability ratio from probabilities in logspace."""

                probabilities_ratio = torch.exp(action_probabilities -
                                                batch.action_probabilities[-1, :].to(TRAIN_DEVICE))
                surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :].to(TRAIN_DEVICE)
                surrogate_loss_1 = torch.clamp(probabilities_ratio, 1. - hp.ppo_clip,
                                               1. + hp.ppo_clip) * batch.advantages[-1, :].to(TRAIN_DEVICE)
                surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(
                    hp.entropy_factor * surrogate_loss_2)
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), hp.max_grad_norm)
                actor_optimizer.step()

                """Update critic"""

                critic_optimizer.zero_grad()

                critic.hidden_layer = batch.critic_hidden_states[:1].to(TRAIN_DEVICE)

                values = critic(state=batch.states[-1, :].unsqueeze(dim=0).to(TRAIN_DEVICE),
                                a_ini=batch.a_ini[-1, :].unsqueeze(dim=0).to(TRAIN_DEVICE))
                critic_loss = F.mse_loss(batch.discounted_returns[-1, :].to(TRAIN_DEVICE),
                                         values.squeeze(1))

                critic_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                critic_optimizer.step()

                approx_kl_divs.append(torch.mean(batch.action_probabilities[-1, :].to(TRAIN_DEVICE) -
                                                 action_probabilities).detach().to(GATHER_DEVICE).numpy())

                """Updating psi network"""
                psi_optimizer.zero_grad()
                reward_loss = F.smooth_l1_loss(predicted_reward,
                                               batch.rewards.view(-1, 1).to(TRAIN_DEVICE))

                # observation_loss = utils.sinkhorn_divergence(x=target_ais,
                #                                              y=next_obs,
                #                                              beta=weights,
                #                                              device=TRAIN_DEVICE)

                observation_loss = utils.kernel_distance(x=target_ais,
                                                         y=next_obs,
                                                         beta=weights,
                                                         device=TRAIN_DEVICE,
                                                         mode=hp.kernel)

                psi_loss = 0.4*reward_loss + (1-0.4)*observation_loss

                psi_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(psi.parameters(), hp.max_grad_norm)
                psi_optimizer.step()

            all_kl_divs.append(np.mean(approx_kl_divs))
            if target_kl is not None and np.mean(approx_kl_divs) > 1.5 * target_kl:
                print(
                    f"\nEarly stopping at step {epoch_idx} due to reaching max kl: {np.mean(approx_kl_divs):.2f}, the limit is 0.3\n")
                break

        if SAVE_METRICS_TENSORBOARD:
            writer.add_scalar("complete_episode_count", complete_episode_count, iteration)
            writer.add_scalar("Approx KL divergence", np.mean(all_kl_divs), iteration)
            writer.add_scalar("total_reward", mean_reward, iteration)
            writer.add_scalar("Observation_Loss", observation_loss, iteration)
            writer.add_scalar("Reward_Loss", reward_loss, iteration)
            writer.add_scalar("total_discounted_reward", disc_return, iteration)
            writer.add_scalar("actor_loss", actor_loss, iteration)
            writer.add_scalar("critic_loss", critic_loss, iteration)
            writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), iteration)
            writer.add_scalar("batch_size", hp.batch_size, iteration)
            writer.add_scalar("actor_lr", hp.actor_learning_rate, iteration)
            writer.add_scalar("critic_lr", hp.critic_learning_rate, iteration)
        if SAVE_PARAMETERS_TENSORBOARD:
            save_load_models.save_parameters(writer, "actor", actor, iteration)
            save_load_models.save_parameters(writer, "value", critic, iteration)
        if iteration % CHECKPOINT_FREQUENCY == 0:
            save_load_models.save_checkpoint(actor,
                                             critic,
                                             psi,
                                             actor_optimizer,
                                             critic_optimizer,
                                             psi_optimizer,
                                             iteration,
                                             stop_conditions,
                                             hp,
                                             ENV_MASK_VELOCITY,
                                             BASE_CHECKPOINT_PATH)
        end_train_time = time.time()
        print(
            f"Statistics Iteration {iteration}:"
            f"\nMean reward: {mean_reward},",
            f"\nDisounted Return: {disc_return},"
            f"\nMean Entropy: {torch.mean(surrogate_loss_2)}, " +
            f"\nApproximate Kl: {np.mean(all_kl_divs)},"
            f"\nComplete_episode_count: {complete_episode_count}, "
            f"\nGather time: {end_gather_time - start_gather_time:.2f}s, " +
            f"\nTrain time: {end_train_time - start_train_time:.2f}s",
            f"\nTotal time: {(end_gather_time - start_gather_time) + (end_train_time - start_train_time):.2f}s",
            "\n-------------------------------------")
        iteration += 1

    return stop_conditions.best_reward


score = train_model(actor, critic, psi, actor_optimizer, critic_optimizer, psi_optimizer, iteration, stop_conditions)

"""saving arguments"""
file_path = os.path.join(log_dir, 'config.txt')
with open(file_path, 'w') as f:
    json.dump(hp.__dict__, f, indent=2)