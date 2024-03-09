import torch
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.linalg import hankel
from abc import ABC, abstractmethod


def pad_and_compute_returns(trajectory_episodes, len_episodes, hp):
    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages and discounted_returns to trajectories.
    """

    episode_count = len(len_episodes)
    advantages_episodes, discounted_returns_episodes = [], []
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []

    for i in range(episode_count):
        single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))
        padded_trajectories["advantages"].append(
            torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                          values=trajectory_episodes["values"][i],
                                          discount=hp.discount,
                                          gae_lambda=hp.gae_lambda), single_padding)))
        padded_trajectories["discounted_returns"].append(
            torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                              discount=hp.discount,
                                              final_value=trajectory_episodes["values"][i][-1]), single_padding)))
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
    return_val["seq_len"] = torch.tensor(len_episodes)

    return return_val

def calc_discounted_return(rewards, discount, final_value):
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1, -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
    return discounted_returns


def compute_advantages(rewards, values, discount, gae_lambda):
    """
    Compute General Advantage.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1, -1):
        advs[i] = advs[i + 1] * multiplier + deltas[i]
    return advs[:-1]

def get_discounted_return(reward, discount):
    """
    Computing the discounted return for every state
    """
    g = polyval(discount, hankel(reward))
    return g[0]

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()


def KP(x, y, beta_j, kernel="energy", s=1.):
    """
    Computes K(x_i,y_j) @ β_j = \sum_j k(x_i-y_j) * β_j
    where k is a kernel function (say, a Gaussian) of deviation s.
    """
    x_i = x[:, None, :]  # Shape (N,d) -> Shape (N,1,d)
    y_j = y[None, :, :]  # Shape (M,d) -> Shape (1,M,d)
    xmy = x_i - y_j    # (N,M,d) matrix, xmy[i,j,k] = (x_i[k]-y_j[k])
    if kernel == "gaussian":
        k = torch.exp(- (xmy**2).sum(2) / (2*(s**2)))
    elif kernel == "laplace":
        k = torch.exp(- xmy.norm(dim=2) / s)
    elif kernel == "energy":
        k = - xmy.norm(dim=2)
    """Matrix-vector product"""
    return k @ beta_j


def scal(alpha, f):
    return torch.dot(alpha.view(-1), f.view(-1))


def kernel_scalar_product(alpha_i, x_i, beta_j, y_j, mode="energy", s=1.):
    kxy_beta = KP(x_i, y_j, beta_j, mode, s)
    return scal(alpha_i, kxy_beta)


def kernel_distance(x, y, beta, device, mode="energy", s=1.):
    batch_size = x.shape[0]
    alpha_i = torch.ones(size=beta.shape, dtype=torch.float32).to(device)
    d2 = (.5*kernel_scalar_product(alpha_i, x, alpha_i, x, mode, s) +
          .5*kernel_scalar_product(beta, y, beta, y, mode, s) -
          kernel_scalar_product(alpha_i, x, beta, y, mode, s))/batch_size
    return d2


def KP_log(x, y, beta_j_log, p=2, blur=1.):
    x_i = x[:, None, :]  # Shape (N,d) -> Shape (N,1,d)
    y_j = y[None, :, :]  # Shape (M,d) -> Shape (1,M,d)
    xmy = x_i - y_j    # (N,M,d) matrix, xmy[i,j,k] = (x_i[k]-y_j[k])
    if p == 2:
        c = - (xmy**2).sum(2) / (2*(blur**2))
    elif p == 1:
        c = - xmy.norm(dim=2) / blur
    return (blur**p)*(c + beta_j_log.view(1, -1)).logsumexp(1, keepdim=True)


D = lambda x : x.detach()


def sinkhorn_divergence(x, y, beta, device, p=2, blur=.01, scaling=.5):
    alpha = torch.ones(size=beta.shape, dtype=torch.float32).to(device)

    def cost(alpha_i=alpha, x_i=x, beta_j=beta, y_j=y, device=device):
        # ε-scaling heuristic (aka. simulated annealing):
        # let ε decrease across iterations, from 1 (=diameter) to the target value
        scales = [torch.tensor([np.exp(e)]).to(device) for e in
                  np.arange(0, np.log(blur), np.log(scaling))] + [blur]

        # 1) Solve the OT_ε(α,β) problem
        f_i, g_j = torch.zeros_like(alpha_i).to(device), torch.zeros_like(beta_j).to(device)
        for scale in scales:
            g_j = - KP_log(x=y_j, y=D(x_i), beta_j_log=D(f_i / scale ** p + alpha_i.log()),
                           p=p, blur=scale)
            f_i = - KP_log(x=x_i, y=D(y_j), beta_j_log=
            D(g_j / scale ** p + beta_j.log()), p=p, blur=scale)

        # 2) Solve the OT_ε(α,α) and OT_ε(β,β) problems
        scales_sym = [scale] * 3  # Symmetric updates converge very quickly
        g_i, f_j = torch.zeros_like(alpha_i).to(device), torch.zeros_like(beta_j).to(device)
        for scale in scales_sym:
            g_i = .5 * (g_i - KP_log(x=x_i, y=x_i, beta_j_log=g_i / scale ** p + alpha_i.log(),
                                     p=p, blur=scale))
            f_j = .5 * (f_j - KP_log(x=y_j, y=y_j, beta_j_log=f_j / scale ** p + beta_j.log(),
                                     p=p, blur=scale))
        # Final step, to get a nice gradient in the backprop pass:
        g_i = - KP_log(x=x_i, y=D(x_i), beta_j_log=D(g_i / scale ** p + alpha_i.log()), p=p, blur=scale)
        f_j = - KP_log(x=y_j, y=D(y_j), beta_j_log=D(f_j / scale ** p + beta_j.log()), p=p, blur=scale)

        # Return the "dual" cost :
        # S_ε(α,β) =        OT_ε(α,β)       - ½OT_ε(α,α) - ½OT_ε(β,β)
        #          = (〈α,f_αβ〉+〈β,g_αβ〉) -  〈α,g_αα〉 - 〈β,f_ββ〉
        return scal(alpha_i, f_i - g_i) + scal(beta_j, g_j - f_j)
    distance = cost(alpha, x, beta, y)
    return distance

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)


class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        # if self.ret_rms:
        #     self.ret_rms.update(self.ret)
        #     rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)