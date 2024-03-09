import torch
import numpy as np
import math


class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """

    def __init__(self, trajectories, data_fields, device, hp):

        # Combine multiple trajectories into
        self.trajectories = {key: value for key, value in trajectories.items()}
        self.batch_len = hp.recurrent_seq_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - self.batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len = np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = hp.batch_size
        self.TrajectorBatch = data_fields

    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size)
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False)
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins=self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1
            return self.TrajectorBatch(**{key: value[eps_idx, series_idx] for key, value
                                     in self.trajectories.items() if key in self.TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)


def split_trajectories_episodes(trajectory_tensors, parallel_rollouts):
    """
    Split trajectories by episode.
    """
    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
    for i in range(parallel_rollouts):
        terminals_tmp = trajectory_tensors["terminals"].clone()
        terminals_tmp[0, i] = 1
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

        split_lens = split_points[1:] - split_points[:-1]
        split_lens[0] += 1

        len_episode = [split_len.item() for split_len in split_lens]
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values":
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)
    return trajectory_episodes, len_episodes