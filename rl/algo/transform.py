import typing as tp
from abc import ABC, abstractmethod
import numpy as np

import torch

from a2c import A2CAgent
from ddpg import DDPGSecondAgent


class Transformer(ABC):
    @abstractmethod
    def __call__(self, trajectory: tp.Dict[str, tp.Any]) -> None:
        """
        Base method for changing trajectory inplace
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        raise NotImplementedError()


class CastLists(Transformer):
    def __call__(self, trajectory: tp.Dict[str, tp.Any]) -> None:
        """
        Stacks info if environment was reset
        Cast lists to np.ndarray and torch.Tensors
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        for k, v in trajectory.items():
            if k == 'state':
                continue
            elif type(v[0]) is torch.Tensor:
                trajectory[k] = torch.squeeze(torch.stack(v))
            else:
                trajectory[k] = np.array(v)


class EstimatorValues(Transformer, ABC):
    def __init__(self, gamma: float, predict_all: bool = False):
        """
        Initialize gamma
        :param gamma: discount factor from Bellman equation
        :param predict_all: bool param if we need to predict all Q-values by
        """
        self.gamma = gamma
        self.predict_all = predict_all

    @abstractmethod
    def estimate_value(self, trajectory: tp.Dict[str, tp.Any], step: int) -> tp.Union[torch.Tensor, float]:
        raise NotImplementedError()

    def __call__(self, trajectory):
        """
        Computes estimation of Q(s_t, a_t)
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        assert not (self.predict_all and 'next_observations' not in trajectory), \
            f'If predict_all param is set, when estimation value function computed using next_observations'
        value_target: float = 0.
        env_steps: int = len(trajectory['rewards'])
        rewards: torch.Tensor = torch.tensor(trajectory['rewards'], dtype=torch.float32)
        dones: torch.Tensor = torch.tensor(trajectory['resets'], dtype=torch.float32)
        trajectory['value_targets'] = [0] * env_steps
        for step in range(env_steps - 1, -1, -1):
            assert not (self.predict_all and (step == env_steps - 1 or dones[step])), \
                f'If predict_all param is set, when trajectories cannot end in sampled states'
            if step == env_steps - 1 or dones[step] or self.predict_all:
                value_target = self.estimate_value(trajectory, step)
                if self.predict_all:
                    # if predict_all is set, make step back from Bellman equation
                    value_target = rewards[step] + value_target * self.gamma
            else:
                # update with discount factor, using Bellman equation
                value_target = rewards[step] + value_target * self.gamma
            trajectory['value_targets'][step] = value_target


class ReinforceValues(EstimatorValues):
    def __init__(self, gamma):
        """
        Initialize gamma
        :param gamma: discount factor from Bellman equation
        """
        super().__init__(gamma)

    def estimate_value(self, trajectory: tp.Dict[str, tp.Any], step: int) -> tp.Union[torch.Tensor, float]:
        """
        Implements logic of computing Q'(s_t, a_t) = sum_{t' >= t} gamma^t r_{t'}
        Here Q(s_{last}, a_{last}) is equal to zero
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :param step: number of step in trajectory
        :return: zero prediction
        """
        return trajectory['rewards'][step]


class A2CValues(EstimatorValues):
    def __init__(self, agent: A2CAgent, gamma: float, predict_all: bool = False):
        """
        Initialize A2C agent (need to predict values)
        :param agent: used A2C agent
        :param gamma: discount factor from Bellman equation
        :param predict_all: bool param if we need to predict all Q-values by
        """
        super().__init__(gamma, predict_all=predict_all)
        self.agent: A2CAgent = agent

    def estimate_value(self, trajectory: tp.Dict[str, tp.Any], step: int) -> tp.Union[torch.Tensor, float]:
        """
        Implement logic Q'(s_t, a_t) = r_t + gamma * Q'(s_{t + 1}, a_{t + 1})
        Here Q'(s_{last}, a_{last}) is predicted by critic
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :param step: number of step in trajectory
        :return: zero prediction
        """
        return self.agent.get_action(trajectory['observations'][step])['values']


class DDPGValues(A2CValues):
    def __init__(self, second_agent: DDPGSecondAgent, gamma: float):
        """
        Initialize A2CValues transformer, where all value functions are predicted by critic
        :param second_agent: agent for estimation value functions
        :param gamma: discount factor from Bellman equation
        """
        super().__init__(second_agent, gamma, predict_all=True)

    def estimate_value(self, trajectory: tp.Dict[str, tp.Any], step: int) -> tp.Union[torch.Tensor, float]:
        """
        Implement logic Q'(s_t, a_t) = r_t + gamma * Q'(s_{t + 1}, a_{t + 1})
        Here Q'(s_{last}, a_{last}) is predicted by critic
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :param step: number of step in trajectory
        :return: zero prediction
        """
        return self.agent.get_action(trajectory['next_observations'][step])['values']


class SampleBuffer(Transformer):
    def __init__(self, batch_size: int):
        """
        Initialize batch size for sampling part of trajectory
        :param batch_size: integer representing batch size
        """
        self.batch_size = batch_size

    def __call__(self, trajectory):
        """
        Sample batch size of objects, add next observations for sampled states
        CONDITION: for sampled objects next state exist in trajectory
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        not_done = np.logical_not(trajectory['resets'])
        indexes = np.random.choice(
            np.array(np.arange(len(trajectory['actions'])))[not_done][:-1],
            size=self.batch_size
        )
        trajectory['next_observations'] = trajectory['observations'][indexes + 1]
        for k, v in trajectory.items():
            if k == 'state' or k == 'next_observations':
                continue
            trajectory[k] = trajectory[k][indexes]
