from abc import abstractmethod, ABC
from collections import defaultdict
import typing as tp

import torch
import torch.nn as nn


class BaseEnv(ABC):
    def __init__(self):
        self.metrics: tp.DefaultDict[str, tp.List[tp.Any]] = defaultdict(list)

    @abstractmethod
    def reset(self):
        """
        Base method for restarting session
        :return: Initial state s
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, a: tp.Any) -> tp.Any:
        """
        Base method for stepping in sessions
        :param a: Made action
        :return:
            next_s -- next state
            r -- collected reward
            done -- is session ended
        """
        raise NotImplementedError()

    def add_summary_scalar(self, name: str, value: tp.Any) -> None:
        """
        Save metrics for iterations by trajectories
        :param name: Metric label
        :param value: Some value which can be visualized
        :return: None
        """
        self.metrics[name].append(value)

    def get_values(self, name: str) -> tp.List[tp.Any]:
        """
        Return values for given metric
        :param name: Metric label
        :return: list of values
        """
        return self.metrics[name]


class BaseAgent(ABC):
    def __init__(self):
        self.model: tp.Optional[nn.Module] = None

    @abstractmethod
    def get_action(self, s: torch.Tensor) -> tp.Dict[str, tp.Any]:
        """
        Base method for making actions
        :param s: Current state
        :return: Chosen action + any additional info
        """
        raise NotImplementedError()


class BaseAlgo(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def loss(self, trajectory: tp.Dict[str, tp.Any]) -> torch.Tensor:
        """
        Base method for loss calculating
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: torch.Tensor with one element to make gradient step
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, trajectory) -> None:
        """
        Base method for making algorithm step with respect to given trajectory
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        raise NotImplementedError()
