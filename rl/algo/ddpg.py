import numpy as np
import pandas as pd
import typing as tp

import torch
import torch.nn as nn

from a2c import A2CEnv, A2CAgent, A2C


class DDPGEnv(A2CEnv):
    def __init__(self, tables: tp.List[pd.DataFrame]):
        """
        Initialize environment for DDPG, which is the same with A2C env
        :param tables: list of tables with features
        """
        super().__init__(tables)


class DDPGFirstAgent(A2CAgent):
    def __init__(self, model: nn.Module):
        """
        Initialize first DDPG agent, the same with A2C agent
        :param model: neural network used in agent
        """
        super().__init__(model)


class DDPGSecondAgent(A2CAgent):
    def __init__(self, model: nn.Module, tau=0.1):
        """
        Initialize second DDPG agent, which is A2C agent, but without gradient computing
        :param model: neural network used in agent, layers are equal to first agent model layers
        :param tau: coefficient for shifting params of network
        """
        super().__init__(model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.tau = tau

    def soft_update(self, other_model: nn.Module) -> None:
        """
        Makes soft update of params with rule
        source_param <- target_param * tau + source_param * (1 - tau)
        :param other_model: target neural network
        :return: None
        """
        for self_param, target_param in zip(self.model.parameters(), other_model.parameters()):
            self_param.data.copy_(target_param.data * self.tau + self_param.data * (1 - self.tau))


class DDPG(A2C):
    def __init__(self,
                 first_agent: DDPGFirstAgent,
                 second_agent: DDPGSecondAgent,
                 optimizer: torch.optim.optimizer,
                 env: A2CEnv,
                 value_loss_coef: float = 0.25,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        """
        Construct DDPG algorithm
        :param first_agent: Used first agent for making gradient steps
        :param second_agent: Used second agent for value function estimation
        :param optimizer: Used optimizer from `torch.optim`
        :param env: Used environment
        :param value_loss_coef: Positive coefficient for value loss
        :param entropy_coef: Positive coefficient for entropy part of loss
        :param max_grad_norm: Limit of grad norm for pytorch clipping
        """
        super().__init__(first_agent, optimizer, env, value_loss_coef, entropy_coef, max_grad_norm)
        self.second_agent = second_agent

    def step(self, trajectory) -> None:
        """
        Implement step for DDPG algorithm
            1) Loss computed for partial trajectory
            2) Gradient step is made
            3) Soft update is applied for second agent
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        self.env.add_summary_scalar("rewards", np.mean(trajectory['rewards']))
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.second_agent.soft_update(self.agent.model)
