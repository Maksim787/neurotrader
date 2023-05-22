import random
import numpy as np
import pandas as pd
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEnv, BaseAgent, BaseAlgo


class A2CEnv(BaseEnv):
    def __init__(self, tables: tp.List[pd.DataFrame]):
        """
        Initialize Reinforce Environment
        :param tables: list of DataFrames for each ticker with features in rows
        """
        super().__init__()
        assert len(tables) > 0, f'Tables can not be empty'
        assert np.array(np.array([len(table) for table in tables]) == len(tables[0])).all(),\
            f'All tables must have the same length'
        self.num_columns: tp.List[str] = tables[0].select_dtypes([np.number]).columns
        self.number_periods: int = len(tables[0])
        self.n_securities: int = len(tables)
        self.tables: tp.List[pd.DataFrame] = tables
        self.index: tp.Optional[int] = None

    def _state(self, index: int) -> torch.Tensor:
        """
        Extract state associated with index
        Gather features from all numerical columns of tables
        :param index: common index of state for all tables
        :return: tensor merged from numerical features
        """
        state = np.hstack([np.array(self.tables[i][self.num_columns].iloc[index]) for i in range(self.n_securities)])
        return torch.tensor(state, dtype=torch.float32)

    def get_revenue(self, action: torch.Tensor) -> float:
        """
        Calculate sum action_i * return_i for all tickers (tables)
        Here returns are located at position `self.index` and column `close`
        :param action: Made action, as tensor of probabilities
        :return: float representing achieved reward
        """
        data = np.array([np.array(self.tables[i]['close']) for i in range(self.n_securities)])
        returns = data[:, self.index]
        return float(np.sum(action * returns))

    def _calculate_reward(self, action: torch.Tensor) -> float:
        """
        Calculate sum action_i * return_i / vol_i for all tickers (tables)
        Here returns are located at position `self.index` and column `close`
        Volatilises are at the same position and column `vol`
        :param action: Made action, as tensor of probabilities
        :return: float representing achieved reward
        """
        data = np.array([np.array(self.tables[i]['close']) for i in range(self.n_securities)])
        vols = np.array([self.tables[i].iloc[self.index]['vol'] for i in range(self.n_securities)])
        returns = data[:, self.index]
        return float(np.sum(action.detach().numpy() * returns / vols))

    def reset(self) -> torch.Tensor:
        """
        Reset index in range[0, len(tables[0]))
        :return: state associated with new index
        """
        self.index = random.randint(0, self.number_periods - 1)
        return self._state(self.index)

    def step(self, action: torch.Tensor) -> tp.Tuple[torch.Tensor, float, bool]:
        """
        Implementation of step by incrementing of `self.index`
        :param action: Made action, as tensor of probabilities
        :return:
            state  - represented by tensor
            reward - float reward for state with respect to action
            done   - was trajectory ended here or not
        """
        assert self.index is not None, f'Reset env firstly'
        self.index += 1
        reward: float = self._calculate_reward(action)
        return self._state(self.index), reward, self.index + 1 == self.number_periods


class Net(nn.Module):
    def __init__(self, input_shape: int, action_shape: int, hidden_size: int):
        """
        Implement neural network for actor-critic architecture
        Orthogonal initialization is applied, as useful for actor-critic methods
        :param input_shape: input dimension (length of state)
        :param action_shape: output dimension (length of action or number of selected tickers/tables)
        :param hidden_size: dimension of hidden layers
        """
        super().__init__()

        self.body: nn.Module = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.action_head: nn.Module = nn.Linear(hidden_size, action_shape)
        self.q_head: nn.Module = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.body[0].weight, 2 ** 0.5)
        nn.init.orthogonal_(self.body[2].weight, 2 ** 0.5)
        nn.init.orthogonal_(self.body[4].weight, 2 ** 0.5)

        nn.init.orthogonal_(self.action_head.weight, 2 ** 0.5)
        nn.init.orthogonal_(self.q_head.weight, 2 ** 0.5)

    def forward(self, s: torch.Tensor):
        """
        Implement forward pass of neural network
        :param s: tensor representing given state
        :return:
            1. tensor of action logits
            2. tensor from one element, as Q-function estimation
        """
        out: torch.Tensor = self.body(s)
        actions: torch.Tensor = self.action_head(out)
        q_val: torch.Tensor = self.q_head(out)
        return actions, q_val


class A2CAgent(BaseAgent):
    def __init__(self, model: nn.Module):
        """
        Save given network in A2C agent
        :param model: neural network for actor-critic architecture
        """
        super().__init__()
        self.model: nn.Module = model

    def get_action(self, s: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        """
        Implement computing actions and values
        Here actions are sampled for computing agent loss, which can not be use with all probabilities
        :param s: Given state for getting action
        :return: dict of
            logits    - agent output of actor-critic network
            probs     - tensor with softmax probabilities of logits
            log_probs - log of probs computed by log_softmax
            actions   - batch of sampled actions in range(0, num tickers)
            q_val     - batch of Q-function estimation
        """
        actions_logits, q_val = self.model(s)
        probs: torch.Tensor = F.softmax(actions_logits, dim=-1)
        log_probs: torch.Tensor = F.log_softmax(actions_logits, dim=-1)
        actions: torch.Tensor = torch.multinomial(probs, num_samples=1).detach()
        return {'actions': actions, 'logits': actions_logits, 'values': q_val, 'log_probs': log_probs, 'probs': probs}


class A2C(BaseAlgo):
    def __init__(self,
                 agent: BaseAgent,
                 optimizer: torch.optim.Optimizer,
                 env: BaseEnv,
                 value_loss_coef: float = 0.25,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        """
        Construct A2C algorithm instance
        :param agent: Used agent
        :param optimizer: Used optimizer from `torch.optim`
        :param env: Used environment
        :param value_loss_coef: Positive coefficient for value loss
        :param entropy_coef: Positive coefficient for entropy part of loss
        :param max_grad_norm: Limit of grad norm for pytorch clipping
        """
        super().__init__(env)
        assert value_loss_coef >= 0, f''
        assert entropy_coef >= 0, f'Entropy coefficient should be non negative'
        assert max_grad_norm > 0, f'Limit for grad norm must be positive'

        self.agent: BaseAgent = agent
        self.optimizer: torch.optim.Optimizer = optimizer
        self.env: BaseEnv = env
        self.value_loss_coef: float = value_loss_coef
        self.entropy_coef: float = entropy_coef
        self.max_grad_norm: float = max_grad_norm

    def agent_loss(self, trajectory: tp.Dict[str, tp.Any]):
        """
        Calculate loss for agent, as mean_i [log probs[actions_i] * Q[i]] where i in range(trajectory length)
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: tensor from one element with agent loss
        """
        adv = (trajectory['value_targets'] - trajectory['values']).detach()
        log_probs_action = trajectory['log_probs'][np.arange(len(trajectory['actions'])), trajectory['actions']]
        loss = - (adv * log_probs_action).mean()
        self.env.add_summary_scalar("agent_loss", loss.item())
        return loss

    def value_loss(self, trajectory: tp.Dict[str, tp.Any]):
        """
        Calculate loss fot critic, as MSE(Q_{predicted}, Q_{target})
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: tensor from one element with value loss
        """
        adv = trajectory['value_targets'].detach() - trajectory['values']
        loss = (adv**2).mean()
        self.env.add_summary_scalar("value_loss", loss.item())
        return loss

    def loss(self, trajectory: tp.Dict[str, tp.Any]):
        """
        Calculate total loss, as weighted sum of agent loss, value loss and minus entropy
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: tensor from one element with total loss
        """
        # computing entropy
        entropy = - torch.mean(torch.sum(trajectory['probs'] * trajectory['log_probs'], 1))
        # subtracting from others losses with positive coefficient
        loss = self.agent_loss(trajectory) + self.value_loss(trajectory) * self.value_loss_coef - \
            entropy * self.entropy_coef
        self.env.add_summary_scalar("loss", loss.item())
        return loss

    def step(self, trajectory):
        """
        Implement step for A2C algorithm
            1) Loss computed for whole trajectory
            2) Gradient step is made
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        self.optimizer.zero_grad()
        self.env.add_summary_scalar("rewards", np.mean(trajectory['rewards']))
        loss = self.loss(trajectory)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
