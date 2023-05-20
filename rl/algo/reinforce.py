import random
import numpy as np
import pandas as pd
import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F


from base import BaseEnv, BaseAgent, BaseAlgo


class ReinforceEnv(BaseEnv):
    """Actions for short/hold/long position respectively"""
    action_list = [-1, 0, 1]

    def __init__(self, table: pd.DataFrame, max_index: tp.Optional[int] = None):
        """
        Initialize Reinforce Environment
        :param table: DataFrame with features in rows
        :param max_index: Max possible index in DataFrame, where trajectory can be started + 1
        """
        super().__init__()
        self.num_columns: int = table.select_dtypes([np.number]).columns
        self.max_index: int = max_index or len(table)
        self.table: pd.DataFrame = table
        self.index: tp.Optional[int] = None

    def _state(self, index: int) -> torch.Tensor:
        """
        Extract state associated with index
        State features are values from numerical columns of table
        :param index: index of state in table
        :return: tensor merged from numerical features
        """
        return torch.tensor(np.array(self.table.iloc[index][self.num_columns]), dtype=torch.float32)

    def _std(self, index: int) -> float:
        """
        Extract standard deviation of `close` column at index
        :param index: Index of row in table to get std
        :return: float representing standard deviation at pos index
        """
        return self.table['vol'].iloc[index]

    def _calculate_reward(self, action: int, index: int) -> float:
        """
        Calculate [-1, 0, 1] * return_{index} with respect to made action (sell/hold/buy)
        After that result is divided by standard deviation
        :param action: Made action, vector of probabilities with sum equal to 1
        :param index: Index in table where to calculate reward for this action
        :return: float representing achieved reward
        """
        return self.action_list[action] * self.table.iloc[index]['close'] / self._std(index)

    def reset(self) -> torch.Tensor:
        """
        Reset index in range[0, max_index)
        :return: state associated with new index
        """
        self.index = random.randint(0, self.max_index)
        return self._state(self.index)

    def get_revenue(self, action: int) -> float:
        """
        Calculate [-1, 0, 1] * return_{index} with respect to made action (sell/hold/buy)
        Here returns are located at position `self.index`
        :param action: Made action in range(0, 3)
        :return: float representing achieved reward
        """
        return self.action_list[action] * self.table.iloc[self.index + 1]['close']

    def step(self, action: int) -> tp.Tuple[torch.Tensor, float, bool]:
        """
        Implementation of step by incrementing of `self.index`
        :param action: Made action in range(0, 3)
        :return:
            state  - represented by tensor
            reward - float reward for state with respect to action
            done   - was trajectory ended here or not
        """
        assert self.index is not None, f'Reset environment firstly'
        self.index += 1
        next_state = self._state(self.index)
        reward = self._calculate_reward(action, self.index)
        return next_state, reward, self.index + 1 == len(self.table)


class ReinforceAgent(BaseAgent):
    def __init__(self, state_length: int, action_length: int, hidden_size: int = 64):
        """
        Construct model by given params
        :param state_length: Dimension of state, as input of model
        :param action_length: Dimension of action, as output of model
        :param hidden_size: Dimension of hidden layers
        """
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(state_length, hidden_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_size, action_length, dtype=torch.float32)
        )

    def get_action(self, s: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        """
        Implement computing action and softmax probabilities
        :param s: Given state for getting action
        :return: dict of
            probs     - tensor with softmax probabilities of logits
            log_probs - log of probs computed by log_softmax
            actions   - batch of sampled actions in range(0, num_actions)
        """
        logits = self.model(s)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).detach()
        return {"actions": actions, "probs": probs, "log_probs": log_probs}


class Reinforce(BaseAlgo):
    def __init__(self,
                 agent: BaseAgent,
                 optimizer: torch.optim.optimizer,
                 env: BaseEnv,
                 entropy_coef: float = 0.1,
                 max_grad_norm: float = 0.5):
        """
        Construct REINFORCE algorithm instance
        :param agent: Used agent
        :param optimizer: Used optimizer from `torch.optim`
        :param env: Used environment
        :param entropy_coef: Positive coefficient for entropy part of loss
        :param max_grad_norm: Limit of grad norm for pytorch clipping
        """
        super().__init__(env)
        assert entropy_coef >= 0, f'Entropy coefficient should be non negative'
        assert max_grad_norm > 0, f'Limit for grad norm must be positive'

        self.agent: BaseAgent = agent
        self.optimizer: torch.optim.optimizer = optimizer
        self.entropy_coef: float = entropy_coef
        self.max_grad_norm: float = max_grad_norm

    def agent_loss(self, trajectory: tp.Dict[str, tp.Any]) -> torch.Tensor:
        """
        Calculate loss for agent, as mean_i [log probs[actions_i] * Q[i]] where i in range(trajectory length)
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: tensor from one element with agent loss
        """
        log_probs_action = trajectory['log_probs'][np.arange(len(trajectory['actions'])), trajectory['actions']]
        loss = - (trajectory['value_targets'] * log_probs_action).mean()
        self.env.add_summary_scalar("agent_loss", loss.item())
        return loss

    def loss(self, trajectory: tp.Dict[str, tp.Any]) -> torch.Tensor:
        """
        Calculate total loss, as weighted sum of agent loss and minus entropy
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: tensor from one element with total loss
        """
        # computing entropy
        entropy = - torch.mean(torch.sum(trajectory['probs'] * trajectory['log_probs'], 1))
        # subtracting from agent loss with positive coefficient
        loss = self.agent_loss(trajectory) - self.entropy_coef * entropy
        self.env.add_summary_scalar("loss", loss.item())
        return loss

    def step(self, trajectory: tp.Dict[str, tp.Any]):
        """
        Implement step for REINFORCE algorithm
            1) Loss computed for whole trajectory
            2) Gradient step is made
        :param trajectory: dict with actions, states and other info from sampled trajectory
        :return: None
        """
        self.env.add_summary_scalar("rewards", np.mean(trajectory["rewards"]))
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
