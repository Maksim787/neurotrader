from collections import defaultdict
import typing as tp

import base
import transform


class EnvRunner:
    def __init__(self,
                 env: base.BaseEnv,
                 agent: base.BaseAgent,
                 n_steps: int,
                 transformers: tp.Optional[tp.List[transform.Transformer]] = None):
        """
        Initialize runner with params
        :param env: Used environment
        :param agent: Used agent
        :param n_steps: Number of steps in trajectory
        :param transformers: Used transformers for trajectories
        """
        self.env: base.BaseEnv = env
        self.agent: base.BaseAgent = agent
        self.state: tp.Dict[str, tp.Any] = {"latest_observation": self.env.reset()}
        self.n_steps: int = n_steps
        self.transformers: tp.Optional[tp.List[transform.Transformer]] = transformers

    def reset(self) -> None:
        """
        Resets metrics and the latest state
        Used for restarting graphs plotting
        :return: None
        """
        self.env.metrics = defaultdict(list)
        self.state["latest_observation"] = self.env.reset()

    def get_next(self) -> tp.Dict[str, tp.Any]:
        """
        Runs specified number of steps to sample trajectory
        It can be discontinuous
        :return: Sampled trajectory
        """
        trajectory: tp.DefaultDict[str, tp.Any] = defaultdict(list, {"actions": []})
        observations, rewards, resets = [], [], []
        self.state["env_steps"] = self.n_steps

        for i in range(self.n_steps):
            observations.append(self.state["latest_observation"])

            # making action using the latest state
            action: tp.Dict[str, tp.Iterable] = self.agent.get_action(self.state["latest_observation"])
            if "actions" not in action:
                raise ValueError(f"Key `actions` was not found in {action.keys()}")

            for key, val in action.items():
                trajectory[key].append(val)

            # collecting next state, reward and info if trajectory is ended
            s, r, done = self.env.step(action["actions"])

            # updating the latest state, and metric collectors
            self.state["latest_observation"] = s
            rewards.append(r)
            resets.append(done)

            # if trajectory is ended, we need to reset it
            if done:
                self.state["env_steps"] = i + 1
                self.state["latest_observation"] = self.env.reset()

        # put results into trajectory info
        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets)
        trajectory["state"] = self.state

        # applying transformers
        if self.transformers is not None:
            for transformer in self.transformers:
                transformer(trajectory)

        return trajectory
