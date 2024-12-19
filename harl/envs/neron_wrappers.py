"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
import gym
import gymnasium
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy
from typing import Any, Mapping, Sequence, Tuple, Union
from harl.models.base.neron import NeronBase

class NeronWrapper(object):

    def __init__(self, env: Any, neron_args: Any) -> None:
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        self.num_nerons_per_agent = neron_args["num_nerons_per_agent"]


    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        pass

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        pass

    def state(self) -> torch.Tensor:
        pass

    def render(self, *args, **kwargs) -> Any:
        pass

    def close(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        pass

    @property
    def num_envs(self) -> int:
        pass

    @property
    def n_agents(self) -> int:
        return self._unwrapped.n_agents*self.num_nerons_per_agent

    @property
    def max_num_agents(self) -> int:
        pass

    @property
    def agents(self) -> Sequence[str]:
        pass

    @property
    def possible_agents(self) -> Sequence[str]:
        pass

    @property
    def observation_space(self) -> Mapping[int, gym.Space]:
        pass

    @property
    def action_space(self) -> Mapping[int, gym.Space]:
        pass

    @property
    def share_observation_space(self) -> Mapping[int, gym.Space]:
        pass

    