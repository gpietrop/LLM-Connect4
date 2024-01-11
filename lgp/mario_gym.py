from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import RenderFrame, ObsType, ActType
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway


class MarioEnv(gymnasium.Env):

    def __init__(self, gateway, java_mario_env, observation_space_limit: int = np.inf):
        self.gateway = gateway
        self.java_mario_env = java_mario_env
        self.action_size = self.java_mario_env.getActionSpace()
        original_observation_space = self.java_mario_env.getObservationSpace()
        self.grid_side = min(observation_space_limit, int(np.sqrt(original_observation_space)))
        self.observation_size = self.grid_side ** 2

    @staticmethod
    def observation_to_ascii(observation: np.ndarray) -> str:
        side = int(np.sqrt(len(observation)))
        observation = observation.reshape(side, side).T
        center = len(observation) // 2
        observation[center][center] = 5.
        string_obs = np.array2string(observation, max_line_width=100)
        string_obs = string_obs.replace(" ", "").replace("[", "").replace("]", "")
        string_obs = string_obs.replace("5.", "m").replace("1.", "e").replace("2.", "x").replace("0.", "-")
        return string_obs

    def _process_observation(self, java_observation):
        array_observations = []
        for obs in java_observation:
            arr = np.zeros(len(obs))
            for i, o in enumerate(obs):
                arr[i] = o
            array_observations.append(arr)
        observation_array = np.array(array_observations)
        transposed_observation_array = observation_array.transpose()
        border = (len(transposed_observation_array) - self.grid_side) // 2
        final_observation = transposed_observation_array[border:border + self.grid_side, border:border + self.grid_side]
        flat_observation = final_observation.flatten()
        flat_observation[flat_observation == 100] = 2
        return flat_observation

    # TODO see how to call Java from here
    # subprocess.run(["ls", "-l"]) maybe
    @classmethod
    def make(cls, level: str = None, observation_space_limit: int = np.inf) -> MarioEnv:
        gateway = JavaGateway()
        java_mario_env = gateway.entry_point
        return cls(gateway, java_mario_env, observation_space_limit)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        java_action = ListConverter().convert(action, self.gateway._gateway_client)
        step_object = self.java_mario_env.step(java_action)
        return (self._process_observation(step_object.observation()),
                step_object.reward(), step_object.terminated(), step_object.truncated(), step_object.information())

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        reset_object = self.java_mario_env.reset()
        return self._process_observation(reset_object.observation()), reset_object.information()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
