import gym_super_mario_bros
import numpy as np
import cv2

from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from typing import Tuple, Dict, Any
Array = np.ndarray
Action = int


def resize_frame(frame: Array, width: int = 144, height: int = 144) -> Array:
    if frame is not None:
        frame = cv2.resize(frame, (width, height),
                           interpolation=cv2.INTER_LINEAR_EXACT)[None, :, :, :]
        return frame.astype(np.uint8)
    else:
        return np.zeros((1, width, height, 3), dtype=np.uint8)


class ResizeFrame(Wrapper):
    def __init__(self, env, width=144, height=144):
        super().__init__(env)
        self.env = env
        self.width = width
        self.height = height

    def step(self, action: Action) -> Tuple[Array, float, bool, Dict[str, Any]]:
        state, reward, done, info = self.env.step(action)
        resized_state = resize_frame(state, self.width, self.height)
        return resized_state, reward, done, info

    def reset(self) -> Array:
        return resize_frame(self.env.reset(), self.width, self.height)


class CustomReward(Wrapper):
    """
    Code borrowed from https://github.com/uvipen/Super-mario-bros-PPO-pytorch
    To make 4-4, 7-4 easier
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

        env.reset()
        _, _, _, info = self.env.step(0)
        self.world = info['world']
        self.stage = info['stage']
        env.reset()

        self.curr_score = 0
        self.current_x = 40


    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self) -> Array:
        self.curr_score = 0
        self.current_x = 40
        return self.env.reset()


if __name__ == "__main__":
    import sys
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    frame = env.reset()
    print(frame.shape, sys.getsizeof(frame), frame.nbytes)
    # 144x144 is found enough for descriping details of the game
    new_frame = resize_frame(frame, width=144, height=144)
    print(new_frame.shape, sys.getsizeof(new_frame), new_frame.nbytes)
    import matplotlib.pyplot as plt

    # plt.imshow(new_frame[0,:,:,0])
    # plt.show()

    env = ResizeFrame(env, width=144, height=144)
    env = CustomReward(env)

    frame = env.reset()
    print(frame.shape, sys.getsizeof(frame), frame.nbytes)
    frame, done, reward, info = env.step(0)
    print(frame.shape, sys.getsizeof(frame), frame.nbytes)
    # print(info)
    # print(env.metadata)
