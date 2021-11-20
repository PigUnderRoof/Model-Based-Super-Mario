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

    frame = env.reset()
    print(frame.shape, sys.getsizeof(frame), frame.nbytes)
    frame = env.step(0)[0]
    print(frame.shape, sys.getsizeof(frame), frame.nbytes)
