import gym_super_mario_bros
import numpy as np
import cv2

Array = np.ndarray


def resize_frame(frame: Array, width: int = 144, height: int = 144) -> Array:
    if frame is not None:
        frame = cv2.resize(frame, (width, height),
                           interpolation=cv2.INTER_LINEAR_EXACT)[None, :, :, :]
        return frame.astype(np.uint8)
    else:
        return np.zeros((1, width, height, 3), dtype=np.uint8)


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
