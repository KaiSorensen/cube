import numpy as np
import gym
from gym import spaces
from cube import Cube, getAllMoves, getInverseMove

class RubiksCubeEnv(gym.Env):
    def __init__(self, max_steps=250):
        super(RubiksCubeEnv, self).__init__()
        self.cube = Cube()
        self.action_space = spaces.Discrete(12)  
        # Each tile is an integer from 0..5. We'll store them in a 54-length array
        self.observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.int32)
        self.max_steps = max_steps
        self.current_step = 0



    def reset(self):
        self.cube.reset()
        # Optionally randomize the cube so the agent learns from non-trivial states
        self.cube.randomize()
        self.current_step = 0
        return np.array(self.cube.getStateArray(), dtype=np.int32)

    def step(self, action):
        # Action is an integer 0..11 (mapping to moves 1..12 in your code)
        move = action + 1  
        self.cube.move(move)

        self.current_step += 1

        # Check if the cube is solved
        done = self.cube.isSolved() or (self.current_step >= self.max_steps)

        # Reward scheme (example):
        # +100 if solved, otherwise partial reward = (# tiles correct / 54)
        if self.cube.isSolved():
            print("\033[92m\n\nSolved in", self.current_step, "steps\n\n\033[0m")
            reward = 150.0
        else:
            correctness = np.sum(
                np.array(self.cube.getStateArray()) == self.cube.getSolvedState()
            )
            percentageRight = correctness / 54.0
            if (done):
                print("\033[31mCorrectness:", percentageRight, "\033[0m")
            reward = correctness - 1 # it should never feel good about not solving it

        # (Optional) Penalty if the move is the inverse of previous
        if self.cube.reversedMove:
            reward -= 2.0

        next_state = np.array(self.cube.getStateArray(), dtype=np.int32)
        info = {}
        return next_state, reward, done, info

    def render(self, mode='human'):
        # If you want text or tkinter rendering
        self.cube.printNicely()
