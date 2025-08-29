import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import pybullet as p

class OT2Env(gym.Env):
    def __init__(self, render=True, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        
        # creating the simulation environment
        self.sim = Simulation(num_agents=1)

        # defining action and observation space
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        goal_low = np.array([-0.1904, -0.1712, -0.1205], dtype=np.float32)
        goal_high = np.array([0.255, 0.2203, 0.2906], dtype=np.float32)

        # initializing the goal space within defined boundaries
        self.goal_space = spaces.Box(low=goal_low, high=goal_high, shape=(3,), dtype=np.float32)
        
        # keeping track of steps and previous distance
        self.steps = 0
        self.prev_distance = None

    def seed(self, seed=None):
        """Ensure reproducibility."""
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, seed=None):
        """Reset the environment and get new goal."""
        self.seed(seed)
        self.goal_position = self.goal_space.sample()
        goal_position = np.array(self.goal_position, dtype=np.float32)

        # resetting simulation and get initial pipette position
        observation = self.sim.reset(num_agents=1)
        robot_id = list(observation.keys())[0]
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id.replace("robotId_", ""))), dtype=np.float32)

        # combinging pipette position and goal position
        observation = np.concatenate([pipette_position, goal_position])

        self.steps = 0
        self.prev_distance = np.linalg.norm(pipette_position - goal_position)
        return observation, {}

    def step(self, action):
        """Execute one time step within the environment."""
        # scaling first, then append drop action
        scaled_action = np.clip(action * np.array([0.5, 0.5, 0.3, 1.0]), -1.0, 1.0)
        
        # and now append 0 for the drop action
        scaled_action = np.append(scaled_action, 0)

        observation = self.sim.run([scaled_action])

        # extractubg the pipette position
        robot_id = list(observation.keys())[0]
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id.replace("robotId_", ""))), dtype=np.float32)
        goal_position = np.array(self.goal_position, dtype=np.float32)

        observation = np.concatenate([pipette_position, goal_position])

        # calculate reward
        reward = self.calculate_reward(pipette_position, goal_position)

        terminated = np.linalg.norm(pipette_position - goal_position) < self.get_threshold()
        truncated = self.steps >= self.max_steps or (self.steps > 50 and abs(self.prev_distance - np.linalg.norm(pipette_position - goal_position)) < 0.001)
        
        # it logs every 10 steps
        if self.steps % 10 == 0:
            print(f"Step {self.steps}: Pos={pipette_position}, Goal={goal_position}, Reward={reward:.2f}, Distance={np.linalg.norm(pipette_position - goal_position):.4f}")

        self.steps += 1
        return observation, reward, terminated, truncated, {}

    def calculate_reward(self, pipette_position, goal_position):
        """Dynamically scaling reward function."""
        current_distance = np.linalg.norm(pipette_position - goal_position)
        base_penalty = -10 * current_distance  # higher penalty for being far
        progress_reward = 5 * (self.prev_distance - current_distance) if self.prev_distance is not None else 0
        goal_bonus = 100 if current_distance < self.get_threshold() else 0
        self.prev_distance = current_distance
        return float(base_penalty + progress_reward + goal_bonus)

    def get_threshold(self):
        """Adaptive threshold based on workspace size."""
        return 0.002 * np.mean([0.255 - (-0.1904), 0.2203 - (-0.1712), 0.2909 - 0.1205])

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
