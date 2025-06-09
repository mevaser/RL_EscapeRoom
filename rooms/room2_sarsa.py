import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv

class SARSARoom(GridWorldEnv):
    """
    Room 2: SARSA (On-policy) implementation - WALL-E Charging Room
    """
    def __init__(self, size: int = 10, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        super().__init__(size=size)

        self.goal_position = (9, 9)
        self._setup_room()

        self.fixed_charging_cells = {(3, 3), (6, 6)}
        self.charging_cells = set()
        self._place_charging_cells()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {(x, y): np.zeros(4) for x in range(size) for y in range(size)}
        self.episode_rewards = []
        self.current_episode = 0

    
    def _setup_room(self):
        for x in [2, 3, 4, 5]:
            self.add_special_tile('obstacles', (x, 4))
        for y in [1, 2, 3]:
            self.add_special_tile('obstacles', (5, y))
        for x in [7, 8]:
            for y in [5, 6]:
                self.add_special_tile('slippery', (x, y))
        self.add_special_tile('prison', (3, 7))
        self.add_special_tile('prison', (7, 2))
        self.add_special_tile('goal', self.goal_position)

    def _place_charging_cells(self):
        self.charging_cells = set(self.fixed_charging_cells)
        while len(self.charging_cells) < 5:
            cell = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if (cell not in self.special_tiles['obstacles'] and
                cell not in self.special_tiles['goal'] and
                cell not in self.special_tiles['prison'] and
                cell not in self.charging_cells):
                self.charging_cells.add(cell)

    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.choice(4)
        return np.argmax(self.Q[state])
    
    def train_episode(self) -> float:
        obs, _ = self.reset()
        state = tuple(obs)
        total_reward = 0
        
        action = self.get_action(state, training=True)
        done = False

        while not done:
            next_obs, reward, terminated, info = self.step(action)
            next_state = tuple(next_obs)
            done = terminated
            total_reward += reward

            next_action = self.get_action(next_state, training=True)

            if not done:
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
                )
            else:
                self.Q[state][action] += self.alpha * (reward - self.Q[state][action])

            state = next_state
            action = next_action

        self.episode_rewards.append(total_reward)
        self.current_episode += 1

        # prevent policy oscillation on goal
        self.Q[self.goal_position] = np.zeros(4)

        return total_reward
    
    def train(self, num_episodes: int = 3000):
        for _ in range(num_episodes):
            self.train_episode()

    def step(self, action: int):
        obs, reward, terminated, info = super().step(action)
        position = tuple(obs)

        if position in self.charging_cells:
            reward += 5  
            self.charging_cells.remove(position)

        # ⚠ no extra goal reward or termination here — super() already handles it.

        return obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._place_charging_cells()
        return obs, info

    def plot_training_progress(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
    
    def plot_q_values(self):
        plt.figure(figsize=(10, 10))
        value_matrix = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.special_tiles['obstacles']:
                    value_matrix[x, y] = float('nan')
                else:
                    value_matrix[x, y] = np.max(self.Q[(x, y)])
        plt.imshow(value_matrix.T, origin='lower')
        plt.colorbar(label='Max Q-Value')
        plt.title('Q-Value Function')
        plt.show()

    def plot_policy(self):
        plt.figure(figsize=(10, 10))
        X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        U, V = np.zeros((self.size, self.size)), np.zeros((self.size, self.size))

        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.special_tiles['obstacles'] or (x, y) in self.special_tiles['goal']:
                    continue
                action = np.argmax(self.Q[(x, y)])
                if action == 0: U[x, y] = -1
                elif action == 1: U[x, y] = 1
                elif action == 2: V[x, y] = 1
                else: V[x, y] = -1

        plt.quiver(X, Y, U.T, V.T)
        plt.title('Learned Policy')
        plt.show()
