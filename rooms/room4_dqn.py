
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNRoom(GridWorldEnv):
    def __init__(self, size: int = 10, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0,
                epsilon_decay: float = 0.995, min_epsilon: float = 0.01, batch_size: int = 64,
                tau: float = 0.001, hidden_size: int = 64):
        super().__init__(size=size)
        self._setup_room()
        self.state_size = 2
        self.action_size = 4
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.hidden_size)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer()
        self.min_experiences = 1000

        self.episode_rewards = []
        self.current_episode = 0
        self.steps_done = 0
        self.best_reward = float('-inf')
        self.successful_trajectories = []
        self.replay_started = False



        self.moving_cars = []
        self.initialize_moving_cars()

    def _setup_room(self):
        for x in [2, 3, 4, 5]:
            self.add_special_tile('obstacles', (x, 2))
            self.add_special_tile('obstacles', (x, 7))
        for y in [3, 4, 5]:
            self.add_special_tile('obstacles', (2, y))
            self.add_special_tile('obstacles', (7, y))
        for x in [1, 2, 3]:
            for y in [8, 9]:
                self.add_special_tile('slippery', (x, y))
        for x in [8, 9]:
            for y in [1, 2, 3]:
                self.add_special_tile('slippery', (x, y))
        self.add_special_tile('prison', (4, 4))
        self.add_special_tile('prison', (6, 6))
        self.add_special_tile('prison', (8, 8))
        self.add_special_tile('goal', (9, 9))

    def initialize_moving_cars(self):
        fixed_positions = [(1, 1), (1, 8)]
        self.moving_cars = [{"origin": pos, "position": pos} for pos in fixed_positions]

        invalid_positions = (
            self.special_tiles['obstacles'] |
            self.special_tiles['goal'] |
            self.special_tiles['prison'] |
            self.special_tiles['slippery'] |
            set(fixed_positions)
        )

        valid_positions = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in invalid_positions]
        np.random.shuffle(valid_positions)

        for i in range(3):
            pos = valid_positions[i]
            self.moving_cars.append({"origin": pos, "position": pos})

    def update_moving_cars(self):
        blocked_tiles = (self.special_tiles['obstacles'] |
                        self.special_tiles['goal'] |
                        self.special_tiles['prison'] |
                        self.special_tiles['slippery'])
        occupied_positions = set(car['position'] for car in self.moving_cars)

        for car in self.moving_cars:
            origin = car['origin']
            if car['position'] != origin:
                occupied_positions.discard(car['position'])
                car['position'] = origin
                occupied_positions.add(origin)
                continue

            possible_moves = [(1,0), (-1,0), (0,1), (0,-1)]
            random.shuffle(possible_moves)
            for dx, dy in possible_moves:
                new_pos = (origin[0] + dx, origin[1] + dy)
                if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
                    continue
                if new_pos in blocked_tiles or new_pos in occupied_positions:
                    continue
                occupied_positions.discard(car['position'])
                car['position'] = new_pos
                occupied_positions.add(new_pos)
                break

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.replay_started = False
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int):
        self.update_moving_cars()
        obs, reward, terminated, info = super().step(action)

        # אם התנגשות עם רכב
        if tuple(obs) in [car["position"] for car in self.moving_cars]:
            reward = -10.0
            terminated = True
            info['success'] = False
            return obs, reward, terminated, info

        # קרבה למטרה מוסיפה עונש קטן
        if self.special_tiles['goal']:
            goal = list(self.special_tiles['goal'])[0]
            distance = np.linalg.norm(np.array(self.agent_position) - np.array(goal))
            reward += -0.03 * distance

            # הצלחה רק אם באמת הגעת ל־goal
            if tuple(obs) in self.special_tiles['goal']:
                reward += 20
                terminated = True
                info['success'] = True
                print(f"🎯 Agent reached goal at position {tuple(obs)}!")
            else:
                info['success'] = False

        return obs, reward, terminated, info
    
    def restore_initial_replay_state(self):
        if not self.replay_started and hasattr(self, "current_replay_trajectory"):
            # שחזור מיקום הסוכן
            start_state = self.current_replay_trajectory[0]["state"]
            self.agent_position = tuple(start_state)

            # שחזור מיקומי רכבים
            start_cars = self.current_replay_trajectory[0]["cars"]
            for i, pos in enumerate(start_cars):
                if i < len(self.moving_cars):
                    self.moving_cars[i]["position"] = pos




    def normalize_state(self, state):
        return np.array(state) / (self.size - 1)

    def get_state_tensor(self, state):
        normalized = self.normalize_state(state)
        return torch.FloatTensor(normalized).unsqueeze(0)

    def get_action(self, state, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            state_tensor = self.get_state_tensor(state)
            action_values = self.qnetwork_local(state_tensor)
            return np.argmax(action_values.numpy())

    def train_step(self):
        if len(self.memory) < self.min_experiences:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train_episode(self) -> float:
        obs, _ = self.reset()
        state = tuple(obs)
        total_reward = 0
        done = False
        trajectory = []

        while not done:
            action = self.get_action(state, training=True)
            cars_state = [car["position"] for car in self.moving_cars]
            next_obs, reward, terminated, info = self.step(action)
            next_state = tuple(next_obs)
            done = terminated
            total_reward += reward

            # שמירת ניסיון
            self.memory.add(
                self.normalize_state(state), 
                action, 
                reward, 
                self.normalize_state(next_state), 
                done
            )
            trajectory.append({
                "state": state,
                "action": action,
                "cars": cars_state
            })

            self.train_step()
            state = next_state
            self.steps_done += 1

        # בדיקת הצלחה ושמירת מסלול מוצלח
        moved_enough = all(trajectory[i]["state"] != trajectory[i + 1]["state"] for i in range(len(trajectory) - 1))
        is_final_step_on_goal = tuple(state) in self.special_tiles['goal']
        if info.get("success", False) and len(trajectory) >= 5 and moved_enough and is_final_step_on_goal:
            self.successful_trajectories.append((total_reward, trajectory))
            print(f"✅ Successful trajectory saved with reward: {total_reward}")


        self.episode_rewards.append(total_reward)
        self.current_episode += 1
        return total_reward


    def train(self, num_episodes: int = 2000):
        for _ in range(num_episodes):
            self.train_episode()
        
        if self.successful_trajectories:
            best_traj = max(self.successful_trajectories, key=lambda x: x[0])[1]
            print(f"✅ Example trajectory saved with {len(best_traj)} steps")
        else:
            print("❌ No successful trajectory found")



    def get_action_from_successful_trajectory(self):
        if not self.successful_trajectories:
            return None

        if not self.replay_started:
            # רק הגדר שהתחיל הREPLAY, הכל השאר כבר נעשה ב-run()
            self.replay_started = True
            self.steps = 0
            return None

        if self.steps < len(self.current_replay_trajectory):
            step_data = self.current_replay_trajectory[self.steps]
            for i, pos in enumerate(step_data["cars"]):
                if i < len(self.moving_cars):
                    self.moving_cars[i]["position"] = pos
            return step_data["action"]

        return None






    def plot_training_progress(self):
        plt.figure(figsize=(10, 5))
        rewards = np.array(self.episode_rewards)
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(smoothed)
        plt.title('Training Progress (Smoothed)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()

        plt.figure(figsize=(10, 5))
        eps = [max(self.epsilon_min, 1.0 * (self.epsilon_decay ** i))
            for i in range(len(self.episode_rewards))]
        plt.plot(eps)
        plt.title('Exploration Rate Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.show()

    def plot_q_values(self):
        plt.figure(figsize=(10, 10))
        value_matrix = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.special_tiles['obstacles']:
                    value_matrix[x, y] = float('nan')
                else:
                    state_tensor = self.get_state_tensor((x, y))
                    with torch.no_grad():
                        q_values = self.qnetwork_local(state_tensor)
                        value_matrix[x, y] = torch.max(q_values).item()
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
                state_tensor = self.get_state_tensor((x, y))
                with torch.no_grad():
                    action = torch.argmax(self.qnetwork_local(state_tensor)).item()

                if action == 0: U[x, y] = -1
                elif action == 1: U[x, y] = 1
                elif action == 2: V[x, y] = 1
                else: V[x, y] = -1

        plt.quiver(X, Y, U.T, V.T)
        plt.title('Learned Policy')
        plt.show()