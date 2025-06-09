# 🤖 Reinforcement Learning Escape Room

An interactive GridWorld-based escape room simulation featuring four rooms, each implementing a different Reinforcement Learning algorithm.

## 🌟 Project Overview

This project implements a unique escape room experience where each room represents a different Reinforcement Learning challenge:

1. **Room 1**: Dynamic Programming (Policy/Value Iteration)
2. **Room 2**: SARSA (On-policy Learning)
3. **Room 3**: Q-Learning (Off-policy Learning)
4. **Room 4**: Deep Q-Network (DQN)

Each room is a 10x10 grid environment with various features like slippery tiles, prison cells, and obstacles.

## 🚀 Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the game:
   ```bash
   python main.py
   ```

## 🎮 Game Features

- Interactive 10x10 GridWorld environment
- Visual representation of agent actions
- Real-time display of Q-tables and rewards
- Customizable learning parameters (α, γ, ε)
- Progress tracking between rooms
- Training and episode management system

## 🏗️ Project Structure

- `main.py` - Main game entry point
- `environment/` - Core environment implementation
  - `grid_world.py` - Base GridWorld environment
  - `renderer.py` - Visual rendering utilities
- `rooms/` - Individual room implementations
  - `room1_dp.py` - Dynamic Programming room
  - `room2_sarsa.py` - SARSA room
  - `room3_qlearning.py` - Q-Learning room
  - `room4_dqn.py` - Deep Q-Network room
- `utils/` - Utility functions and helpers

## 🎯 Room Details

### Room 1 - Dynamic Programming
- Known model (transition + reward)
- Policy/Value Iteration implementation
- Features: Slippery tiles, prison cells

### Room 2 - SARSA
- Unknown model
- On-policy learning
- ε-greedy exploration
- Features: Slippery tiles, prison cells

### Room 3 - Q-Learning
- Unknown model
- Off-policy learning
- Optimal policy convergence
- Features: Slippery tiles, prison cells

### Room 4 - Deep Q-Network
- Neural network Q-value approximation
- Experience replay buffer
- Target network implementation
- Features: Slippery tiles, prison cells

## 🎛️ Parameters

Each algorithm's parameters can be adjusted:
- Learning rate (α)
- Discount factor (γ)
- Exploration rate (ε)

## 📊 Visualization

- Real-time display of agent's actions
- Q-table/policy visualization
- Cumulative reward plotting
- Training progress monitoring 