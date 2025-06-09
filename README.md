
# 🎮 Reinforcement Learning Escape Room

An interactive multi-room GridWorld simulation showcasing Reinforcement Learning algorithms across four unique challenges. Each room represents a different RL paradigm, combined with puzzle-inspired mechanics and visuals from classic games and movies.

---

## 🚀 Project Overview

The **RL Escape Room** is a modular learning environment where an agent progresses through four rooms, each designed with different RL algorithms and increasingly complex tasks:

| Room | Algorithm | Theme | Special Feature |
| ---- | --------- | ----- | ---------------- |
| 1 | **Dynamic Programming (Value Iteration)** | Harry Potter | Snitch collection with state masks |
| 2 | **SARSA (On-Policy Learning)** | WALL-E | Random charging cell placement |
| 3 | **Q-Learning (Off-Policy Learning)** | Squid Game | Sequential shape collection (circle → square → triangle) |
| 4 | **Deep Q-Network (DQN)** | Rush Hour (Driving Cars) | Moving obstacles with path planning |

The entire simulation is built with Python using `pygame`, `gymnasium`, and `PyTorch`.

---

## 🧠 Key Concepts Demonstrated

- GridWorld-based RL environment
- Dynamic state-dependent reward systems
- Complex environment features: slippery tiles, prison tiles, dynamic obstacles
- Visualization of training, policies, and Q-values
- Parameter tuning via Tkinter-based GUI before each training
- Fully interactive gameplay after training

---

## 🏗️ Project Structure

```
.
├── main.py               # Main game loop and room manager
├── requirements.txt      # Dependencies
├── rooms/                # Each RL room logic
│   ├── room1_dp.py
│   ├── room2_sarsa.py
│   ├── room3_qlearning.py
│   └── room4_dqn.py
├── environment/
│   ├── grid_world.py     # Base GridWorld logic
│   ├── renderer.py       # Renderer for rooms 1,2,4
│   └── renderer3_math.py # Special renderer for math-based shapes (room 3)
├── utils/
│   └── parameter_dialogs.py  # Tkinter parameter input dialogs
└── assets/images/        # All visual assets per room
```

---

## 📦 Installation

1️⃣ Clone this repository:

```bash
git clone <your-repo-url>
cd rl-escape-room
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

✅ Required packages:

- numpy
- pygame
- matplotlib
- torch
- gymnasium

See `requirements.txt` for exact versions.

---

## 🎮 Running the Simulation

Simply execute:

```bash
python main.py
```

You'll be guided room-by-room.

✅ Before entering each room, you’ll be prompted to enter RL hyperparameters via a user-friendly Tkinter GUI.

✅ Once training completes, you control the agent interactively to test the learned policy.

---

## 🏰 Room Descriptions

### Room 1: **Dynamic Programming (Harry Potter - Snitch Puzzle)**

- Collect all snitches using Value Iteration.
- Full state space includes a bitmask for snitch collection state.
- Optimal policy is precomputed (model-based).
- Obstacles, slippery tiles and prison tiles included.

### Room 2: **SARSA (WALL-E Charging Room)**

- Agent must collect random charging cells before reaching goal.
- Charging cells are re-randomized every reset.
- Fully on-policy learning using SARSA updates.
- Adaptive exploration during training.

### Room 3: **Q-Learning (Squid Game - Shape Puzzle)**

- Sequential object collection task (circle → square → triangle).
- Q-table includes extended state (current stage).
- Off-policy learning enables faster convergence.
- Shape-based penalties for wrong order collection.

### Room 4: **Deep Q-Network (Rush Hour Driving Cars Puzzle)**

- Moving car obstacles inspired by Rush Hour board game.
- Neural network approximation of Q-values using PyTorch.
- Replay buffer, target networks and soft updates (DQN architecture).
- Smooth difficulty increase via dynamic cars path.

---

## 🎯 Sample Screenshots

| Harry Potter (Room 1) | WALL-E (Room 2) | Squid Game (Room 3) | Rush Hour (Room 4) |
| ---- | ---- | ---- | ---- |
| ![](assets/images/harry_potter.png) | ![](assets/images/battery.jpg) | ![](assets/images/shapes.png) | ![](assets/images/rush_hour.png) |

*(Replace with your actual screenshots based on the images you provided)*

---

## 🧪 Training Visualization

- 📈 Training curves (episode reward)
- 📊 Policy heatmaps
- 📉 Epsilon decay graphs
- 🧭 Policy arrows visualization

---

## 🛠 Features Summary

- Fully interactive environment
- Tkinter-based parameter configuration before each training
- Dynamic difficulty via randomized layouts
- Smooth transitions between rooms
- End-of-room popups and full game completion message
- GPU compatible (for DQN training)

---

## ⚠️ Known Limitations

- All rooms run sequentially; no save/load checkpoints.
- `pygame` rendering loop requires display access.
- DQN training speed may vary based on hardware.

---

## 💡 Educational Goals

This project is designed to help students and practitioners:

- Understand the difference between model-based and model-free RL
- Gain intuition for on-policy vs. off-policy methods
- Implement full RL pipelines (train → evaluate → visualize → deploy)
- Tackle increasingly complex reward structures
- Combine game design with AI training flows

---

## 🙏 Credits

- Inspired by famous games and movies:  
  *Harry Potter, WALL-E, Squid Game, Rush Hour*

- Developed as part of a reinforcement learning educational showcase.

---

## 📝 License

Open for educational and non-commercial use.  
Feel free to fork, adapt and extend!
