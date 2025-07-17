import os
import pygame
import numpy as np
import random
from utils.parameter_dialogs import (
    get_dp_params,
    get_sarsa_params,
    get_qlearning_params,
    get_dqn_params,
)
from environment.renderer import GridWorldRenderer
from environment.renderer3_math import GridWorldRenderer as MathRenderer
from rooms.room1_dp import DPRoom
from rooms.room2_sarsa import SARSARoom
from rooms.room3_qlearning import QLearningRoom
from rooms.room4_dqn import DQNRoom


def setup_room(self, room_num, action_mode):
    """Setup the selected room with the specified algorithm and mode (train/run)."""
    print(f"\nSetting up Room {room_num} in {action_mode} mode...")

    self.selected_room = room_num
    self.action_mode = action_mode

    if room_num == 1:
        print("Room 1 - Dynamic Programming (Quidditch)")
        gamma, theta = get_dp_params()
        self.room = DPRoom(gamma=gamma, theta=theta)
        self.renderer = GridWorldRenderer(
            window=self.window,
            window_size=self.window_size,
            background_path="assets/images/room1_background.jpg",
        )

        if action_mode == "train":
            print("Running value iteration...")
            self.room.value_iteration()
            self.room.plot_value_function()
            self.room.plot_policy()
            self.save_agent_state(room_num)
            print("Training complete! Press SPACE to see the agent move.")
        else:
            # 👇 Pass greedy=True to force epsilon=0 during run
            if not self.load_agent_state(room_num, greedy=True):
                self.show_not_trained_popup()
                return False
            print("Loaded trained agent. Press SPACE to see the agent move.")

    elif room_num == 2:
        print("Room 2 - SARSA (WALL-E Charging Room)")

        if action_mode == "train":
            alpha, gamma, epsilon = get_sarsa_params()
        else:
            # Dummy values – not used in run mode
            alpha = gamma = epsilon = 0

        self.room = SARSARoom(alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.renderer = GridWorldRenderer(
            window=self.window,
            window_size=self.window_size,
            background_path="assets/images/room2_background.jpg",
        )

        if action_mode == "train":
            self.training = True
            print("Training SARSA agent for 1000 episodes...")
            self.room.train(num_episodes=1000)
            self.room.plot_training_progress()
            self.room.plot_q_values()
            self.room.plot_policy()
            self.training = False
            self.save_agent_state(room_num)
            print("Training complete! Press SPACE to see the agent move.")
        else:
            if not self.load_agent_state(room_num):
                self.show_not_trained_popup()
                return False
            self.room.load_rewards_from_file()
            self.room.plot_training_progress()

            print("Loaded trained agent. Press SPACE to see the agent move.")

    elif room_num == 3:
        print("Room 3 - Q-Learning (Squid Game)")

        if action_mode == "train":
            alpha, gamma, epsilon, epsilon_decay, min_epsilon = get_qlearning_params()
        else:
            # Dummy values – won't be used in run mode
            alpha = gamma = epsilon = epsilon_decay = min_epsilon = 0

        self.room = QLearningRoom(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
        )
        self.renderer = MathRenderer(
            window=self.window,
            window_size=self.window_size,
            background_path="assets/images/room3_background.jpg",
        )

        if action_mode == "train":
            self.training = True
            print("Training Q-Learning agent for 2000 episodes...")
            self.room.train(num_episodes=2000)
            self.training = False
            self.save_agent_state(room_num)
            print("Training complete! Press SPACE to see the agent move.")
        else:
            if not self.load_agent_state(room_num):
                self.show_not_trained_popup()
                return False
            self.room.load_rewards_from_file()  # ⬅️ הוסף
            self.room.plot_training_progress()
            print("Loaded trained agent. Press SPACE to see the agent move.")

    elif room_num == 4:
        print("Room 4 - DQN (Rush Hour)")

        if action_mode == "train":
            (
                learning_rate,
                gamma,
                epsilon,
                epsilon_decay,
                min_epsilon,
                batch_size,
                tau,
                hidden_size,
            ) = get_dqn_params()
        else:
            # Use dummy but valid default values (same as in get_dqn_params or expected by model)
            learning_rate = 1e-3
            gamma = 0.99
            epsilon = 1.0
            epsilon_decay = 0.999
            min_epsilon = 0.05
            batch_size = 64
            tau = 0.005
            hidden_size = 64

        self.room = DQNRoom(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            batch_size=batch_size,
            tau=tau,
            hidden_size=hidden_size,
        )

        self.renderer = GridWorldRenderer(
            window=self.window,
            window_size=self.window_size,
            background_path="assets/images/room4_background.jpg",
        )

        if action_mode == "train":
            self.training = True
            print("Training DQN agent for 2000 episodes...")
            self.room.train(num_episodes=2000)
            self.room.plot_training_progress()
            self.room.plot_q_values()
            self.room.plot_policy()
            self.training = False
            self.save_agent_state(room_num)
            print("Training complete! Press SPACE to see the agent move.")
        else:
            if not self.load_agent_state(room_num):
                self.show_not_trained_popup()
                return False
            print("Loaded trained agent. Press SPACE to see the agent move.")
            self.room.run_single_episode_for_replay()

    self.room.reset()
    self.snitch_mask = 0
    return True


def reset_room_state(self):
    """Reset the room state to prepare for a new game."""
    self.room = None
    self.renderer = None
    self.training = False
    self.snitch_mask = 0
    self.selected_room = None
    self.action_mode = None


def capture_background_snapshot(self):
    """Capture a snapshot of the current room background."""
    if self.selected_room == 3:
        task_text = "Squid Game Room: Collect Circle, then Square, then Triangle"
        snapshot = self.renderer.render_with_shapes(
            agent_position=self.room.agent_position,
            special_tiles=self.room.special_tiles,
            shapes_positions=self.room.shapes_positions,
            collected_mask=self.room.collected_mask,
            shape_indices=self.room.shape_indices,
            current_task=task_text,
        ).copy()

    elif self.selected_room == 2:
        snapshot = self.renderer.render(
            self.room.agent_position,
            self.room.special_tiles,
            info={
                "Room": f"{self.selected_room}/4",
                "Steps": self.room.steps,
                "Training": self.training,
            },
            charging_cells=self.room.charging_cells,
        ).copy()
    elif self.selected_room == 4:
        snapshot = self.renderer.render(
            self.room.agent_position,
            self.room.special_tiles,
            info={
                "Room": f"{self.selected_room}/4",
                "Steps": self.room.steps,
                "Training": self.training,
            },
            moving_cars=self.room.moving_cars,
        ).copy()
    else:
        snapshot = self.renderer.render(
            self.room.agent_position,
            self.room.special_tiles,
            info={
                "Room": f"{self.selected_room}/4",
                "Steps": self.room.steps,
                "Training": self.training,
            },
        ).copy()
    return pygame.surfarray.make_surface(np.transpose(snapshot, (1, 0, 2)))


def run_game_loop(self):
    """Main game loop for playing the selected room"""
    running = True
    clock = pygame.time.Clock()
    summary = None  # Initialize summary variable
    self.exit_game_loop = False

    while running and not self.exit_game_loop:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check visualization button clicks
                viz_buttons = self.draw_visualization_buttons(mouse_pos)
                for button_type, button_rect in viz_buttons:
                    if button_rect.collidepoint(event.pos):
                        print(f"[DEBUG] Clicked button: {button_type}")
                        self.handle_visualization_click(button_type)
                        break

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    self.show_controls = not self.show_controls
                elif event.key == pygame.K_SPACE and not self.training:
                    if self.selected_room == 1:
                        x, y = self.room.agent_position
                        action = self.room.policy[x, y, self.snitch_mask]
                    elif self.selected_room == 3:
                        state = tuple(self.room.agent_position)
                        q_state = self.room.get_q_state(state)
                        action = self.room.get_action(state, training=False)
                        print(
                            f"[DEBUG] Room 3 - state: {state}, mask: {self.room.collected_mask}, q_state: {q_state}"
                        )
                        print(
                            f"[DEBUG] Q-values: {self.room.Q.get(q_state)} | Action selected: {action}"
                        )

                    elif self.selected_room == 4:
                        state = tuple(self.room.agent_position)
                        action = self.room.get_action(state, training=False)
                    else:
                        action = self.room.get_action(
                            self.room.agent_position, training=False
                        )

                    obs, reward, terminated, truncated, info = self.room.step(action)

                    if info.get("timeout", False):
                        print("\nEpisode ended due to step limit (timeout).")

                    if self.selected_room == 1:
                        pos = tuple(self.room.agent_position)
                        if (
                            hasattr(self.room, "snitch_indices")
                            and pos in self.room.snitch_indices
                        ):
                            idx = self.room.snitch_indices[pos]
                            if not (self.snitch_mask & (1 << idx)):
                                self.snitch_mask |= 1 << idx

                    if terminated and (
                        self.selected_room != 4 or info.get("success") is True
                    ):
                        print("\nGoal reached! 🎉")
                        self.show_room_completed_popup()
                        running = False
                        continue

                elif event.key == pygame.K_r:
                    print("\nResetting room...")
                    self.room.reset()

        if self.room and self.renderer:
            info_dict = {
                "Room": f"{self.selected_room}/4",
                "Steps": self.room.steps,
                "Training": self.training,
            }

            if self.selected_room == 1:
                info_dict.update(
                    {
                        "snitch_collected": self.room.collected_snitch,
                        "snitch_total": self.room.total_snitch,
                    }
                )

            if self.selected_room == 2:
                info_dict["Batteries"] = len(self.room.collected_batteries)
                self.renderer.render(
                    self.room.agent_position,
                    self.room.special_tiles,
                    info=info_dict,
                    charging_cells=self.room.charging_cells,
                )

            elif self.selected_room == 3:
                collected = bin(self.room.collected_mask).count("1")
                total = len(self.room.shapes_positions)
                info_dict["Shapes"] = f"{collected}/{total}"
                self.renderer.render_with_shapes(
                    agent_position=self.room.agent_position,
                    special_tiles=self.room.special_tiles,
                    info=info_dict,
                    shapes_positions=self.room.shapes_positions,
                    collected_mask=self.room.collected_mask,
                    shape_indices=self.room.shape_indices,
                    current_task="shapes",
                )

            elif self.selected_room == 4:
                self.renderer.render(
                    self.room.agent_position,
                    self.room.special_tiles,
                    info=info_dict,
                    moving_cars=self.room.moving_cars,
                )

            else:
                self.renderer.render(
                    self.room.agent_position,
                    self.room.special_tiles,
                    info=info_dict,
                )

            # Draw UI buttons LAST (on top of everything) - ALWAYS draw them
            self.draw_visualization_buttons(mouse_pos)

            # Force display update to ensure buttons are visible
            pygame.display.flip()

        clock.tick(60)

    if summary:
        self.show_summary_popup(summary)
