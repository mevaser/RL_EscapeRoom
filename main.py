import pygame
import sys
import random
import numpy as np
from rooms.room1_dp import DPRoom
from rooms.room2_sarsa import SARSARoom
from rooms.room3_qlearning import QLearningRoom
from rooms.room4_dqn import DQNRoom
from environment.renderer import GridWorldRenderer
from environment.renderer3_math import GridWorldRenderer as MathRenderer
from utils.parameter_dialogs import get_dp_params
from utils.parameter_dialogs import get_sarsa_params
from utils.parameter_dialogs import get_qlearning_params
from utils.parameter_dialogs import get_dqn_params




class RLEscapeRoom:
    def __init__(self):
        self.current_room = 1
        self.total_rooms = 4
        self.room = None
        self.renderer = None
        self.training = False
        self.snitch_mask = 0
        self.show_controls = True
        self._setup_current_room()

    def _setup_current_room(self):
        print(f"\nEntering Room {self.current_room}...")

        if self.current_room == 1:
            print("Room 1 - Dynamic Programming (Quidditch)")

            # קבלת פרמטרים מהמשתמש
            gamma, theta = get_dp_params()

            # יצירת החדר עם הפרמטרים שהמשתמש הזין
            self.room = DPRoom(gamma=gamma, theta=theta)
            
            # שמירה על הרקע כמו שהיה
            self.renderer = GridWorldRenderer(background_path="assets/images/room1_background.jpg")

            self.room.value_iteration()
            self.room.plot_value_function()
            self.room.plot_policy()
            print("Policy calculated! Press SPACE to see the agent move.")


        elif self.current_room == 2:
            print("Room 2 - SARSA (WALL-E Charging Room)")
            alpha, gamma, epsilon = get_sarsa_params()
            self.room = SARSARoom(alpha=alpha, gamma=gamma, epsilon=epsilon)
            self.renderer = GridWorldRenderer(background_path="assets/images/room2_background.jpg")
            self.training = True
            print("\nTraining SARSA agent for 1000 episodes...")
            self.room.train(num_episodes=1000)
            self.room.plot_training_progress()
            self.room.plot_q_values()
            self.room.plot_policy()
            self.training = False
            print("Training complete! Press SPACE to see the agent move.")


        elif self.current_room == 3:
            alpha, gamma, epsilon, epsilon_decay, min_epsilon = get_qlearning_params()
            self.room = QLearningRoom(alpha=alpha, gamma=gamma, epsilon=epsilon,
                                    epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
            self.renderer = MathRenderer(background_path="assets/images/room3_background.jpg")
            self.training = True
            print("\nTraining Q-Learning agent for 1000 episodes...")
            self.room.train(num_episodes=1000)
            self.room.plot_training_progress()
            self.training = False
            print("Training complete! Press SPACE to see the agent move.")


        elif self.current_room == 4:
            print("Room 4 - DQN")

            (learning_rate, gamma, epsilon, epsilon_decay, min_epsilon,
            batch_size, tau, hidden_size) = get_dqn_params()

            self.room = DQNRoom(
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                min_epsilon=min_epsilon,
                batch_size=batch_size,
                tau=tau,
                hidden_size=hidden_size
            )

            self.renderer = GridWorldRenderer(background_path="assets/images/room4_background.jpg")
            self.training = True
            print("\nTraining DQN agent for 2000 episodes...")
            self.room.train(num_episodes=2000)
            self.room.plot_training_progress()
            self.room.plot_q_values()
            self.room.plot_policy()
            self.training = False
            print("Training complete! Press SPACE to see the agent move.")


        self.room.reset()
        self.snitch_mask = 0

    def capture_background_snapshot(self):
        if self.current_room == 3:
            task_text = "Squid Game Room: Collect Circle, then Square, then Triangle"
            snapshot = self.renderer.render_with_shapes(
                self.room.agent_position,
                self.room.special_tiles,
                shapes_positions=self.room.shapes_positions,
                collected_shapes=self.room.collected_shapes,
                current_task=task_text
            ).copy()
        elif self.current_room == 2:
            snapshot = self.renderer.render(
                self.room.agent_position,
                self.room.special_tiles,
                info={"Room": f"{self.current_room}/4", "Steps": self.room.steps, "Training": self.training},
                charging_cells=self.room.charging_cells
            ).copy()
        elif self.current_room == 4:
            snapshot = self.renderer.render(
                self.room.agent_position,
                self.room.special_tiles,
                info={"Room": f"{self.current_room}/4", "Steps": self.room.steps, "Training": self.training},
                moving_cars=self.room.moving_cars
            ).copy()
        else:
            snapshot = self.renderer.render(
                self.room.agent_position,
                self.room.special_tiles,
                info={"Room": f"{self.current_room}/4", "Steps": self.room.steps, "Training": self.training}
            ).copy()
        return pygame.surfarray.make_surface(np.transpose(snapshot, (1, 0, 2)))

    def show_room_completed_popup(self):
        popup_running = True
        pygame_surface = self.capture_background_snapshot()

        popup_width, popup_height = 400, 250
        popup_x = (self.renderer.window_size - popup_width) // 2
        popup_y = (self.renderer.window_size - popup_height) // 2
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

        button_width, button_height = 150, 50
        button_x = (self.renderer.window_size - button_width) // 2
        button_y = popup_y + popup_height - 80
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        font_large = pygame.font.Font(None, 40)
        font_small = pygame.font.Font(None, 30)

        while popup_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        popup_running = False

            self.renderer.window.blit(pygame_surface, (0, 0))
            pygame.draw.rect(self.renderer.window, (255, 255, 255), popup_rect)
            pygame.draw.rect(self.renderer.window, (0, 0, 0), popup_rect, 3)

            text_surface = font_large.render("Room completed!", True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(self.renderer.window_size//2, popup_y + 60))
            self.renderer.window.blit(text_surface, text_rect)

            subtext_surface = font_small.render("Press OK to move to next room", True, (80, 80, 80))
            subtext_rect = subtext_surface.get_rect(center=(self.renderer.window_size//2, popup_y + 110))
            self.renderer.window.blit(subtext_surface, subtext_rect)

            pygame.draw.rect(self.renderer.window, (0, 128, 0), button_rect)
            button_text = font_small.render("OK", True, (255, 255, 255))
            button_text_rect = button_text.get_rect(center=button_rect.center)
            self.renderer.window.blit(button_text, button_text_rect)

            pygame.display.flip()
            pygame.time.Clock().tick(30)


    def show_game_completed_popup(self):
        popup_running = True
        pygame_surface = self.capture_background_snapshot()

        popup_width, popup_height = 500, 300
        popup_x = (self.renderer.window_size - popup_width) // 2
        popup_y = (self.renderer.window_size - popup_height) // 2
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

        button_width, button_height = 180, 60
        button_x = (self.renderer.window_size - button_width) // 2
        button_y = popup_y + popup_height - 80
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        font_large = pygame.font.Font(None, 42)
        font_small = pygame.font.Font(None, 32)

        while popup_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        popup_running = False

            self.renderer.window.blit(pygame_surface, (0, 0))
            pygame.draw.rect(self.renderer.window, (255, 255, 255), popup_rect)
            pygame.draw.rect(self.renderer.window, (0, 0, 0), popup_rect, 3)

            text_surface = font_large.render("Congratulations!", True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(self.renderer.window_size//2, popup_y + 60))
            self.renderer.window.blit(text_surface, text_rect)

            subtext_surface = font_small.render("You have completed the Escape Room!", True, (80, 80, 80))
            subtext_rect = subtext_surface.get_rect(center=(self.renderer.window_size//2, popup_y + 120))
            self.renderer.window.blit(subtext_surface, subtext_rect)

            pygame.draw.rect(self.renderer.window, (0, 128, 0), button_rect)
            button_text = font_small.render("OK", True, (255, 255, 255))
            button_text_rect = button_text.get_rect(center=button_rect.center)
            self.renderer.window.blit(button_text, button_text_rect)

            pygame.display.flip()
            pygame.time.Clock().tick(30)

        pygame.quit()
        sys.exit()


    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        self.show_controls = not self.show_controls
                    elif event.key == pygame.K_SPACE and not self.training:
                        if self.current_room == 1:
                            x, y = self.room.agent_position
                            action = self.room.policy[x, y, self.snitch_mask]
                        elif self.current_room == 3:
                            state = tuple(self.room.agent_position)
                            action = self.room.get_action(state, training=False)
                        elif self.current_room == 4:
                            action = self.room.get_action_from_successful_trajectory()
                            if action is None:
                                continue  # דלג על הצעד – לא לעשות כלום


                        else:
                            action = self.room.get_action(self.room.agent_position, training=False)



                        obs, reward, terminated, info = self.room.step(action)

                        if info.get("timeout", False):
                            print("\nEpisode ended due to step limit (timeout).")

                        if self.current_room == 1:
                            pos = tuple(self.room.agent_position)
                            if hasattr(self.room, 'snitch_indices') and pos in self.room.snitch_indices:
                                idx = self.room.snitch_indices[pos]
                                if not (self.snitch_mask & (1 << idx)):
                                    self.snitch_mask |= (1 << idx)
                        


                        if terminated and (self.current_room != 4 or info.get("success") == True):
                            print("\nGoal reached! 🎉")
                            if self.current_room == self.total_rooms:
                                self.show_game_completed_popup()
                            else:
                                self.show_room_completed_popup()
                            if self.current_room < self.total_rooms:
                                self.current_room += 1
                                self._setup_current_room()
                            continue



                    elif event.key == pygame.K_r:
                        print("\nResetting room...")
                        self.room.reset()

                    elif event.key == pygame.K_n and self.current_room < self.total_rooms:
                        self.current_room += 1
                        self._setup_current_room()

            if self.room and self.renderer:
                info_dict = {"Room": f"{self.current_room}/4", "Steps": self.room.steps, "Training": self.training}

                if self.current_room == 1:
                    info_dict.update({"snitch_collected": self.room.collected_snitch, "snitch_total": self.room.total_snitch})

                if self.current_room == 2:
                    info_dict["Batteries"] = len(self.room.collected_batteries)
                    self.renderer.render(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict,
                        charging_cells=self.room.charging_cells
                    )


                elif self.current_room == 3:
                    task_text = "Squid Game Room: Collect Circle, then Square, then Triangle"
                    
                    # 💡 תעדכן את info_dict מראש לפני הקריאה:
                    collected = len(self.room.collected_shapes)
                    total = len(self.room.shapes_positions)
                    info_dict["Shapes"] = f"{collected}/{total}"

                    self.renderer.render_with_shapes(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict,
                        shapes_positions=self.room.shapes_positions,
                        collected_shapes=self.room.collected_shapes,
                        current_task="shapes"
                    )


                elif self.current_room == 4:
                    if self.room.successful_trajectories and not hasattr(self.room, "current_replay_trajectory"):
                        _, traj = random.choice(self.room.successful_trajectories)
                        self.room.current_replay_trajectory = traj
                        # 🎯 מיד אחרי בחירת האפיזודה - מיקום המכוניות במקום הנכון
                        start_cars = traj[0]["cars"]
                        for i, pos in enumerate(start_cars):
                            if i < len(self.room.moving_cars):
                                self.room.moving_cars[i]["position"] = pos
                    # הסר את השורה הזו כי היא מיותרת עכשיו
                    # self.room.restore_initial_replay_state()
                    self.room.restore_initial_replay_state()

                    self.renderer.render(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict,
                        moving_cars=self.room.moving_cars
                    )

                else:
                    self.renderer.render(self.room.agent_position, self.room.special_tiles, info=info_dict)

            clock.tick(60)

        if self.renderer:
            self.renderer.close()

if __name__ == "__main__":
    print("\n\U0001F3AE Welcome to RL Escape Room! \U0001F3AE")
    game = RLEscapeRoom()
    game.run()
