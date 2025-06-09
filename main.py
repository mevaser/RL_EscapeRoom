
import pygame
import numpy as np
from rooms.room1_dp import DPRoom
from rooms.room2_sarsa import SARSARoom
from rooms.room3_qlearning import QLearningRoom
from rooms.room4_dqn import DQNRoom
from environment.renderer import GridWorldRenderer
from environment.renderer3_math import GridWorldRenderer as MathRenderer

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
            self.room = DPRoom()
            self.renderer = GridWorldRenderer(background_path="assets/images/room1_background.jpg")
            self.room.value_iteration()
            self.room.plot_value_function()
            print("Policy calculated! Press SPACE to see the agent move.")

        elif self.current_room == 2:
            print("Room 2 - SARSA (WALL-E Charging Room)")
            self.room = SARSARoom()
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
            print("Room 3 - Q-Learning (Math Room)")
            self.room = QLearningRoom()
            self.renderer = MathRenderer(background_path="assets/images/room3_background.jpg")
            self.training = True
            print("\nTraining Q-Learning agent for 1000 episodes...")
            self.room.train(num_episodes=1000)
            self.room.plot_training_progress()
            self.training = False
            print("Training complete! Press SPACE to see the agent move.")

        elif self.current_room == 4:
            print("Room 4 - DQN")
            self.room = DQNRoom()
            self.renderer = GridWorldRenderer(background_path="assets/images/room4_background.jpg")
            self.training = True
            print("\nTraining DQN agent for 1000 episodes...")
            self.room.train(num_episodes=1000)
            self.room.plot_training_progress()
            self.room.plot_q_values()
            self.room.plot_policy()
            self.training = False
            print("Training complete! Press SPACE to see the agent move.")

        self.room.reset()
        self.snitch_mask = 0

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

                        if terminated:
                            print("\nGoal reached! 🎉 Moving automatically to next room.")
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
                info_dict = {
                    "Room": f"{self.current_room}/4",
                    "Steps": self.room.steps,
                    "Training": self.training
                }

                if self.current_room == 1:
                    info_dict.update({
                        "snitch_collected": self.room.collected_snitch,
                        "snitch_total": self.room.total_snitch
                    })

                if self.current_room == 2:
                    self.renderer.render(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict,
                        charging_cells=self.room.charging_cells
                    )
                elif self.current_room == 3:
                    task_text = "Squid Game Room: Collect Circle, then Square, then Triangle"
                    self.renderer.render_with_shapes(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict,
                        shapes_positions=self.room.shapes_positions,
                        collected_shapes=self.room.collected_shapes,   # 👈 זה השורה שהייתה חסרה
                        current_task=task_text
                    )


                elif self.current_room == 4:
                    self.renderer.render(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict,
                        moving_cars=self.room.moving_cars
                    )

                else:
                    self.renderer.render(
                        self.room.agent_position,
                        self.room.special_tiles,
                        info=info_dict
                    )



            clock.tick(60)

        if self.renderer:
            self.renderer.close()

if __name__ == "__main__":
    print("\n🎮 Welcome to RL Escape Room! 🎮")
    game = RLEscapeRoom()
    game.run()
