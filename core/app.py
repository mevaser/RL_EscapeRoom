import os
import pygame

from core.screen_manager import show_start_screen, show_train_run_choice
from core.agent_state import save_agent_state, load_agent_state
from core.button_manager import (
    draw_visualization_buttons,
    draw_back_button,
    handle_visualization_click,
)
from core.popup_manager import (
    show_not_trained_popup,
    show_room_completed_popup,
    show_summary_popup,
)
from core.room_handler import (
    setup_room,
    reset_room_state,
    capture_background_snapshot,
    run_game_loop,
)


class RLEscapeRoom:
    def __init__(self):
        """Initialize the RL Escape Room main application state"""
        self.current_room = None
        self.room = None
        self.renderer = None
        self.training = False
        self.snitch_mask = 0
        self.show_controls = True
        self.game_state = "start_screen"
        self.selected_room = None
        self.action_mode = None

        # Needed for screen_manager / transitions
        self.room_buttons = []
        self.train_run_buttons = []

        # Create saved_models directory if it doesn't exist
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")

        # Initialize Pygame window
        pygame.init()
        self.window_size = 640
        self.grid_cell_size = self.window_size // 10  # assuming 10x10 grid
        self.grid_height = self.grid_cell_size * 10
        self.info_panel_height = 130
        self.total_height = self.grid_height + self.info_panel_height

        self.window = pygame.display.set_mode((self.window_size, self.total_height))

        pygame.display.set_caption("RL Escape Room")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Bind external methods to this instance
        self.show_start_screen = show_start_screen.__get__(self)
        self.show_train_run_choice = show_train_run_choice.__get__(self)
        self.save_agent_state = save_agent_state.__get__(self)
        self.load_agent_state = load_agent_state.__get__(self)
        self.draw_visualization_buttons = draw_visualization_buttons.__get__(self)
        self.draw_back_button = draw_back_button.__get__(self)
        self.handle_visualization_click = handle_visualization_click.__get__(self)
        self.show_not_trained_popup = show_not_trained_popup.__get__(self)
        self.show_room_completed_popup = show_room_completed_popup.__get__(self)
        self.show_summary_popup = show_summary_popup.__get__(self)
        self.setup_room = setup_room.__get__(self)
        self.reset_room_state = reset_room_state.__get__(self)
        self.capture_background_snapshot = capture_background_snapshot.__get__(self)
        self.run_game_loop = run_game_loop.__get__(self)

    def get_visualization_buttons(self):
        """Returns a list of visualization buttons with their properties."""
        buttons = [
            (
                "policy",
                "Policy",
                (255, 255, 255),
                (100, 150, 255),
                "View learned policy",
            ),
            ("reward", "Reward", (255, 255, 255), (255, 100, 150), "Reward curve"),
        ]

        if hasattr(self.room, "plot_q_values"):
            buttons.insert(
                1,
                ("qmap", "Q Map", (255, 255, 255), (150, 100, 255), "Q-value heatmap"),
            )

        if hasattr(self.room, "plot_value_function"):
            buttons.insert(
                1,
                (
                    "vmap",
                    "V Map",
                    (255, 255, 255),
                    (100, 255, 200),
                    "State value function",
                ),
            )

        if hasattr(self.room, "plot_epsilon_curve"):
            buttons.append(
                (
                    "epsilon",
                    "Epsilon",
                    (255, 255, 255),
                    (100, 200, 255),
                    "Epsilon decay curve",
                )
            )

        if hasattr(self.room, "plot_success_rate"):
            buttons.append(
                ("success", "Success", (255, 255, 255), (120, 255, 120), "Success rate")
            )

        buttons.append(
            ("back", "↩ Menu", (255, 255, 255), (180, 180, 180), "Back to room menu")
        )

        return buttons

    def run(self):
        """Main application loop for state transitions"""
        running = True

        while running:
            if self.game_state == "start_screen":
                self.show_start_screen()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        for button_rect, room_num in self.room_buttons:
                            if button_rect.collidepoint(event.pos):
                                self.selected_room = room_num
                                self.game_state = "train_run_choice"
                                break

            elif self.game_state == "train_run_choice":
                self.show_train_run_choice()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.train_run_buttons[0].collidepoint(event.pos):
                            if self.setup_room(self.selected_room, "train"):
                                self.game_state = "playing"
                        elif self.train_run_buttons[1].collidepoint(event.pos):
                            if self.setup_room(self.selected_room, "run"):
                                self.game_state = "playing"
                        elif self.train_run_buttons[2].collidepoint(event.pos):
                            self.game_state = "start_screen"

            elif self.game_state == "playing":
                self.run_game_loop()
                self.game_state = "start_screen"

        if self.renderer:
            self.renderer.close()
