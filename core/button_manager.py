import pygame


def draw_visualization_buttons(self, mouse_pos=None):
    """Draws visualization buttons for policy, Q-values, and training progress."""
    button_defs = self.get_visualization_buttons()

    button_width = 110
    button_height = 30
    spacing_x = 20
    spacing_y = 10
    radius = 6
    font = self.font_small

    max_panel_width = self.window_size - 40  # השארת שוליים
    button_width = 110
    spacing_x = 20
    cols = 3
    rows = (len(button_defs) + cols - 1) // cols

    total_width = cols * button_width + (cols - 1) * spacing_x
    start_x = (self.window_size - total_width) // 2
    start_y = 15

    buttons = []

    for i, (btype, label, text_color, bg_color, tooltip) in enumerate(button_defs):
        col = i % cols
        row = i // cols
        x = start_x + col * (button_width + spacing_x)
        y = start_y + row * (button_height + spacing_y)

        rect = pygame.Rect(x, y, button_width, button_height)
        is_hover = mouse_pos and rect.collidepoint(mouse_pos)
        hover_bg = tuple(min(255, c + 30) for c in bg_color) if is_hover else bg_color

        pygame.draw.rect(self.window, hover_bg, rect, border_radius=radius)
        pygame.draw.rect(self.window, (0, 0, 0), rect, 2, border_radius=radius)

        text = font.render(label, True, text_color)
        self.window.blit(text, text.get_rect(center=rect.center))

        if is_hover:
            tip_surf = font.render(tooltip, True, (255, 255, 255))
            tip_bg = pygame.Surface(
                (tip_surf.get_width() + 10, tip_surf.get_height() + 6)
            )
            tip_bg.fill((60, 60, 120))
            self.window.blit(tip_bg, (rect.x, rect.y - 28))
            self.window.blit(tip_surf, (rect.x + 5, rect.y - 25))

        buttons.append((btype, rect))

    return buttons


def draw_back_button(self, mouse_pos=None):
    """
    Draws the 'Back to Room Selection' button on the top-right.
    """
    button_width = 180
    button_height = 38
    y = 30
    x = self.window_size - button_width - 20
    font = self.font_small

    back_rect = pygame.Rect(x, y, button_width, button_height)
    is_hover = mouse_pos and back_rect.collidepoint(mouse_pos)

    back_bg = (220, 220, 255) if is_hover else (200, 200, 240)
    pygame.draw.rect(self.window, back_bg, back_rect, border_radius=8)
    pygame.draw.rect(self.window, (0, 0, 0), back_rect, 2, border_radius=8)
    back_text = font.render("Back to Room Selection", True, (30, 30, 60))
    self.window.blit(back_text, back_text.get_rect(center=back_rect.center))

    return back_rect


def handle_visualization_click(self, button_type):
    print(f"[DEBUG] Clicked button: {button_type}")

    if button_type == "policy" and hasattr(self.room, "plot_policy"):
        print("[DEBUG] Executing: plot_policy")
        self.room.plot_policy()
    elif button_type == "qmap" and hasattr(self.room, "plot_q_values"):
        print("[DEBUG] Executing: plot_q_values")
        self.room.plot_q_values()
    elif button_type == "vmap" and hasattr(self.room, "plot_value_function"):
        print("[DEBUG] Executing: plot_value_function")
        self.room.plot_value_function()
    elif button_type == "reward" and hasattr(self.room, "plot_training_progress"):
        print("[DEBUG] Executing: plot_training_progress")
        self.room.plot_training_progress()
    elif button_type == "epsilon" and hasattr(self.room, "plot_epsilon_curve"):
        print("[DEBUG] Executing: plot_epsilon_curve")
        self.room.plot_epsilon_curve()
    elif button_type == "success" and hasattr(self.room, "plot_success_rate"):
        print("[DEBUG] Executing: plot_success_rate")
        self.room.plot_success_rate()
    elif button_type == "back":
        print("[DEBUG] Executing: back to train_run_choice")
        self.reset_room_state()
        self.game_state = "train_run_choice"
        self.exit_game_loop = True
