import pygame


def draw_visualization_buttons(self, mouse_pos=None):
    """
    Draws 2x2 buttons for visualizations (Policy, Q Map, Reward, Back to Menu),
    centered in the top info panel.
    """
    button_width = 110
    button_height = 30
    spacing_x = 20
    spacing_y = 10
    radius = 6
    font = self.font_small

    button_defs = [
        ("policy", "Policy", (255, 255, 255), (100, 150, 255), "View learned policy"),
        ("qmap", "Q Map", (255, 255, 255), (150, 100, 255), "Q-value heatmap"),
        ("reward", "Reward", (255, 255, 255), (255, 100, 150), "Reward graph"),
        ("back", "↩ Menu", (255, 255, 255), (180, 180, 180), "Back to room menu"),
    ]

    cols, rows = 2, 2
    total_width = cols * button_width + (cols - 1) * spacing_x
    start_x = (self.window_size - total_width) // 2
    start_y = 15  # Info panel top offset

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
            # Tooltip
            tip_surf = font.render(tooltip, True, (255, 255, 255))
            tip_bg = pygame.Surface(
                (tip_surf.get_width() + 10, tip_surf.get_height() + 6)
            )
            tip_bg.fill((60, 60, 120))
            self.window.blit(tip_bg, (rect.x, rect.y - 28))
            self.window.blit(tip_surf, (rect.x + 5, rect.y - 25))

        buttons.append((btype, rect))

    return buttons


def draw_back_and_stop_buttons(self, mouse_pos=None):
    """
    Draws the 'Back to Room Selection' and 'Stop Game' buttons on top-right.
    """
    button_width = 160
    button_height = 38
    y = 30
    spacing = 20
    start_x = self.window_size - (button_width * 2 + spacing + 20)
    font = self.font_small

    back_rect = pygame.Rect(start_x, y, button_width, button_height)
    stop_rect = pygame.Rect(
        start_x + button_width + spacing, y, button_width, button_height
    )

    is_hover_back = mouse_pos and back_rect.collidepoint(mouse_pos)
    is_hover_stop = mouse_pos and stop_rect.collidepoint(mouse_pos)

    # Back button
    back_bg = (220, 220, 255) if is_hover_back else (200, 200, 240)
    pygame.draw.rect(self.window, back_bg, back_rect, border_radius=8)
    pygame.draw.rect(self.window, (0, 0, 0), back_rect, 2, border_radius=8)
    back_text = font.render("Back to Room Selection", True, (30, 30, 60))
    self.window.blit(back_text, back_text.get_rect(center=back_rect.center))

    # Stop button
    stop_bg = (255, 180, 180) if is_hover_stop else (255, 220, 220)
    pygame.draw.rect(self.window, stop_bg, stop_rect, border_radius=8)
    pygame.draw.rect(self.window, (0, 0, 0), stop_rect, 2, border_radius=8)
    stop_text = font.render("Stop Game", True, (120, 60, 60))
    self.window.blit(stop_text, stop_text.get_rect(center=stop_rect.center))

    return back_rect, stop_rect


def handle_visualization_click(self, button_type):
    """
    Respond to visualization button clicks by calling room plotting methods.
    """
    if button_type == "policy" and hasattr(self.room, "plot_policy"):
        self.room.plot_policy()
    elif button_type == "qmap" and hasattr(self.room, "plot_q_values"):
        self.room.plot_q_values()
    elif button_type == "reward" and hasattr(self.room, "plot_training_progress"):
        self.room.plot_training_progress()
