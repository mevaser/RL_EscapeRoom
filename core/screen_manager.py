import pygame


def show_start_screen(self):
    """Display the main start screen with room selection buttons"""
    self.window.fill((50, 50, 100))  # Dark blue background

    # Title
    title_text = self.font_large.render("RL Escape Room", True, (255, 255, 255))
    title_rect = title_text.get_rect(center=(self.window_size // 2, 100))
    self.window.blit(title_text, title_rect)

    rooms = [
        ("Room 1 – Harry Potter (DP)", 200),
        ("Room 2 – WALL-E (SARSA)", 300),
        ("Room 3 – Squid Game (Q-Learning)", 400),
        ("Room 4 – Rush Hour (DQN)", 500),
    ]

    button_width = 500
    button_height = 60
    self.room_buttons = []

    for i, (text, y) in enumerate(rooms):
        x = (self.window_size - button_width) // 2
        button_rect = pygame.Rect(x, y, button_width, button_height)
        self.room_buttons.append((button_rect, i + 1))

        pygame.draw.rect(self.window, (100, 150, 255), button_rect)
        pygame.draw.rect(self.window, (255, 255, 255), button_rect, 3)

        button_text = self.font_medium.render(text, True, (255, 255, 255))
        text_rect = button_text.get_rect(center=button_rect.center)
        self.window.blit(button_text, text_rect)

    pygame.display.flip()


def show_train_run_choice(self):
    """Display the train/run choice screen"""
    self.window.fill((50, 50, 100))

    room_names = [
        "Harry Potter (DP)",
        "WALL-E (SARSA)",
        "Squid Game (Q-Learning)",
        "Rush Hour (DQN)",
    ]
    title_text = self.font_large.render(
        f"Room {self.selected_room} - {room_names[self.selected_room-1]}",
        True,
        (255, 255, 255),
    )
    title_rect = title_text.get_rect(center=(self.window_size // 2, 150))
    self.window.blit(title_text, title_rect)

    button_width = 400
    button_height = 80
    spacing = 30

    center_x = (self.window_size - button_width) // 2
    start_y = 280

    train_button = pygame.Rect(center_x, start_y, button_width, button_height)
    run_button = pygame.Rect(
        center_x, start_y + button_height + spacing, button_width, button_height
    )

    pygame.draw.rect(self.window, (0, 150, 0), train_button)
    pygame.draw.rect(self.window, (255, 255, 255), train_button, 3)
    train_text = self.font_medium.render("Train Agent", True, (255, 255, 255))
    self.window.blit(train_text, train_text.get_rect(center=train_button.center))

    pygame.draw.rect(self.window, (150, 0, 0), run_button)
    pygame.draw.rect(self.window, (255, 255, 255), run_button, 3)
    run_text = self.font_medium.render("Run Agent", True, (255, 255, 255))
    self.window.blit(run_text, run_text.get_rect(center=run_button.center))

    back_button = pygame.Rect(50, 50, 200, 50)
    pygame.draw.rect(self.window, (100, 100, 150), back_button)
    pygame.draw.rect(self.window, (255, 255, 255), back_button, 2)
    back_text = self.font_small.render(
        "← Back to Room Selection", True, (255, 255, 255)
    )
    self.window.blit(back_text, back_text.get_rect(center=back_button.center))

    self.train_run_buttons = [train_button, run_button, back_button]
    pygame.display.flip()
