import tkinter as tk
from tkinter import font as tkFont


def get_dp_params():
    """
    Open a Tkinter window to get DP parameters from user (all in one window, improved visuals).
    """

    def submit():
        nonlocal gamma, theta
        gamma = float(gamma_entry.get())
        theta = float(theta_entry.get())
        root.destroy()

    root = tk.Tk()
    root.title("DP Parameters")

    # קצת נגדיל גופן
    default_font = tkFont.Font(size=12)

    # Gamma
    tk.Label(root, text="Discount Factor (gamma):", font=default_font).grid(
        row=0, column=0, padx=10, pady=10
    )
    gamma_entry = tk.Entry(root, font=default_font)
    gamma_entry.insert(0, "0.99")
    gamma_entry.grid(row=0, column=1, padx=10, pady=10)

    # Theta
    tk.Label(root, text="Convergence Threshold (theta):", font=default_font).grid(
        row=1, column=0, padx=10, pady=10
    )
    theta_entry = tk.Entry(root, font=default_font)
    theta_entry.insert(0, "1e-6")
    theta_entry.grid(row=1, column=1, padx=10, pady=10)

    # Submit button - נגדיל אותו
    submit_button = tk.Button(
        root,
        text="OK",
        font=tkFont.Font(size=12, weight="bold"),
        width=10,
        height=2,
        command=submit,
    )
    submit_button.grid(row=2, columnspan=2, pady=15)

    # Center window on screen
    root.eval("tk::PlaceWindow . center")

    gamma, theta = 0.99, 1e-6  # fallback defaults
    root.mainloop()

    return gamma, theta


def get_sarsa_params():
    """
    Open a Tkinter window to get SARSA parameters from user (all in one window).
    """

    def submit():
        nonlocal alpha, gamma, epsilon
        alpha = float(alpha_entry.get())
        gamma = float(gamma_entry.get())
        epsilon = float(epsilon_entry.get())
        root.destroy()

    root = tk.Tk()
    root.title("SARSA Parameters")

    default_font = tkFont.Font(size=12)

    # Alpha
    tk.Label(root, text="Learning Rate (alpha):", font=default_font).grid(
        row=0, column=0, padx=10, pady=10
    )
    alpha_entry = tk.Entry(root, font=default_font)
    alpha_entry.insert(0, "0.1")
    alpha_entry.grid(row=0, column=1, padx=10, pady=10)

    # Gamma
    tk.Label(root, text="Discount Factor (gamma):", font=default_font).grid(
        row=1, column=0, padx=10, pady=10
    )
    gamma_entry = tk.Entry(root, font=default_font)
    gamma_entry.insert(0, "0.99")
    gamma_entry.grid(row=1, column=1, padx=10, pady=10)

    # Epsilon
    tk.Label(root, text="Exploration Rate (epsilon):", font=default_font).grid(
        row=2, column=0, padx=10, pady=10
    )
    epsilon_entry = tk.Entry(root, font=default_font)
    epsilon_entry.insert(0, "1")
    epsilon_entry.grid(row=2, column=1, padx=10, pady=10)

    submit_button = tk.Button(
        root,
        text="OK",
        font=tkFont.Font(size=12, weight="bold"),
        width=10,
        height=2,
        command=submit,
    )
    submit_button.grid(row=3, columnspan=2, pady=15)

    root.eval("tk::PlaceWindow . center")

    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    root.mainloop()

    return alpha, gamma, epsilon


def get_qlearning_params():
    def submit():
        nonlocal submitted
        submitted = True
        try:
            nonlocal alpha, gamma, epsilon, epsilon_decay, min_epsilon
            alpha = float(alpha_entry.get())
            gamma = float(gamma_entry.get())
            epsilon = float(epsilon_entry.get())
            epsilon_decay = float(epsilon_decay_entry.get())
            min_epsilon = float(min_epsilon_entry.get())
        except ValueError:
            print("Invalid input, using defaults.")
        root.destroy()

    # ערכי ברירת מחדל
    alpha, gamma, epsilon, epsilon_decay, min_epsilon = 0.1, 0.99, 1.0, 0.995, 0.01
    submitted = False

    root = tk.Tk()
    root.title("Q-Learning Parameters")

    default_font = tkFont.Font(size=12)

    labels_defaults = [
        ("Learning Rate (alpha):", alpha),
        ("Discount Factor (gamma):", gamma),
        ("Exploration Rate (epsilon):", epsilon),
        ("Epsilon Decay:", epsilon_decay),
        ("Minimum Epsilon:", min_epsilon),
    ]
    entries = []
    for i, (label, default) in enumerate(labels_defaults):
        tk.Label(root, text=label, font=default_font).grid(
            row=i, column=0, padx=10, pady=10
        )
        entry = tk.Entry(root, font=default_font)
        entry.insert(0, str(default))
        entry.grid(row=i, column=1, padx=10, pady=10)
        entries.append(entry)

    alpha_entry, gamma_entry, epsilon_entry, epsilon_decay_entry, min_epsilon_entry = (
        entries
    )

    submit_button = tk.Button(
        root,
        text="OK",
        font=tkFont.Font(size=12, weight="bold"),
        width=10,
        height=2,
        command=submit,
    )
    submit_button.grid(row=len(entries), columnspan=2, pady=15)

    root.eval("tk::PlaceWindow . center")
    root.mainloop()

    if not submitted:
        print("User closed the window – using default parameters.")

    return alpha, gamma, epsilon, epsilon_decay, min_epsilon


def get_dqn_params():
    """
    Open a Tkinter window to get DQN parameters from user (all in one window).
    """

    def submit():
        nonlocal learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch_size, tau, hidden_size
        learning_rate = float(learning_rate_entry.get())
        gamma = float(gamma_entry.get())
        epsilon = float(epsilon_entry.get())
        epsilon_decay = float(epsilon_decay_entry.get())
        min_epsilon = float(min_epsilon_entry.get())
        batch_size = int(batch_size_entry.get())
        tau = float(tau_entry.get())
        hidden_size = int(hidden_size_entry.get())
        root.destroy()

    root = tk.Tk()
    root.title("DQN Parameters")

    default_font = tkFont.Font(size=12)

    # יצירת כל השדות:
    labels_entries = [
        ("Learning Rate (lr):", "0.001"),
        ("Discount Factor (gamma):", "0.99"),
        ("Exploration Rate (epsilon):", "1.0"),
        ("Epsilon Decay:", "0.995"),
        ("Minimum Epsilon:", "0.01"),
        ("Batch Size:", "64"),
        ("Soft Update (tau):", "0.001"),
        ("Hidden Layer Size:", "64"),
    ]

    entries = []
    for i, (label_text, default_value) in enumerate(labels_entries):
        tk.Label(root, text=label_text, font=default_font).grid(
            row=i, column=0, padx=10, pady=5
        )
        entry = tk.Entry(root, font=default_font)
        entry.insert(0, default_value)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries.append(entry)

    (
        learning_rate_entry,
        gamma_entry,
        epsilon_entry,
        epsilon_decay_entry,
        min_epsilon_entry,
        batch_size_entry,
        tau_entry,
        hidden_size_entry,
    ) = entries

    submit_button = tk.Button(
        root,
        text="OK",
        font=tkFont.Font(size=12, weight="bold"),
        width=10,
        height=2,
        command=submit,
    )
    submit_button.grid(row=len(labels_entries), columnspan=2, pady=15)

    root.eval("tk::PlaceWindow . center")

    # ערכי ברירת מחדל
    (
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
        batch_size,
        tau,
        hidden_size,
    ) = (0.001, 0.99, 1.0, 0.995, 0.01, 64, 0.001, 64)

    root.mainloop()

    return (
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
        batch_size,
        tau,
        hidden_size,
    )
