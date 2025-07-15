import os
import pickle


def save_agent_state(self, room_num):
    """Save the trained agent state to a pickle file."""
    model_path = f"saved_models/room{room_num}_agent.pkl"

    if room_num == 1:
        # Dynamic Programming agent
        agent_state = {
            "policy": self.room.policy,
            "V": self.room.V,
            "gamma": self.room.gamma,
            "theta": self.room.theta,
        }
    elif room_num == 2:
        # SARSA agent
        agent_state = {
            "Q": self.room.Q,
            "alpha": self.room.alpha,
            "gamma": self.room.gamma,
            "epsilon": self.room.epsilon,
        }
    elif room_num == 3:
        # Q-Learning agent
        agent_state = {
            "Q": self.room.Q,
            "alpha": self.room.alpha,
            "gamma": self.room.gamma,
            "epsilon": self.room.epsilon,
        }
    elif room_num == 4:
        # DQN agent
        agent_state = {
            "qnetwork_local_state_dict": self.room.qnetwork_local.state_dict(),
            "qnetwork_target_state_dict": self.room.qnetwork_target.state_dict(),
            "epsilon": self.room.epsilon,
        }
    else:
        raise ValueError(f"Unsupported room number: {room_num}")

    # Save to disk
    with open(model_path, "wb") as f:
        pickle.dump(agent_state, f)

    print(f"✅ Agent state saved to {model_path}")


def load_agent_state(self, room_num):
    """Load the trained agent state from disk."""
    model_path = f"saved_models/room{room_num}_agent.pkl"

    if not os.path.exists(model_path):
        return False

    with open(model_path, "rb") as f:
        agent_state = pickle.load(f)

    if room_num == 1:
        self.room.policy = agent_state["policy"]
        self.room.V = agent_state["V"]
    elif room_num == 2:
        self.room.Q = agent_state["Q"]
    elif room_num == 3:
        self.room.Q = agent_state["Q"]
    elif room_num == 4:
        self.room.qnetwork_local.load_state_dict(
            agent_state["qnetwork_local_state_dict"]
        )
        self.room.qnetwork_target.load_state_dict(
            agent_state["qnetwork_target_state_dict"]
        )
        self.room.epsilon = agent_state["epsilon"]

    print(f"✅ Agent state loaded from {model_path}")
    return True
