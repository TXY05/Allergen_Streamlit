import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import random
from gymnasium import spaces
from stable_baselines3 import A2C, SAC, PPO

# ==========================
# Page setup
# ==========================
st.set_page_config (page_title = "RL Model Comparison", layout = "wide")
st.title ("RL Model Comparison Dashboard")


# ==========================
# Loaders
# ==========================
@st.cache_resource
def load_sb3 ( model_type, path ):
    if model_type == "A2C":
        return A2C.load (path, device = "cpu")
    if model_type == "SAC":
        return SAC.load (path, device = "cpu")
    if model_type == "PPO":
        return PPO.load (path, device = "cpu")
    raise ValueError ("Unsupported SB3 model")


class DQNNetwork (nn.Module):
    def __init__ ( self, state_dim, action_dim ):
        super ().__init__ ()
        self.net = nn.Sequential (
            nn.Linear (state_dim, 128),
            nn.ReLU (),
            nn.Linear (128, 128),
            nn.ReLU (),
            nn.Linear (128, action_dim),
        )

    def forward ( self, x ):
        return self.net (x)


@st.cache_resource
def load_dqn_pt ( path, obs_size ):
    checkpoint = torch.load (path, map_location = "cpu")
    # Full model
    if isinstance (checkpoint, nn.Module):
        model = checkpoint
        model.eval ()
        return model
    # State dict
    else:
        # auto-detect output size
        n_actions = None
        for k, v in checkpoint.items ():
            if 'net.4.weight' in k:
                n_actions = v.shape [0]
                break
        if n_actions is None:
            raise ValueError ("Cannot detect output size from state dict.")
        model = DQNNetwork (obs_size, n_actions)
        # remove 'module.' prefix if exists
        new_state_dict = {k.replace ('module.', ''): v for k, v in checkpoint.items ()}
        model.load_state_dict (new_state_dict)
        model.eval ()
        return model


# ==========================
# Model registry
# ==========================
MODELS = {
    "A2C": {
        "type": "sb3",
        "algo": "A2C",
        "path": "models/best_a2c_model.zip",
    },
    "SAC": {
        "type": "sb3",
        "algo": "SAC",
        "path": "models/best_sac_model.zip",
    },
    "PPO": {
        "type": "sb3",
        "algo": "PPO",
        "path": "models/ppo.zip",
    },
    "DQN (PyTorch)": {
        "type": "pt",
        "algo": "DQN",
        "path": "models/dqn_allergen_model.pt",
    },
}

# ==========================
# Sidebar
# ==========================
st.sidebar.header ("Simulation Settings")
selected_models = st.sidebar.multiselect ("Select models", MODELS.keys (), default = list (MODELS.keys ()))
steps = st.sidebar.slider ("Steps", 60, 1440, 1440, 60)
run = st.sidebar.button ("Run")


# ==========================
# AllergenEnvironment
# ==========================
class AllergenEnvironment (gym.Env):
    INTERIOR_VOLUME = 68
    MECHANICAL_VENTILATION_FLOW_RATE = 4.2
    MECHANICAL_SUPPLY_FILTER_EFFICIENCY = 0.9
    AIR_PURIFIER_FLOW_RATE = 4.8
    VACUUM_FLOW_RATE = 0.6
    AIR_PURIFIER_FILTER_EFFICIENCY = 0.9
    NATURAL_VENTILATION_FLOW_RATE = 0
    NATURAL_INFILTRATION_FLOW_RATE = 0.56
    PARTICLE_FRACTION_IN_FLOW_PATH = 0.7
    DEPOSITION_RATE_OF_PARTICLE = 0.0067

    def __init__ ( self, dqn_action_space = False ):
        super ().__init__ ()
        # standard state space
        self.observation_space = spaces.Box (
            low = np.array ([12, 12, 0.46, 0, 0], dtype = np.float32),
            high = np.array ([300, 135, 50, 50, 1439], dtype = np.float32),
            dtype = np.float32
        )
        # Action space
        if dqn_action_space:
            # Adjust to match DQN output size
            self.action_space = spaces.MultiBinary (8)  # example: match checkpoint
        else:
            self.action_space = spaces.MultiBinary (3)

        self.episode_length = 1440
        self.current_step_in_episode = 0
        self.state = None

        # Example emission schedule
        self.EMISSION_SCHEDULE = [0] * 48  # replace with real schedule

    def allergenConcentrationCalculation ( self, E, Co, Ci, Qs, Qn, Ql, Qh, eta_s, P, Qf, eta_f, beta, V ):
        outdoor_term = Co * (Qs * (1 - eta_s) + Qn + (Ql + Qh) * P)
        removal_term = Ci * (Qf * eta_f + beta * V + (Qs + Qn + Ql + Qh))
        return (E + outdoor_term - removal_term) / V

    def reset ( self, *, seed = None, options = None ):
        super ().reset (seed = seed)
        if seed is not None:
            random.seed (seed)
            np.random.seed (seed)
        self.state = np.array ([25.0, random.uniform (12, 135), random.uniform (0.46, 50), 0.0, 0.0],
                               dtype = np.float32)
        self.current_step_in_episode = 0
        return self.state, {}

    def step ( self, action ):
        indoor_allergen, outdoor_allergen, indoor_dust, energy, minute = self.state
        purifier, vent, vacuum = map (int, action [:3])  # first 3 actions used for actual env
        dt = 1
        step_energy = 0.0

        Qs, eta_s = (self.MECHANICAL_VENTILATION_FLOW_RATE,
                     self.MECHANICAL_SUPPLY_FILTER_EFFICIENCY) if vent == 1 else (0, 0)
        Qf, eta_f = (self.AIR_PURIFIER_FLOW_RATE, self.AIR_PURIFIER_FILTER_EFFICIENCY) if purifier == 1 else (0, 0)
        Qh = self.VACUUM_FLOW_RATE if vacuum == 1 else 0
        if vent == 1: step_energy += 400
        if purifier == 1: step_energy += 30
        if vacuum == 1: step_energy += 1000 / 60; indoor_dust *= 0.45

        Qn = self.NATURAL_VENTILATION_FLOW_RATE
        Ql = self.NATURAL_INFILTRATION_FLOW_RATE
        P = self.PARTICLE_FRACTION_IN_FLOW_PATH
        beta = self.DEPOSITION_RATE_OF_PARTICLE
        V = self.INTERIOR_VOLUME

        index = int ((minute // 30) % 48)
        E = self.EMISSION_SCHEDULE [index]

        dCi_dt = self.allergenConcentrationCalculation (E, outdoor_allergen, indoor_allergen,
                                                        Qs, Qn, Ql, Qh, eta_s, P, Qf, eta_f, beta, V)
        indoor_allergen = float (np.clip (indoor_allergen + dCi_dt, 0, 300))
        indoor_dust += beta * V
        energy = float (step_energy)
        reward = self.calculate_reward (indoor_allergen, energy)

        minute += dt
        if minute >= 1440: minute = 0

        self.state = np.array ([indoor_allergen, outdoor_allergen, indoor_dust, energy, minute], dtype = np.float32)
        self.current_step_in_episode += 1
        truncated = self.current_step_in_episode >= self.episode_length
        terminated = False
        return self.state, reward, terminated, truncated, {}

    def calculate_reward ( self, allergen, energy ):
        reward_allergen = 1 if allergen < 25 else -1
        reward_energy = -energy / 1000
        w_allergen, w_energy = 1, 3
        return reward_allergen * w_allergen + reward_energy * w_energy


# ==========================
# DQN predict
# ==========================
def dqn_predict(model, obs):
    """
    Given a model and observation, return the action vector from ACTION_LIST
    """
    ACTION_LIST = np.array ([
        [0, 0, 0],  # all off
        [1, 0, 0],  # purifier only
        [0, 1, 0],  # ventilation only
        [0, 0, 1],  # vacuum only
        [1, 1, 0],  # purifier + ventilation
        [1, 0, 1],  # purifier + vacuum
        [0, 1, 1],  # ventilation + vacuum
        [1, 1, 1],  # all on
    ], dtype = np.int64)

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_t)
    # choose the index of max Q-value
    action_idx = torch.argmax(q_values, dim=1).item()
    # map to actual MultiBinary action
    action = ACTION_LIST[action_idx]
    return action

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


@st.cache_resource
def load_ppo_custom ( actor_path, critic_path, obs_size, action_size ):
    actor = PPOActor (obs_size, action_size)
    critic = PPOCritic (obs_size)

    actor.load_state_dict (torch.load (actor_path, map_location = "cpu"))
    critic.load_state_dict (torch.load (critic_path, map_location = "cpu"))

    actor.eval ()
    critic.eval ()
    return actor, critic

def ppo_predict(actor, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = actor(obs_t).squeeze(0).numpy()
    # For MultiBinary action, sample 0/1 from probabilities
    action = (np.random.rand(len(probs)) < probs).astype(np.int64)
    return action


# ==========================
# Run evaluation
# ==========================
if run and selected_models:
    results = {}

    with st.spinner("Running models..."):
        for name in selected_models:
            cfg = MODELS[name]

            # Create environment
            env = AllergenEnvironment()  # same env for all
            obs, _ = env.reset()
            rewards = []

            # Load model
            for name in selected_models:
                cfg = MODELS [name]

                # Create environment
                env = AllergenEnvironment ()  # same env for all
                obs, _ = env.reset ()
                rewards = []

                # Load model
                if cfg ["algo"] == "PPO" and cfg ["path"].endswith (".pth"):
                    # custom PPO actor/critic
                    obs_size = env.observation_space.shape [0]
                    action_size = env.action_space.shape [0]
                    actor, critic = load_ppo_custom ("models/ppo_actor.pth", "models/ppo_critic.pth", obs_size,
                                                     action_size)
                    model_type = "custom_ppo"
                elif cfg ["type"] == "sb3":
                    model = load_sb3 (cfg ["algo"], cfg ["path"])
                    model_type = "sb3"
                else:  # DQN
                    obs_size = env.observation_space.shape [0]
                    model = load_dqn_pt (cfg ["path"], obs_size)
                    model_type = "dqn"

            # Run simulation
            for _ in range (steps):
                if model_type == "sb3":
                    action, _ = model.predict (obs, deterministic = True)
                elif model_type == "custom_ppo":
                    action = ppo_predict (actor, obs)  # returns MultiBinary action
                else:  # DQN
                    action = dqn_predict (model, obs)  # returns action vector from ACTION_LIST

                obs, reward, terminated, truncated, _ = env.step (action)
                rewards.append (reward)

                if terminated or truncated:
                    obs, _ = env.reset ()

            # Store cumulative rewards
            results[name] = np.cumsum(rewards)

    # ==========================
    # Plot cumulative reward
    # ==========================
    st.subheader("Cumulative Reward Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in results.items():
        ax.plot(curve, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ==========================
    # Summary
    # ==========================
    st.subheader("Final Rewards")
    for name, curve in results.items():
        st.write(f"{name}: **{curve[-1]:.2f}**")
else:
    st.info("Select models and click Run.")

