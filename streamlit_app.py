import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

from stable_baselines3 import A2C, SAC, PPO

# =====================================================
# Page setup
# =====================================================
st.set_page_config(page_title="RL Model Comparison", layout="wide")
st.title("RL Model Comparison Dashboard")

# =====================================================
# Loaders
# =====================================================
@st.cache_resource
def load_sb3(model_type, path):
    if model_type == "A2C":
        return A2C.load(path, device="cpu")
    if model_type == "SAC":
        return SAC.load(path, device="cpu")
    if model_type == "PPO":
        return PPO.load(path, device="cpu")
    raise ValueError("Unsupported SB3 model")

@st.cache_resource
def load_dqn_pt(path):
    model = torch.load(path, map_location="cpu")
    model.eval()
    return model

# =====================================================
# Model registry (EDIT PATHS)
# =====================================================
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

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("Simulation Settings")

selected_models = st.sidebar.multiselect(
    "Select models",
    MODELS.keys(),
    default=list(MODELS.keys())
)

steps = st.sidebar.slider("Steps", 60, 1440, 1440, 60)
run = st.sidebar.button("Run")

# =====================================================
# Environment
# =====================================================
def make_env():
    import random

    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np

    # =====================================================
    # 1.AllergenEnvironment
    # =====================================================
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

        def __init__ ( self ):
            super ().__init__ ()

            self.EMISSION_SCHEDULE = [
                0, 0, 0, 0,  # 0:00 - 2:00
                0, 0, 0, 0,  # 2:00 - 4:00
                0, 0, 0, 0,  # 4:00 - 6:00
                0, 0, 31, 30,  # 6:00 - 8:00
                990, 31, 0, 0,  # 8:00 - 10:00
                0, 0, 0, 31,  # 10:00 - 12:00
                30, 990, 31, 0,  # 12:00 - 14:00
                0, 0, 0, 0,  # 14:00 - 16:00
                0, 0, 0, 31,  # 16:00 - 18:00
                50, 31, 30, 990,  # 18:00 - 20:00
                31, 0, 0, 0,  # 20:00 - 22:00
                0, 0, 0, 0,  # 22:00 - 0:00
            ]

            # state = [indoor_allergen, outdoor_allergen, dust, energy, minute]
            self.observation_space = spaces.Box (
                low = np.array ([12, 12, 0.46, 0, 0], dtype = np.float32),
                high = np.array ([300, 135, 50, 50, 1439], dtype = np.float32),
                dtype = np.float32
            )

            self.action_space = spaces.MultiBinary (3)
            self.episode_length = 1440
            self.current_step_in_episode = 0

            self.state = None

        def allergenConcentrationCalculation (
                self,
                E, Co, Ci,
                Qs, Qn, Ql, Qh,
                eta_s, P,
                Qf, eta_f,
                beta, V
        ):
            outdoor_term = Co * (Qs * (1 - eta_s) + Qn + (Ql + Qh) * P)
            removal_term = Ci * (Qf * eta_f + beta * V + (Qs + Qn + Ql + Qh))
            return (E + outdoor_term - removal_term) / V

        def reset ( self, *, seed: int = None, options: dict = None ):
            super ().reset (seed = seed)
            if seed is not None:
                random.seed (seed)
                np.random.seed (seed)

            self.state = np.array ([
                25.0,  # indoor allergen
                random.uniform (12, 135),  # outdoor allergen
                random.uniform (0.46, 50),
                0.0,  # energy per step (your env uses step energy)
                0.0  # minute of day
            ], dtype = np.float32)

            self.current_step_in_episode = 0
            return self.state, {}

        def step ( self, action ):
            indoor_allergen, outdoor_allergen, indoor_dust, energy, minute = self.state

            purifier = int (action [0])
            vent = int (action [1])
            vacuum = int (action [2])

            dt = 1  # 1 minute per step
            step_energy = 0.0

            # Vent
            if vent == 1:
                Qs = self.MECHANICAL_VENTILATION_FLOW_RATE
                eta_s = self.MECHANICAL_SUPPLY_FILTER_EFFICIENCY
                step_energy += 400
            else:
                Qs = 0
                eta_s = 0

            # Purifier
            if purifier == 1:
                Qf = self.AIR_PURIFIER_FLOW_RATE
                eta_f = self.AIR_PURIFIER_FILTER_EFFICIENCY
                step_energy += 30
            else:
                Qf = 0
                eta_f = 0

            # Vacuum
            if vacuum == 1:
                Qh = self.VACUUM_FLOW_RATE
                step_energy += 1000 / 60
                indoor_dust *= 0.45
            else:
                Qh = 0

            Qn = self.NATURAL_VENTILATION_FLOW_RATE
            Ql = self.NATURAL_INFILTRATION_FLOW_RATE
            P = self.PARTICLE_FRACTION_IN_FLOW_PATH
            beta = self.DEPOSITION_RATE_OF_PARTICLE
            V = self.INTERIOR_VOLUME

            index = int ((minute // 30) % 48)
            E = self.EMISSION_SCHEDULE [index]

            dCi_dt = self.allergenConcentrationCalculation (
                E = E, Co = outdoor_allergen, Ci = indoor_allergen,
                Qs = Qs, Qn = Qn, Ql = Ql, Qh = Qh,
                eta_s = eta_s, P = P,
                Qf = Qf, eta_f = eta_f,
                beta = beta, V = V
            )

            # update indoor allergen (IMPORTANT: clip to avoid explosion)
            indoor_allergen = float (np.clip (indoor_allergen + dCi_dt, 0, 300))
            indoor_dust += beta * V

            energy = float (step_energy)  # per step energy
            reward = self.calculate_reward (indoor_allergen, energy)

            # minute update
            minute += dt
            if minute >= 1440:
                minute = 0

            self.state = np.array ([
                indoor_allergen, outdoor_allergen, indoor_dust, energy, minute
            ], dtype = np.float32)

            self.current_step_in_episode += 1
            truncated = self.current_step_in_episode >= self.episode_length
            terminated = False
            return self.state, reward, terminated, truncated, {}

        def calculate_reward ( self, allergen, energy ):
            reward_allergen = 1 if allergen < 25 else -1
            reward_energy = (-energy / 1000)

            # your weights
            w_allergen = 1
            w_energy = 3

            return reward_allergen * w_allergen + reward_energy * w_energy

        def render ( self, mode = "human" ):
            t = self.convert_to_time (self.state [4])
            print (f"Time: {t} | Indoor: {self.state [0]:.2f} | Outdoor: {self.state [1]:.2f} "
                   f"| Dust: {self.state [2]:.2f} | Energy: {self.state [3]:.2f}")

        def convert_to_time ( self, minutes ):
            hours = int (minutes // 60)
            mins = int (minutes % 60)
            return f"{hours:02}:{mins:02}"


# =====================================================
# Action for DQN (.pt)
# =====================================================
def dqn_predict(model, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_t)
    return torch.argmax(q_values, dim=1).item()

# =====================================================
# Run evaluation
# =====================================================
if run and selected_models:

    env = make_env()
    results = {}

    with st.spinner("Running models..."):
        for name in selected_models:
            cfg = MODELS[name]
            obs, _ = env.reset()
            rewards = []

            # Load model
            if cfg["type"] == "sb3":
                model = load_sb3(cfg["algo"], cfg["path"])
            else:
                model = load_dqn_pt(cfg["path"])

            for _ in range(steps):
                if cfg["type"] == "sb3":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = dqn_predict(model, obs)

                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)

                if terminated or truncated:
                    obs, _ = env.reset()

            results[name] = np.cumsum(rewards)

    # =================================================
    # Plot
    # =================================================
    st.subheader("Cumulative Reward Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in results.items():
        ax.plot(curve, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # =================================================
    # Summary
    # =================================================
    st.subheader("Final Rewards")
    for name, curve in results.items():
        st.write(f"{name}: **{curve[-1]:.2f}**")

else:
    st.info("Select models and click Run.")
