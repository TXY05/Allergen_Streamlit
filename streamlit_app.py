import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import random
import time
from datetime import datetime
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
    if isinstance (checkpoint, nn.Module):
        model = checkpoint
    else:
        n_actions = None
        for k, v in checkpoint.items ():
            if "net.4.weight" in k:
                n_actions = v.shape [0]
                break
        if n_actions is None:
            raise ValueError ("Cannot detect output size from state dict.")
        model = DQNNetwork (obs_size, n_actions)
        new_state_dict = {k.replace ("module.", ""): v for k, v in checkpoint.items ()}
        model.load_state_dict (new_state_dict)
    model.eval ()
    return model


# ==========================
# Custom PPO
# ==========================
class PPOActor (nn.Module):
    def __init__ ( self, state_dim, action_dim ):
        super ().__init__ ()
        self.fc1 = nn.Linear (state_dim, 128)
        self.fc2 = nn.Linear (128, 128)
        self.fc3 = nn.Linear (128, action_dim)

    def forward ( self, x ):
        x = torch.relu (self.fc1 (x))
        x = torch.relu (self.fc2 (x))
        return torch.sigmoid (self.fc3 (x))


class PPOCritic (nn.Module):
    def __init__ ( self, state_dim ):
        super ().__init__ ()
        self.fc1 = nn.Linear (state_dim, 128)
        self.fc2 = nn.Linear (128, 128)
        self.fc3 = nn.Linear (128, 1)

    def forward ( self, x ):
        x = torch.relu (self.fc1 (x))
        x = torch.relu (self.fc2 (x))
        return self.fc3 (x)


@st.cache_resource
def load_ppo_custom ( actor_path, critic_path, obs_size, action_size ):
    actor = PPOActor (obs_size, action_size)
    critic = PPOCritic (obs_size)
    actor.load_state_dict (torch.load (actor_path, map_location = "cpu"))
    critic.load_state_dict (torch.load (critic_path, map_location = "cpu"))
    actor.eval ()
    critic.eval ()
    return actor, critic


def ppo_predict ( actor, obs ):
    obs_t = torch.tensor (obs, dtype = torch.float32).unsqueeze (0)
    with torch.no_grad ():
        probs = actor (obs_t).squeeze (0).numpy ()
    action = (np.random.rand (len (probs)) < probs).astype (int)
    return action


# ==========================
# DQN predict
# ==========================
def dqn_predict ( model, obs ):
    ACTION_LIST = np.array ([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype = np.int64)
    obs_t = torch.tensor (obs, dtype = torch.float32).unsqueeze (0)
    with torch.no_grad ():
        q_values = model (obs_t)
    action_idx = torch.argmax (q_values, dim = 1).item ()
    return ACTION_LIST [action_idx]


# ==========================
# Model registry
# ==========================
MODELS = {
    "A2C": {"type": "sb3", "algo": "A2C", "path": "models/best_a2c_model.zip"},
    "SAC": {"type": "sb3", "algo": "SAC", "path": "models/best_sac_model.zip"},
    "PPO ": {"type": "custom_ppo", "actor_path": "models/ppo_actor.pth",
                     "critic_path": "models/ppo_critic.pth"},
    "DQN ": {"type": "pt", "algo": "DQN", "path": "models/dqn_allergen_model.pt"},
}

# ==========================
# Sidebar
# ==========================
st.sidebar.header ("Simulation Settings")
live_mode = st.sidebar.checkbox ("Live Mode (real-time updates)", value = False)
selected_models = st.sidebar.multiselect ("Select models", MODELS.keys (), default = list (MODELS.keys ()))

if live_mode:
    if len (selected_models) > 1:
        st.sidebar.warning ("⚠️ Live mode only supports one model at a time")
        selected_models = [selected_models [0]]
    update_interval = st.sidebar.slider ("Update interval (steps)", 1, 20, 5)

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

    def __init__ ( self, agent_type = "sb3", dqn_action_space = False ):
        super ().__init__ ()
        self.observation_space = spaces.Box (
            low = np.array ([0, 12, 0.46, 0, 0], dtype = np.float32),
            high = np.array ([300, 135, 50, 50, 1439], dtype = np.float32),
            dtype = np.float32
        )
        if agent_type == "SAC":
            self.action_space = spaces.Box (low = 0, high = 1, shape = (3,), dtype = np.float32)
        elif dqn_action_space:
            self.action_space = spaces.MultiBinary (8)
        else:
            self.action_space = spaces.MultiBinary (3)

        self.episode_length = 1440
        self.current_step_in_episode = 0
        self.state = None

        self.EMISSION_SCHEDULE = [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 31, 30,
            990, 31, 0, 0,
            0, 0, 0, 31,
            30, 990, 31, 0,
            0, 0, 0, 0,
            0, 0, 0, 31,
            50, 31, 30, 990,
            31, 0, 0, 0,
            0, 0, 0, 0,
        ]

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

        if isinstance (action, np.ndarray) and action.dtype == np.float32:
            action = (action > 0.5).astype (int)

        purifier, vent, vacuum = map (int, action [:3])
        dt = 1
        step_energy = 0.0

        Qs, eta_s = (
            (self.MECHANICAL_VENTILATION_FLOW_RATE, self.MECHANICAL_SUPPLY_FILTER_EFFICIENCY) if vent == 1 else (0, 0))
        Qf, eta_f = ((self.AIR_PURIFIER_FLOW_RATE, self.AIR_PURIFIER_FILTER_EFFICIENCY) if purifier == 1 else (0, 0))
        Qh = self.VACUUM_FLOW_RATE if vacuum == 1 else 0
        if vent == 1: step_energy += 400
        if purifier == 1: step_energy += 30
        if vacuum == 1:
            step_energy += 1000 / 60
            indoor_dust *= 0.45

        Qn = self.NATURAL_VENTILATION_FLOW_RATE
        Ql = self.NATURAL_INFILTRATION_FLOW_RATE
        P = self.PARTICLE_FRACTION_IN_FLOW_PATH
        beta = self.DEPOSITION_RATE_OF_PARTICLE
        V = self.INTERIOR_VOLUME

        index = int ((minute // 30) % 48)
        E = self.EMISSION_SCHEDULE [index]

        dCi_dt = self.allergenConcentrationCalculation (E, outdoor_allergen, indoor_allergen, Qs, Qn, Ql, Qh, eta_s, P,
                                                        Qf, eta_f, beta, V)
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
# Live visualization function
# ==========================
def run_live_simulation ( model_name, cfg, steps, update_interval = 5 ):
    """Run simulation with live updates"""

    algo = cfg.get ("algo", "")
    if algo == "SAC":
        env_type = "SAC"
    elif algo == "A2C":
        env_type = "A2C"
    else:
        env_type = cfg ["type"]

    env = AllergenEnvironment (agent_type = env_type)

    obs, _ = env.reset ()

    if cfg ["type"] == "custom_ppo":
        obs_size = env.observation_space.shape [0]
        action_size = env.action_space.shape [0]
        actor, critic = load_ppo_custom (cfg ["actor_path"], cfg ["critic_path"], obs_size, action_size)
        model_type = "custom_ppo"
    elif cfg.get ("algo", "") == "SAC" or cfg ["type"] == "sb3":
        model = load_sb3 (cfg ["algo"], cfg ["path"])
        model_type = "sb3"
    else:
        obs_size = env.observation_space.shape [0]
        model = load_dqn_pt (cfg ["path"], obs_size)
        model_type = "dqn"

    col1, col2 = st.columns ([2, 1])

    with col1:
        chart_placeholder = st.empty ()

    with col2:
        metrics_placeholder = st.empty ()
        icon_placeholder = st.empty ()

    allergen_history = []
    energy_history = []
    reward_history = []
    action_history = []

    for step in range (steps):
        if model_type == "sb3":
            action, _ = model.predict (obs, deterministic = True)
            if cfg.get ("algo", "") == "SAC":
                action = (action > 0.5).astype (int)
        elif model_type == "custom_ppo":
            action = ppo_predict (actor, obs)
        else:
            action = dqn_predict (model, obs)

        obs, reward, terminated, truncated, _ = env.step (action)

        allergen_history.append (obs [0])
        energy_history.append (obs [3])
        reward_history.append (reward)
        action_history.append (action [:3])

        if step % update_interval == 0 or step == steps - 1:
            with chart_placeholder.container ():
                fig, (ax1, ax2, ax3) = plt.subplots (3, 1, figsize = (8, 9))

                ax1.plot (allergen_history, color = '#ff6b6b', linewidth = 2)
                ax1.axhline (y = 25, color = 'green', linestyle = '--', label = 'Target (25 µg/m³)')
                ax1.set_ylabel ('Indoor Allergen (µg/m³)', fontsize = 10)
                ax1.set_title (f'{model_name} - Live Simulation', fontsize = 12, fontweight = 'bold')
                ax1.legend (loc = 'upper right')
                ax1.grid (True, alpha = 0.3)
                ax1.set_xlim (0, steps)

                ax2.plot (np.cumsum (reward_history), color = '#4ecdc4', linewidth = 2)
                ax2.set_ylabel ('Cumulative Reward', fontsize = 10)
                ax2.grid (True, alpha = 0.3)
                ax2.set_xlim (0, steps)

                cumulative_energy = np.cumsum (energy_history) / 1000  # Convert to kWh
                ax3.plot (cumulative_energy, color = '#f39c12', linewidth = 2)
                ax3.set_xlabel ('Step', fontsize = 10)
                ax3.set_ylabel ('Cumulative Energy (kWh)', fontsize = 10)
                ax3.grid (True, alpha = 0.3)
                ax3.set_xlim (0, steps)

                plt.tight_layout ()
                st.pyplot (fig)
                plt.close ()

                st.subheader ("Device Operating Status")
                fig4, (ax_p, ax_v, ax_vac) = plt.subplots (3, 1, figsize = (10, 8), sharex = True)

                # Helper to plot status
                def plot_device_status ( ax, history, index, name, color ):
                    # Extract binary signal for specific device from action_history
                    status = [1 if a [index] == 1 else 0 for a in action_history]
                    steps = np.arange (len (status))

                    # Plot 'X' only where status is 1 (ON)
                    on_steps = [i for i, val in enumerate (status) if val == 1]
                    on_vals = [1] * len (on_steps)

                    ax.scatter (on_steps, on_vals, marker = 'x', color = color, label = f'{name} ON')
                    ax.set_yticks ([0, 1])
                    ax.set_yticklabels (['OFF', 'ON'])
                    ax.set_ylim (-0.2, 1.2)
                    ax.set_ylabel (name)
                    ax.grid (True, alpha = 0.3)

                # Plotting the three devices
                plot_device_status (ax_p, action_history, 0, "Purifier", "#3498db")
                plot_device_status (ax_v, action_history, 1, "Vent", "#e67e22")
                plot_device_status (ax_vac, action_history, 2, "Vacuum", "#9b59b6")

                ax_vac.set_xlabel ("Step")
                plt.tight_layout ()
                st.pyplot (fig4)

            with metrics_placeholder.container ():
                current_time = datetime.now ().strftime ("%Y-%m-%d %H:%M:%S")
                minute = int (obs [4])
                hour = minute // 60
                minute_of_hour = minute % 60

                purifier, vent, vacuum = action_history [-1]

                st.markdown (f"""
                ### 📊 Current Status
                **Time:** `{current_time}`  
                **Simulation Hour:** `{hour:02d}:{minute_of_hour:02d}`  
                **Step:** `{step + 1}/{steps}`

                ---

                **Indoor Allergen:** `{obs [0]:.2f}` µg/m³  
                **Outdoor Allergen:** `{obs [1]:.2f}` µg/m³  
                **Indoor Dust:** `{obs [2]:.2f}` µg/m³  
                **Energy (step):** `{obs [3]:.2f}` W  
                **Cumulative Energy:** `{sum (energy_history) / 1000:.3f}` kWh  
                **Reward (step):** `{reward:.3f}`  
                **Cumulative Reward:** `{sum (reward_history):.1f}`

                ---

                ### 🎮 Active Controls
                """)

                col_a, col_b, col_c = st.columns (3)
                with col_a:
                    if purifier:
                        st.markdown ("💨 **Purifier**  \n✅ ON")
                    else:
                        st.markdown ("💨 **Purifier**  \n❌ OFF")
                with col_b:
                    if vent:
                        st.markdown ("🌬️ **Vent**  \n✅ ON")
                    else:
                        st.markdown ("🌬️ **Vent**  \n❌ OFF")
                with col_c:
                    if vacuum:
                        st.markdown ("🧹 **Vacuum**  \n✅ ON")
                    else:
                        st.markdown ("🧹 **Vacuum**  \n❌ OFF")

            with icon_placeholder.container ():
                if 6 <= hour < 18:
                    st.markdown ("<div style='text-align: center; font-size: 60px;'>☀️</div>", unsafe_allow_html = True)
                    st.markdown ("<div style='text-align: center;'>Daytime</div>", unsafe_allow_html = True)
                else:
                    st.markdown ("<div style='text-align: center; font-size: 60px;'>🌙</div>", unsafe_allow_html = True)
                    st.markdown ("<div style='text-align: center;'>Nighttime</div>", unsafe_allow_html = True)



            time.sleep (0.1)

        if terminated or truncated:
            obs, _ = env.reset ()

    st.success (f"✅ Simulation complete! Final cumulative reward: {sum (reward_history):.1f}")


# ==========================
# Run evaluation
# ==========================
def render_env_info_panel(obs,action,reward,energy_history,reward_history,step,steps,mean_indoor,mean_outdoor,mean_dust,mean_energy,mean_reward):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    minute = int(obs[4])
    hour = minute // 60
    minute_of_hour = minute % 60

    purifier, vent, vacuum = action

    st.markdown(f"""
    **Mean Indoor Allergen:** `{mean_indoor:.2f}` µg/m³  
    **Mean Outdoor Allergen:** `{mean_outdoor:.2f}` µg/m³  
    **Mean Dust:** `{mean_dust:.2f}` µg/m³  
    **Mean Energy (step):** `{mean_energy:.2f}` W  
    **Cumulative Energy:** `{sum(energy_history) / 1000:.3f}` kWh  
    **Reward (step):** `{reward:.3f}`  
    **Cumulative Reward:** `{reward_history[-1]:.1f}`  
    **Mean Reward (per step):** `{mean_reward:.3f}`
    """)

if run and selected_models:
    if live_mode:
        name = selected_models [0]
        cfg = MODELS [name]
        st.subheader (f"🔴 Live Simulation: {name}")
        run_live_simulation (name, cfg, steps, update_interval)
    else:
        mean_rewards = {}
        indoor_allergen_history = {}
        outdoor_allergen_history = {}
        energy_history = {}
        dust_history = {}
        results = {}
        action_history = {}

        common_outdoor = random.uniform (12, 135)

        with st.spinner ("Running models..."):
            for name in selected_models:
                cfg = MODELS [name]

                env_type = "SAC" if cfg.get ("algo", "") == "SAC" else cfg ["type"]
                env = AllergenEnvironment (agent_type = env_type)
                obs, _ = env.reset()
                obs[1] = common_outdoor  # set the first step of outdoor allergen

                rewards = []

                if cfg ["type"] == "custom_ppo":
                    obs_size = env.observation_space.shape [0]
                    action_size = env.action_space.shape [0]
                    actor, critic = load_ppo_custom (cfg ["actor_path"], cfg ["critic_path"], obs_size, action_size)
                    model_type = "custom_ppo"
                elif cfg.get ("algo", "") == "SAC" or cfg ["type"] == "sb3":
                    model = load_sb3 (cfg ["algo"], cfg ["path"])
                    model_type = "sb3"
                else:
                    obs_size = env.observation_space.shape [0]
                    model = load_dqn_pt (cfg ["path"], obs_size)
                    model_type = "dqn"

                last_obs = None
                last_action = None
                last_reward = None

                for step_idx in range (steps):
                    if model_type == "sb3":
                        action, _ = model.predict (obs, deterministic = True)
                        if cfg.get ("algo", "") == "SAC":
                            action = (action > 0.5).astype (int)
                    elif model_type == "custom_ppo":
                        action = ppo_predict (actor, obs)
                    else:
                        action = dqn_predict (model, obs)

                    obs [1] = common_outdoor
                    obs, reward, terminated, truncated, _ = env.step (action)

                    rewards.append (reward)
                    indoor_allergen_history.setdefault (name, []).append (obs [0])
                    outdoor_allergen_history.setdefault (name, []).append (obs [1])
                    dust_history.setdefault (name, []).append (obs [2])
                    action_history.setdefault (name, []).append (action [:3])
                    energy_history.setdefault (name, []).append (obs [3])

                    last_obs = obs
                    last_action = action [:3]
                    last_reward = reward

                    if terminated or truncated:
                        obs, _ = env.reset ()

                results [name] = np.cumsum (rewards)
                mean_rewards[name] = float(np.mean(rewards))

        st.subheader ("Cumulative Reward Comparison")
        fig, ax = plt.subplots (figsize = (10, 5))
        for name, curve in results.items ():
            ax.plot (curve, label = name)
        ax.set_xlabel ("Step")
        ax.set_ylabel ("Cumulative Reward")
        ax.legend ()
        ax.grid (True)
        st.pyplot (fig)

        st.subheader ("Allergen Levels Over Time")
        fig2, ax2 = plt.subplots (figsize = (12, 6))

        # Plot Indoor Allergen curves
        for name, curve in indoor_allergen_history.items ():
            ax2.plot (curve, label = f"Indoor: {name}")

        # Plot Outdoor Allergen curves
        for name, curve in outdoor_allergen_history.items ():
            # Using a dashed line to visually distinguish outdoor from indoor
            ax2.plot (curve, label = f"Outdoor: {name}", linestyle = '--')

        # Add Target line
        ax2.axhline (y = 25, color = 'green', linestyle = ':', alpha = 0.7, label = 'Target (25 µg/m³)')

        # Formatting the single graph
        ax2.set_xlabel ("Step")
        ax2.set_ylabel ("Allergen Concentration (µg/m³)")
        ax2.set_title ("Indoor & Outdoor Allergen Concentration")
        ax2.legend (loc = 'upper right', bbox_to_anchor = (1.15, 1))  # Moved legend slightly outside if it gets crowded
        ax2.grid (True, linestyle = '--', alpha = 0.6)

        plt.tight_layout ()
        st.pyplot (fig2)

        st.subheader ("Cumulative Energy Consumption Over Time")
        fig3, ax3 = plt.subplots (figsize = (10, 5))
        for name, curve in energy_history.items ():
            cumulative_energy = np.cumsum (curve) / 1000
            ax3.plot (cumulative_energy, label = name)
        ax3.set_xlabel ("Step")
        ax3.set_ylabel ("Cumulative Energy (kWh)")
        ax3.legend ()
        ax3.grid (True)
        st.pyplot (fig3)

        st.subheader ("🧠 Final Environment Status")

        for name in selected_models:
            st.markdown (f"## 🔹 {name}")

            mean_indoor = np.mean (indoor_allergen_history [name])
            mean_outdoor = np.mean (outdoor_allergen_history [name])
            mean_dust = np.mean (dust_history [name])
            mean_energy = np.mean (energy_history [name])

            render_env_info_panel (
                obs = last_obs,
                action = last_action,
                reward = last_reward,
                energy_history = energy_history [name],
                reward_history = results [name],
                step = len (results [name]),
                steps = steps,
                mean_indoor = mean_indoor,
                mean_outdoor = mean_outdoor,
                mean_dust = mean_dust,
                mean_energy = mean_energy,
                mean_reward = mean_rewards[name]
            )

            st.markdown ("---")

else:
    st.info ("Select models and click Run.")