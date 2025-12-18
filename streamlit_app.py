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
        "path": "models/a2c.zip",
    },
    "SAC": {
        "type": "sb3",
        "algo": "SAC",
        "path": "models/sac.zip",
    },
    "PPO": {
        "type": "sb3",
        "algo": "PPO",
        "path": "models/ppo.zip",
    },
    "DQN (PyTorch)": {
        "type": "pt",
        "algo": "DQN",
        "path": "models/dqn.pt",
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

steps = st.sidebar.slider("Steps", 50, 2000, 500, 50)
run = st.sidebar.button("Run")

# =====================================================
# Environment
# =====================================================
def make_env():
    # Replace with your custom environment
    return gym.make("CartPole-v1")

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
