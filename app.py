
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces
import joblib
import cloudpickle
import sys
from PIL import Image
from io import StringIO
#import wbgapi as wb

# ==================== Dataset Preparation ====================



@st.cache_data
def fetch_and_prepare_data():
    """Fetch and prepare the dataset from the World Bank API."""
    st.write("### 🌍 Dataset Preparation")
    st.write("""
    The dataset is fetched from the **World Bank API** using the `wbgapi` library. It includes key indicators related to CO2 emissions, economic growth, energy use, and environmental factors. The data is cleaned, reshaped, and preprocessed for use in the DRL model.
    """)
   
    # Define indicators with their human-readable names
    indicators = {
        'EN.GHG.CO2.MT.CE.AR5': 'CO2_emissions',  # CO2 emissions (kt)
        'NY.GDP.MKTP.KD.ZG': 'GDP_growth',  # GDP growth (annual %)
        'EG.USE.PCAP.KG.OE': 'Energy_use_per_capita',  # Energy use per capita (kg of oil equivalent)
        'EG.FEC.RNEW.ZS': 'Renewable_energy_share',  # Renewable energy share of total energy consumption (%)
        'SP.POP.GROW': 'Population_growth',  # Population growth (annual %)
        'EG.ELC.COAL.ZS': 'Coal_energy_share',  # Coal energy share of electricity production (%)
        'EG.ELC.RNWX.ZS': 'Renewable_electricity_share',  # Renewable electricity share of total electricity production (%)
        'SP.URB.GROW': 'Urban_population_growth',  # Urban population growth (annual %)
        'AG.LND.FRST.ZS': 'Forest_area',  # Forest area (% of land area)
        'NV.IND.TOTL.ZS': 'Industrial_value_added',  # Industrial value added (% of GDP)
        'EG.ELC.ACCS.ZS': 'Access_to_electricity',  # Access to electricity (% of population)
        'EG.USE.COMM.FO.ZS': 'Fossil_fuel_energy_consumption',  # Fossil fuel energy consumption (% of total)
        'EN.ATM.PM25.MC.M3': 'Air_pollution_PM2.5',  # PM2.5 air pollution (mean annual exposure)
        'GB.XPD.RSDV.GD.ZS': 'R&D_expenditure',  # Research and development expenditure (% of GDP)
    }

    # Fetch data for all countries and the world
    st.write("Fetching data from the World Bank API...")
    data=pd.read_csv('data.csv')

    st.write("Dataset preparation complete! 🎉")
    return data

# ==================== Environment Class ====================
class CarbonEmissionEnv(gym.Env):
    def __init__(self, data, columns_to_normalize):
        super(CarbonEmissionEnv, self).__init__()
        self.data = data
        self.columns_to_normalize = columns_to_normalize
        self.current_step = 0
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.columns_to_normalize),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.state = self.data.iloc[self.current_step][self.columns_to_normalize].values

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self.data.iloc[self.current_step][self.columns_to_normalize].values
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.state if done else self.data.iloc[self.current_step][self.columns_to_normalize].values
       
        reward = self._calculate_reward(action)
        info = self.get_action_details(action) if done else {}
       
        return next_state, reward, done, False, info

    def _calculate_reward(self, action):
        co2, gdp, _, renewable, *_, air_pollution = self.state
        return (0.5 * (1 - co2) + 0.2 * gdp + 0.2 * renewable + 0.1 * (1 - air_pollution))

    def get_action_details(self, action):
        actions_info = {
            0: {'action': 'Reduce CO2 emissions', 'policy': 'Energy transition initiatives', 'reward': 0.5},
            1: {'action': 'Boost Green GDP', 'policy': 'Sustainable economic practices', 'reward': 0.3},
            2: {'action': 'Promote Renewables', 'policy': 'Renewable energy investments', 'reward': 0.4},
            3: {'action': 'Improve Air Quality', 'policy': 'Stricter emissions controls', 'reward': 0.2},
            4: {'action': 'Green R&D', 'policy': 'Research in clean tech', 'reward': 0.1}
        }
        return actions_info.get(action, {})

# ==================== Streamlit App ====================
st.set_page_config(page_title="CO2 Optimization DRL", layout="wide", page_icon="🌍")

image1 = Image.open("mypiclogo.png")
st.image(image1) 
st.write("benjaminukaimo@gmail.com")
st.write("+2347067193071")
st.write("My name is Uka Benjamin. I'm a data scientist with good experiences")



# Apply custom CSS for enhanced background and lighter sidebar color
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,brown,pink, gray, #f5f7fa);
    color: #333;
}
[data-testid="stSidebar"] {
    background: linear-gradient(skyblue, blue) !important;
    color: white;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}
body {
    font-family: "Source Sans Pro", sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #2c3e50;
}
.stMarkdown p {
    color: #333;
}
.stDataFrame {
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.stProgress > div > div > div {
    background-color: #4CAF50;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("🌱 Deep Reinforcement Learning for CO2 Emission Optimization")
st.write("""
Welcome to the **CO2 Emission Optimization Dashboard**! This application showcases a **Deep Reinforcement Learning (DRL)** model designed to optimize CO2 emissions by balancing economic growth, renewable energy adoption, and environmental sustainability. The model is trained on real-world data from the **World Bank API** and provides actionable insights for policymakers and researchers.
""")

# Load and preprocess data
# Load the dataset
data = pd.read_csv('co2_emission_reduction.csv')

# Handle missing values
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
columns_to_normalize = [
    'CO2_emissions', 'GDP_growth', 'Energy_use_per_capita', 'Renewable_energy_share',
    'Population_growth', 'Coal_energy_share', 'Renewable_electricity_share',
    'Urban_population_growth', 'Forest_area', 'Industrial_value_added',
    'Access_to_electricity', 'Fossil_fuel_energy_consumption', 'Air_pollution_PM2.5',
    'R&D_expenditure'
]
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
print(data)



# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Insights", "🏋️ Model Training", "📈 Model Evaluation", "🔮 Live Predictions", "About"])

# Data Insights Tab
with tab1:
    st.header("📊 Data Insights")
    st.write("""
    The dataset includes **14 key indicators** related to CO2 emissions, economic growth, energy use, and environmental factors. Below is a preview of the processed data:
    """)
    st.dataframe(data.head(), use_container_width=True)

    st.write("### Key Features Distribution")
    fig, ax = plt.subplots(figsize=(10, 20))
    data[columns_to_normalize[:]].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Distribution of Key Features")
    st.pyplot(fig)

# Model Training Tab
with tab2:
    st.header("🏋️ Model Training")
    st.write("""
    The DRL model is trained using **Proximal Policy Optimization (PPO)**, a state-of-the-art reinforcement learning algorithm. The model learns to optimize CO2 emissions by balancing multiple objectives, including economic growth, renewable energy adoption, and air quality improvement.
    """)
   
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress... This may take several minutes"):
            env = make_vec_env(lambda: CarbonEmissionEnv(data, columns_to_normalize), n_envs=1)
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                learning_rate=0.0003,
                batch_size=64,
                gamma=0.99,
                device="cpu"
            )
            model.learn(total_timesteps=50000)
            model.save("carbon_model")
            st.session_state.model = model
            st.success("Model training complete! 🎉")

# Model Evaluation Tab
with tab3:
    st.header("📈 Model Evaluation")
    st.write("""
    Evaluate the performance of the trained DRL model. This section visualizes the model's actions and rewards over time, providing insights into its decision-making process.
    """)
   
    if 'model' in st.session_state:
        st.write("### Evaluation Results")
        env = make_vec_env(lambda: CarbonEmissionEnv(data, columns_to_normalize), n_envs=1)
        obs = env.reset()
        actions = []
        rewards = []
       
        for _ in range(1000):
            action, _ = st.session_state.model.predict(obs)
            obs, reward, done, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if done:
                break

        # Plot actions and rewards
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(actions, label="Actions", color='blue')
        ax1.set_title("Policy Actions Over Time")
        ax1.set_ylabel("Action")
        ax1.legend()

        ax2.plot(rewards, label="Rewards", color='orange')
        ax2.set_title("Rewards Over Time")
        ax2.set_ylabel("Reward")
        ax2.legend()

        st.pyplot(fig)
        st.write(f"**Average Reward:** {np.mean(rewards):.2f}")
    else:
        st.warning("Please train the model first to evaluate its performance.")

# Live Predictions Tab
with tab4:
    st.header("🔮 Live Predictions")
    st.write("""
    Use the trained model to simulate real-world scenarios and explore policy recommendations for reducing CO2 emissions. The model provides actionable insights based on the current state of the environment.
    """)
    if st.button("▶️ Start Prediction"):
        st.write("Live predictions will be displayed here...")

with tab5:
    st.header("About   ℹ ")
    st.write("""
    This application is developed by **[Your Name]**, a data scientist specializing in **AI for Sustainability**. The goal is to demonstrate how advanced machine learning techniques can be applied to real-world environmental challenges.
    """)
    st.write("**Technologies Used:**")
    st.write("- Python, Streamlit, Stable-Baselines3")
    st.write("- World Bank API for data collection")
    st.write("- Proximal Policy Optimization (PPO) for DRL")



# Footer
st.markdown("---")
st.write("Developed with ❤️ by **[Uka Benjamin Imo]** | [GitHub](https://github.com/uka-ben) | [LinkedIn](https://www.linkedin.com/in/benjamin-uka-imo)")
