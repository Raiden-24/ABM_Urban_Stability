import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import copy

from model.urban_model import UrbanStabilityModel
from config import CONFIG

st.set_page_config(
    page_title="Urban Stability Dashboard", 
    page_icon="🏙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Initialize Session State
# -----------------------------
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None
if 'run_params' not in st.session_state:
    st.session_state.run_params = None

# -----------------------------
# App Header
# -----------------------------
st.title("🏙️ Urban Stability ABM Dashboard")
st.markdown("""
Welcome to the **Urban Stability Agent-Based Model** dashboard. 
Adjust the policy and environmental parameters in the sidebar, then run the simulation to analyze the impact on stability (USI), trust, and resource distribution.
""")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Simulation Parameters")

with st.sidebar.expander("🌍 Environment & Shocks", expanded=True):
    total_supply = st.slider("Initial Resource Supply", 500, 5000, CONFIG["total_supply"], step=100)
    shock_magnitude = st.slider("Shock Magnitude", 0.0, 1.0, CONFIG["shock"]["magnitude"], step=0.05)

with st.sidebar.expander("📜 Policy Controls", expanded=True):
    enable_subsidy = st.checkbox("Enable Subsidy", False, help="Provides targeted resource assistance based on need.")
    enable_cap = st.checkbox("Enable Consumption Cap", False, help="Strict upper limit on individual resource consumption.")
    enable_pricing = st.checkbox("Enable Pricing Multiplier", False, help="Dynamic pricing based on overall demand.")

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 Run Simulation", use_container_width=True, type="primary")

# -----------------------------
# Build Config & Run Sim
# -----------------------------
if run_clicked:
    config = copy.deepcopy(CONFIG)
    config["total_supply"] = total_supply
    config["shock"]["magnitude"] = shock_magnitude
    config["policy"]["subsidy"]["enabled"] = enable_subsidy
    config["policy"]["consumption_cap"]["enabled"] = enable_cap
    config["policy"]["pricing_multiplier"]["enabled"] = enable_pricing

    with st.spinner("Simulating urban dynamics..."):
        model = UrbanStabilityModel(config=config)

        while model.running and model.current_step < config["num_steps"]:
            model.step()

        st.session_state.sim_data = model.datacollector.get_model_vars_dataframe()
        st.session_state.run_params = config

# -----------------------------
# Output View
# -----------------------------
if st.session_state.sim_data is not None:
    data = st.session_state.sim_data
    
    # Calculate key metrics
    start_usi = data["USI"].iloc[0]
    final_usi = data["USI"].iloc[-1]
    
    start_trust = data["Average_Trust"].iloc[0]
    final_trust = data["Average_Trust"].iloc[-1]
    
    start_gini = data["Gini"].iloc[0]
    final_gini = data["Gini"].iloc[-1]
    
    peak_demand = data["Total_Demand"].max()
    
    # KPIs
    st.subheader("📊 Simulation Summary")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Final USI (Stability)", f"{final_usi:.3f}", f"{final_usi - start_usi:.3f}")
    with cols[1]:
        st.metric("Final Avg Trust", f"{final_trust:.3f}", f"{final_trust - start_trust:.3f}")
    with cols[2]:
        st.metric("Final Gini (Inequality)", f"{final_gini:.3f}", f"{final_gini - start_gini:.3f}", delta_color="inverse")
    with cols[3]:
        st.metric("Peak Demand", f"{peak_demand:.0f}")

    st.markdown("---")

    # Tabs for detailed charts
    tab1, tab2, tab3 = st.tabs(["Main Indicators", "Stability Components", "Supply & Demand"])

    with tab1:
        st.markdown("### Core Metrics: USI & Trust")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Urban Stability Index (USI)**")
            st.line_chart(data["USI"], color="#1f77b4")
        with c2:
            st.markdown("**Average Trust**")
            st.line_chart(data["Average_Trust"], color="#2ca02c")

    with tab2:
        st.markdown("### Stability Components")
        st.markdown("Breakdown of Resource Sufficiency (S_R), Inequality Stability (S_I), and Organizational (S_O) stability.")
        st.line_chart(data[["S_R", "S_I", "S_O"]])

    with tab3:
        st.markdown("### Resource Dynamics")
        st.markdown("Comparison of Total Demand vs System Supply over time.")
        st.line_chart(data[["Total_Supply", "Total_Demand"]])

    # Download button
    st.sidebar.markdown("---")
    csv = data.to_csv().encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Results Data",
        data=csv,
        file_name='simulation_results.csv',
        mime='text/csv',
        use_container_width=True
    )
    
else:
    st.info("👈 Adjust parameters and click **Run Simulation** to see results.")