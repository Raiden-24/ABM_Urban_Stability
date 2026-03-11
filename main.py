from model.urban_model import UrbanStabilityModel
import matplotlib.pyplot as plt

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------

model = UrbanStabilityModel()

while model.running and model.current_step < model.config["num_steps"]:
    model.step()

    if model.current_step % 5 == 0:
        print(
            f"Step {model.current_step}: "
            f"Gini = {model.current_gini:.4f}, "
            f"USI = {model.current_usi:.4f}"
        )

# --------------------------------------------------
# COLLECT DATA
# --------------------------------------------------

data = model.datacollector.get_model_vars_dataframe()

# --------------------------------------------------
# PLOT 1 : Average Trust
# --------------------------------------------------

plt.figure()
plt.plot(data["Average_Trust"])
plt.title("Average Trust Over Time")
plt.xlabel("Time Step")
plt.ylabel("Trust")

# --------------------------------------------------
# PLOT 2 : Total Demand
# --------------------------------------------------

plt.figure()
plt.plot(data["Total_Demand"])
plt.title("Total Demand Over Time")
plt.xlabel("Time Step")
plt.ylabel("Demand")

# --------------------------------------------------
# PLOT 3 : Urban Stability Index
# --------------------------------------------------

plt.figure()
plt.plot(data["USI"])
plt.title("Urban Stability Index Over Time")
plt.xlabel("Time Step")
plt.ylabel("USI")

# --------------------------------------------------
# PLOT 4 : Gini Coefficient
# --------------------------------------------------

plt.figure()
plt.plot(data["Gini"])
plt.title("Gini Coefficient Over Time")
plt.xlabel("Time Step")
plt.ylabel("Gini")

# --------------------------------------------------
# PLOT 5 : Supply vs Demand
# --------------------------------------------------

plt.figure()
plt.plot(data["Total_Supply"], label="Supply")
plt.plot(data["Total_Demand"], label="Demand")
plt.title("Supply vs Demand Over Time")
plt.xlabel("Time Step")
plt.ylabel("Resources")
plt.legend()

# --------------------------------------------------
# PLOT 6 : USI Component Breakdown
# --------------------------------------------------

plt.figure()
plt.plot(data["S_R"], label="Resource Sufficiency (S_R)")
plt.plot(data["S_I"], label="Inequality Stability (S_I)")
plt.plot(data["S_O"], label="Oscillation Stability (S_O)")
plt.title("USI Component Breakdown")
plt.xlabel("Time Step")
plt.ylabel("Metric Value")
plt.legend()

# --------------------------------------------------
# SHOW ALL PLOTS
# --------------------------------------------------

plt.show()