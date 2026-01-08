import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation_results.csv")

# Filter for one model (e.g., PPO Latent)
subset = df[df["Model_Name"] == "PPO_Latent"]

# Pivot: Rows=Force, Cols=Angle, Values=Success Rate
heatmap_data = subset.pivot_table(index="Wind_Force", columns="Wind_Angle", values="Success")

sns.heatmap(heatmap_data, cmap="RdYlGn", annot=True)
plt.title("PPO Latent: Success Rate by Wind Condition")
plt.show()