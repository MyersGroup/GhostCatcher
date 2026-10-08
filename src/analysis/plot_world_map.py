import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Data for the African populations
data = pd.read_csv("afr_lat_long.txt", sep=r"\s+")
data = data.loc[2:]

plt.clf()

# Create the map
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figsize for better aspect ratio

# Create a Basemap instance
m = Basemap(
    projection="cyl", llcrnrlat=-38, urcrnrlat=40, llcrnrlon=-20, urcrnrlon=52, ax=ax
)
m.drawmapboundary(
    fill_color="lightblue", linewidth=0
)  # Set the map boundary fill color to blue and remove the boundary line
m.drawlsmask(
    land_color="lightgray", ocean_color="lightblue"
)  # Fill land with grey and ocean with blue, remove lakes
m.drawcoastlines()

# Convert latitude and longitude to map projection coordinates
x, y = m(data["Longitude"].values, data["Latitude"].values)
s = data["Samples"]

# Plot data points
m.scatter(x[:-5], y[:-5], s=450, color="red", alpha=1, edgecolor="k", marker="v")
m.scatter(x[-5:], y[-5:], s=400, color="green", alpha=1, edgecolor="k", marker=",")

# for i, population in enumerate(data['Population']):
#     plt.text(x[i], y[i], population, fontsize=9, ha='left', va='bottom', zorder=10)

plt.tight_layout()
# Show the plot
plt.savefig("afr_lat_long.svg", dpi=300, transparent=True)
plt.show()
