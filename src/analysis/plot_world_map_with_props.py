import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob
import numpy as np


def plot_proportions():
    prop_list_all = []
    pop_labels = [
        "esn",
        "gwd",
        "msl",
        "lwk",
        "mbutipygmy",
        "biakapygmy",
        "san",
        "mandenka",
        "bantusafrica",
        "bantukenya",
        "yoruba",
        "I10871",
        "I5950",
        "ela001",
        "new001",
        "baa001",
        "Somali",
        "Luo",
        "Masai",
        "Ju_hoan_North",
        "Dinka",
        "Khomani_San",
    ]
    ancient_labels = ["I10871", "I5950", "ela001", "new001", "baa001"]
    modern_labels = [pop for pop in pop_labels if pop not in ancient_labels]
    for pop in pop_labels:
        print(pop)
        prop_list = []
        if pop in [
            "esn",
            "gwd",
            "msl",
            "lwk",
            "mbutipygmy",
            "biakapygmy",
            "san",
            "mandenka",
            "bantusafrica",
            "bantukenya",
            "yoruba",
        ]:
            prefix = "data/hgdp_1gp_v3/back_to_africa/"
        elif pop in ancient_labels:
            prefix = "data/hgdp_1gp_ancients/back_to_africa/"
        else:
            prefix = "data/SGDP_only_moderns/back_to_africa/"
        for file in glob.glob(
            prefix + str(pop) + "_overall_membership_*_sample_id_*.csv"
        ):
            df = pd.read_csv(file, sep=r"\s+")
            r = ((df['genpos'].shift(2) - df['genpos'].shift(-2))/(df['pos'].shift(2) - df['pos'].shift(-2)))
            df = df.loc[r < r.quantile(0.5)]
            prop_list.append(100 * (df["prob_1"] > 0.5).mean())
        prop_list_all.append(prop_list)
    data = pd.DataFrame(prop_list_all).transpose()
    print(data)
    pop_labels = [
        "ESN",
        "GWD",
        "MSL",
        "LWK",
        "Mbuti",
        "Biaka",
        "San",
        "Mandenka",
        "BantuSAfrica",
        "BantuKenya",
        "Yoruba",
        "I10871",
        "I5950",
        "ela001",
        "new001",
        "baa001",
        "Somali",
        "Luo",
        "Masai",
        "JuHoanNorth",
        "Dinka",
        "KhomaniSan",
    ]
    data.columns = pop_labels
    mean_proportions = data.mean().sort_values(ascending=False)
    return mean_proportions


# Data for the African populations
data = pd.read_csv("afr_lat_long.txt", sep=r"\s+")
data = data.loc[2:]
props = plot_proportions()
data["BTA proportion"] = data["Population"].map(props)

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
c = data["BTA proportion"]
print(data)

# Plot data points
sc = m.scatter(
    x[:-5],
    y[:-5],
    s=350,
    c=c[:-5],
    vmin=0,
    vmax=5,
    cmap="Reds",
    edgecolor=None,
    marker="v",
)
m.scatter(
    x[-5:],
    y[-5:],
    s=300,
    c=c[-5:],
    vmin=0,
    vmax=5,
    cmap="Reds",
    edgecolor=None,
    marker=",",
)

# Add colorbar
ticks = np.linspace(0, 5, num=6)  # e.g. [0,  max_prop/5, …, max_prop]
cbar = m.colorbar(sc, location="right", pad="25%", ticks=ticks)
cbar.set_label("Eurasian Proportion %", fontsize=24)
tick_labels = [str(int(t)) for t in ticks[:-1]] + [">5"]
cbar.ax.set_yticklabels(tick_labels)
cbar.ax.tick_params(labelsize=20)

# for i, population in enumerate(data['Population']):
#     plt.text(x[i], y[i], population, fontsize=9, ha='left', va='bottom', zorder=10)

plt.tight_layout()
# Show the plot
plt.savefig("afr_lat_long.svg", dpi=300, transparent=True)
plt.show()
