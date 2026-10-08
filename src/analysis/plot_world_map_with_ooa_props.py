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
            prefix = "data/hgdp_1gp_v3/deepadmix_nohmm/"
        elif pop in ancient_labels:
            prefix = "data/hgdp_1gp_ancients/deepadmix_nohmm/"
        else:
            prefix = "data/SGDP_only_moderns/deepadmix_nohmm/"
        for file in glob.glob(
            prefix + str(pop) + "_overall_membership_*_sample_id_*.csv"
        ):
            if pop in [
                "Somali",
                "Luo",
                "Masai",
                "Ju_hoan_North",
                "Dinka",
                "Khomani_San",
            ]:
                file_runall = file.replace(
                    pop + "_overall_membership", "SGDP_specific_afr_overall_membership"
                )
                df = pd.read_csv(file_runall, sep="\t")
            else:
                df = pd.read_csv(file, sep="\t")
            minor_comp = (
                "prob_1" if df["prob_1"].mean() < df["prob_0"].mean() else "prob_0"
            )
            df_bta = pd.read_csv(
                file.replace("deepadmix_nohmm/", "back_to_africa_using_relate_gammas/"),
                sep="\t",
            )
            bta_prop = df_bta["prob_1"].mean()
            prop_list.append(100 * (df[minor_comp].mean() - bta_prop) / (1 - bta_prop))
            # prop_list.append(
            #     100
            #     * np.nansum(df[minor_comp] > 0.8)
            #     / (np.nansum(df[minor_comp] > 0.8) + np.nansum(df[minor_comp] < 0.2))
            # )
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
    s=250,
    c=c[:-5],
    vmin=0,
    # vmax=40,
    cmap="Reds",
    edgecolor="k",
    marker="v",
)
m.scatter(
    x[-5:],
    y[-5:],
    s=200,
    c=c[-5:],
    vmin=0,
    # vmax=40,
    cmap="Reds",
    edgecolor="k",
    marker=",",
)

# Add colorbar
ticks = np.linspace(0, 40, num=5)  # e.g. [0,  max_prop/5, …, max_prop]
cbar = m.colorbar(sc, location="right", pad="25%", ticks=ticks)
cbar.set_label("OOA-like ancestry Proportion %", fontsize=24)
# tick_labels = [str(int(t)) for t in ticks[:-1]] + [">50"]
tick_labels = [str(int(t)) for t in ticks]
cbar.ax.set_yticklabels(tick_labels)
cbar.ax.tick_params(labelsize=20)

# for i, population in enumerate(data['Population']):
#     plt.text(x[i], y[i], population, fontsize=9, ha='left', va='bottom', zorder=10)

plt.tight_layout()
# Show the plot
plt.savefig("afr_lat_long.svg", dpi=300, transparent=True)
plt.show()
