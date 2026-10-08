import numpy as np
import rpy2.robjects as robjects
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as spi
import scipy.stats as stats
import matplotlib as mpl
from scipy.signal import savgol_filter

font = {'family' : 'normal', 'size' : 22}
mpl.rc('font', **font)
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
mpl.rcParams['xtick.labelsize'] = 20          # Set global font size for x-tick labels
mpl.rcParams['ytick.labelsize'] = 20          # Set global font size for y-tick labels
mpl.rcParams['xtick.major.size'] = 10           # Set global length for major x-ticks
mpl.rcParams['ytick.major.size'] = 10           # Set global length for major y-ticks
mpl.rcParams['axes.linewidth'] = 2            # Set global thickness for axis lines
mpl.rcParams['xtick.major.size'] = 10         # Set global length for major x-ticks
mpl.rcParams['ytick.major.size'] = 10         # Set global length for major y-ticks
mpl.rcParams['xtick.major.width'] = 2         # Set global width for major x-ticks
mpl.rcParams['ytick.major.width'] = 2         # Set global width for major y-ticks

sample_names = pd.read_csv('hgdp_new/sample.names', header=None, sep=' ')

chr_list = [1]

pop = "Hazara"
num_sam = sample_names[sample_names[0] == pop].shape[0]
mos_local = robjects.r["load"](
    f"MOSAIC_RESULTS/localanc_{pop}_2way_1-{num_sam}_1-5_1858_50_0.99_100.RData"
)

global_avg_gb = []
global_avg_mosaic = []
local_gb_all = []
local_mosaic_all = []

for sam_no, sam in enumerate(sample_names[sample_names[0] == pop][1]):
    if len(glob.glob(f'../recent/{pop.lower()}_cmgrid_overall_membership_{sam}_sample_id_*.csv')) < 2:
        continue
    mosaic_local_anc_wg = []
    gb_local_anc_wg = []
    positions = []
    for chr in chr_list:
        local_anc2 = np.array(robjects.r["localanc"][chr-1])
        local_anc2 = local_anc2[:, 0::2] + local_anc2[:, 1::2]
        pos = np.array(robjects.r["g.loc"][chr-1])
        hgdp_sample_of_interest = local_anc2[0, sam_no]
        for file_no, file in enumerate(glob.glob(f'../recent/{pop.lower()}_cmgrid_overall_membership_{sam}_sample_id_*.csv')):
            if file_no == 0:
                df_gb = pd.read_csv(file, sep='\s+')
            else:
                df_gb[['prob_'+str(i) for i in range(df_gb.shape[1]-3)]] += pd.read_csv(file, sep='\s+')[['prob_'+str(i) for i in range(df_gb.shape[1]-3)]]
        df_gb = df_gb[df_gb['chr'] == chr]
        start_pos = np.min(np.where(pos > df_gb['pos'].min())[0])
        end_pos = np.max(np.where(pos < df_gb['pos'].max())[0])
        pos = pos[start_pos:end_pos]
        hgdp_sample_of_interest = hgdp_sample_of_interest[start_pos:end_pos]
        f = spi.interp1d(df_gb['pos'], df_gb['prob_1'], kind='linear')
        mosaic_local_anc_wg.extend(hgdp_sample_of_interest.tolist())
        local_mosaic_all.extend(hgdp_sample_of_interest.tolist())
        gb_local_anc_wg.extend(f(pos).tolist())
        local_gb_all.extend(f(pos).tolist())
        positions.extend((pos).tolist())

    global_avg_gb.append(np.mean(gb_local_anc_wg)/2)
    global_avg_mosaic.append(np.mean(mosaic_local_anc_wg)/2)
    print(sam + " " + str(np.mean(mosaic_local_anc_wg)/2) + " " + str(np.mean(gb_local_anc_wg)/2) + " " + str(stats.pearsonr(mosaic_local_anc_wg, gb_local_anc_wg)[0]))

    combined_data = np.vstack((mosaic_local_anc_wg[::20], gb_local_anc_wg[::20]))
    combined_data = pd.DataFrame(combined_data.T, columns=['Mosaic', 'GhostCatcher'])
    positions = np.array(positions)[::20]
    x_values = np.arange(len(mosaic_local_anc_wg[::20]))  # X-axis values for each site
    y_values_mosaic = np.clip(savgol_filter(mosaic_local_anc_wg[::20], window_length=11, polyorder=2), 0, 2)  # Smooth data
    y_values_gb = np.clip(savgol_filter(gb_local_anc_wg[::20], window_length=11, polyorder=2), 0, 2)  # Smooth data

    # Plotting the two heatmaps as subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 4), sharex=True, gridspec_kw={'hspace': 0.5})

    # Smooth local ancestry for Mosaic
    ax1.fill_between(x_values, 0, y_values_mosaic, facecolor='pink', alpha=0.8)
    ax1.fill_between(x_values, y_values_mosaic, 2, facecolor='green', alpha=0.8)
    ax1.plot(x_values, y_values_mosaic, color='black')
    ax1.set_title('Mosaic', fontsize=20)

    # Remove axis labels and ticks
    x_ticks_indices = np.linspace(0, len(positions) - 1, 10, dtype=int)
    ax1.set_xticks([])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticks(x_ticks_indices)  # Set the tick marks
    ax1.set_xticklabels([''] * len(x_ticks_indices))  # Set empty labels


    # Smooth local ancestry for GhostCatcher
    ax2.fill_between(x_values, 0, y_values_gb, facecolor='pink', alpha=0.8)
    ax2.fill_between(x_values, y_values_gb, 2, facecolor='green', alpha=0.8)
    ax2.plot(x_values, y_values_gb, color='black')
    ax2.set_title('GhostCatcher', fontsize=20)

    # Remove axis labels and ticks
    ax2.set_xticks([])
    ax2.set_yticks([0, 1, 2])

    # x_ticks_indices = [0, len(positions) - 1]  # Indices for min and max positions
    # x_ticks_labels = [f"{int(positions[i]/1e7)*1e7}" for i in x_ticks_indices]  # Labels for min and max positions
    # ax2.set_xticks(x_ticks_indices)
    # ax2.set_xticklabels(x_ticks_labels, fontsize=18, rotation=45, ha='right')
    # x_ticks_indices = [i for i, pos in enumerate(positions) if pos < positions[i-1]]
    # x_ticks_labels = [f"chr{int(count)+1}" for count, i in enumerate(x_ticks_indices)]
    # ax2.set_xticks(x_ticks_indices)
    # ax2.set_xticklabels(x_ticks_labels, fontsize=18, rotation=45, ha='right')
    # ax2.set_xticks([])
    ax2.set_xticks(x_ticks_indices)  # Set the tick marks
    ax2.set_xticklabels([''] * len(x_ticks_indices))  # Set empty labels
    ax2.set_xlabel('Genomic position', fontsize=20)
    plt.subplots_adjust(bottom=0.15)
    # plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15) 
    plt.tight_layout()
    plt.savefig(f'../recent/{pop.lower()}_cmgrid_supervised_chr{chr}_overall_membership_{sam}.png', dpi=600, transparent=True)
    plt.show()


### Overall correlation
print("Overall correlation: " + str(stats.pearsonr(local_mosaic_all, local_gb_all)[0]) + " " + str(stats.spearmanr(local_mosaic_all, local_gb_all)[0]))

### OLS fir for global average and plot
global_avg_gb = np.array(global_avg_gb)
global_avg_mosaic = np.array(global_avg_mosaic)
global_avg_gb = global_avg_gb.reshape(-1,1)
global_avg_mosaic = global_avg_mosaic.reshape(-1,1)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(global_avg_mosaic, global_avg_gb)
print("R^2: " + str(reg.score(global_avg_mosaic, global_avg_gb)))
print("Slope: " + str(reg.coef_))
print("Intercept: " + str(reg.intercept_))

## plot the OLS fit
plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.scatter(global_avg_mosaic, global_avg_gb)
ax.plot(global_avg_mosaic, reg.predict(global_avg_mosaic), color='red')
plt.xlabel('Mosaic prop.')
plt.ylabel('GhostCatcher prop.')
plt.title('OLS fit of Mosaic and GhostCatcher ancestry')
plt.savefig('mosaic_vs_gb.png', dpi=300)
