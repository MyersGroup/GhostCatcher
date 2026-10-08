import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import sys
import statsmodels.api as sm
from matplotlib.colors import Normalize
from matplotlib import cm

import matplotlib as mpl
font = {'family' : 'normal', 'size' : 22}
mpl.rc('font', **font)
plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
mpl.rcParams['xtick.labelsize'] = 16          # Set global font size for x-tick labels
mpl.rcParams['ytick.labelsize'] = 16          # Set global font size for y-tick labels
mpl.rcParams['xtick.major.size'] = 10           # Set global length for major x-ticks
mpl.rcParams['ytick.major.size'] = 10           # Set global length for major y-ticks
mpl.rcParams['axes.linewidth'] = 2            # Set global thickness for axis lines
mpl.rcParams['xtick.major.size'] = 10         # Set global length for major x-ticks
mpl.rcParams['ytick.major.size'] = 10         # Set global length for major y-ticks
mpl.rcParams['xtick.major.width'] = 2         # Set global width for major x-ticks
mpl.rcParams['ytick.major.width'] = 2         # Set global width for major y-ticks

output_file_name = sys.argv[1]

sample_list = []
chr_list = []
for file in glob.glob(output_file_name + "_fixed_params_chr*_sample*.pkl"):
    sample = int(file.split("sample")[-1].split(".")[0])
    chr = int(file.split("chr")[-1].split('_sample')[0])
    sample_list.append(sample)
    chr_list.append(chr)

sample_list = np.unique(sample_list)
chr_list = np.unique(chr_list)
# chr_list = [2]

post_overall = []
num_overall = []
denom_overall = []
for sample in sample_list:
    post = pd.read_csv(glob.glob(output_file_name + '_overall_membership_*_sample_id_{0}.csv'.format(sample))[0], sep=r'\s+')
    post_overall.extend(post[['prob_' + str(i) for i in range(post.shape[1] - 3)]].values)
    for chr in chr_list:
        fixed_params_file_name = output_file_name + "_fixed_params_chr{0}_sample{1}.pkl".format(chr, sample)
        with open(fixed_params_file_name, "rb") as f_pkl:
            data = pickle.load(f_pkl)
        try:
            (mut_scaling_file, hmm_file, force_build, start_time, end_time, ignore_first_epoch, ignore_last_epoch, masking_threshold, poplabels_file, coal_count, denom, denom_unscaled, proportion_of_coalescing, epoch_index, gt_ref_file, unique_groups_file, exact_pos_file) = data
        except:
            (sample_id_file, mut_scaling_file, hmm_file, force_build, start_time, end_time, ignore_first_epoch, ignore_last_epoch, masking_threshold, poplabels_file, coal_count, denom, denom_unscaled, proportion_of_coalescing, epoch_index, gt_ref_file, unique_groups_file, exact_pos_file) = data 
        _, num_groups, num_epochs = denom.shape
        start_index = 1 if ignore_first_epoch else 0
        end_index = num_epochs-2 if ignore_last_epoch else num_epochs-1
        # end_index = int((end_time - start_time)*num_epochs/4)  ### use this when coal rates supplied while running GB !!caution!!
        ### change end-index based on end_time and start_time
        for i in range(len(proportion_of_coalescing)):
            sum_prop_coal = np.zeros(num_groups)
            for c in range(len(proportion_of_coalescing[i])):
                if epoch_index[i][c] >= start_index and epoch_index[i][c] <= end_index:
                    ratio_prop_of_coal = np.array(proportion_of_coalescing[i][c])/np.sum(proportion_of_coalescing[i][c])
                    sum_prop_coal = sum_prop_coal + ratio_prop_of_coal
            num_overall.append(sum_prop_coal)
            denom_overall.append(np.sum(denom[i,:,start_index:end_index+1], axis=1))

# Process data
post_overall = np.array(post_overall)
num = np.array(num_overall)
denom = np.array(denom_overall)

# Removing columns where all sums are zero
num = num[:, np.sum(num, axis=0) != 0]
denom = denom[:, np.sum(denom, axis=0) != 0]

# Normalize num and denom
num -= np.mean(num, axis=0)
denom -= np.mean(denom, axis=0)

num /= np.std(num, axis=0)
denom /= np.std(denom, axis=0)

## Removing columns where there are nan
num = num[:, ~np.isnan(num).any(axis=0)]
denom = denom[:, ~np.isnan(denom).any(axis=0)]

# Combine num and denom to form the feature matrix
X = np.hstack((num, denom))
model = sm.OLS(post_overall[:, 1], sm.add_constant(X)).fit()
print(model.summary())

# PCA Transformation
pca = PCA(n_components=4)
X_reduced = pca.fit_transform(X)

# Creating DataFrame for PCA components and Posterior probabilities
df_pca = pd.DataFrame(X_reduced, columns=['PC' + str(i) for i in range(1, 5)])
for i in range(post_overall.shape[1]):
    df_pca['Posterior_' + str(i)] = post_overall[:, i]

# Determine the component with the highest posterior for each sample
df_pca['Posterior_argmax'] = np.argmax(post_overall, axis=1)

# Mapping components to more interpretable labels (e.g., 'component 1', 'component 2', ...)
df_pca['Posterior_bin'] = df_pca['Posterior_argmax'].map(lambda x: 'Component ' + str(x + 1))

# Separate the components
components = np.sort(df_pca['Posterior_bin'].unique())
if len(df_pca) > 100000:
    df_pca = df_pca.sample(100000)

# Linear regression with PCs to predict posterior for each component
for i in range(post_overall.shape[1]):
    df_pca['constant'] = 1
    model = sm.OLS(df_pca['Posterior_' + str(i)], df_pca[['PC' + str(j) for j in range(1, 5)] + ['constant']]).fit()
    print(f"Summary for Posterior_{i}:")
    print(model.summary())

# Plotting KDEs for each component pairwise
# plt.clf()
# fig, ax = plt.subplots(2, len(components), figsize=(5 * len(components), 10), dpi=300)

# # To store the min/max limits after plotting for PC1-PC2 and PC3-PC4
# x1_min, x1_max = float('inf'), float('-inf')
# y1_min, y1_max = float('inf'), float('-inf')

# x3_min, x3_max = float('inf'), float('-inf')
# y3_min, y3_max = float('inf'), float('-inf')

# # First pass to determine the limits
# for idx, component in enumerate(components):
#     # PC1 vs PC2 for each component (first row)
#     sns.kdeplot(data=df_pca[df_pca['Posterior_bin'] == component], x='PC1', y='PC2', ax=ax[0, idx], 
#                 fill=True, color=palette[idx], cbar=False, cut=0.5)
    
#     # Infer the limits from the plot
#     x1_min_temp, x1_max_temp = ax[0, idx].get_xlim()
#     y1_min_temp, y1_max_temp = ax[0, idx].get_ylim()
    
#     x1_min = min(x1_min, x1_min_temp)
#     x1_max = max(x1_max, x1_max_temp)
#     y1_min = min(y1_min, y1_min_temp)
#     y1_max = max(y1_max, y1_max_temp)
    
#     # PC3 vs PC4 for each component (second row)
#     sns.kdeplot(data=df_pca[df_pca['Posterior_bin'] == component], x='PC3', y='PC4', ax=ax[1, idx], 
#                 fill=True, color=palette[idx], cbar=False, cut=0.5)
    
#     # Infer the limits from the plot
#     x3_min_temp, x3_max_temp = ax[1, idx].get_xlim()
#     y3_min_temp, y3_max_temp = ax[1, idx].get_ylim()
    
#     x3_min = min(x3_min, x3_min_temp)
#     x3_max = max(x3_max, x3_max_temp)
#     y3_min = min(y3_min, y3_min_temp)
#     y3_max = max(y3_max, y3_max_temp)

pc1_min = df_pca['PC1'].quantile(0.02)
pc1_max = df_pca['PC1'].quantile(0.98)
pc2_min = df_pca['PC2'].quantile(0.02)
pc2_max = df_pca['PC2'].quantile(0.98)
pc3_min = df_pca['PC3'].quantile(0.02)
pc3_max = df_pca['PC3'].quantile(0.98)
pc4_min = df_pca['PC4'].quantile(0.02)
pc4_max = df_pca['PC4'].quantile(0.98)
palette = ['purple','green','red','blue','orange','brown','pink','gray','olive','cyan']

# Second pass to re-plot with consistent limits
plt.clf()
fig, ax = plt.subplots(2, len(components), figsize=(5 * len(components), 10), dpi=300)

for idx, component in enumerate(components):
    sns.kdeplot(
        data=df_pca[df_pca['Posterior_bin'] == component], 
        x='PC1', 
        y='PC2', 
        ax=ax[0,idx], 
        fill=True, 
        color=palette[idx], 
        cbar=False, 
        cut=0, 
        clip=((pc1_min, pc1_max), (pc2_min, pc2_max))
    )
    ax[0,idx].set_xlim(pc1_min, pc1_max)
    ax[0,idx].set_ylim(pc2_min, pc2_max)
    ax[0,idx].set_xlabel('PC1', fontsize=18)
    ax[0,idx].set_ylabel('PC2', fontsize=18)
    ax[0,idx].set_title(f'{component}', fontsize=18, loc='center')

    # PC3 vs PC4 for each component (second row)
    sns.kdeplot(
        data=df_pca[df_pca['Posterior_bin'] == component], 
        x='PC3', 
        y='PC4', 
        ax=ax[1,idx], 
        fill=True, 
        color=palette[idx], 
        cbar=False, 
        cut=0, 
        clip=((pc3_min, pc3_max), (pc4_min, pc4_max))
    )
    ax[1, idx].set_xlim(pc3_min, pc3_max)  # Apply the global x-axis limit for PC3
    ax[1, idx].set_ylim(pc4_min, pc4_max)  # Apply the global y-axis limit for PC4
    ax[1, idx].set_xlabel(f'PC3', fontsize=18)
    ax[1, idx].set_ylabel(f'PC4', fontsize=18)
    # ax[1,idx].set_title(f'{component}', fontsize=18, loc='center')

plt.tight_layout()
plt.savefig(output_file_name + '_pca.svg', dpi=300)
plt.show()


# Balanced KDE plots
# plt.clf()
# fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=200)

# # PC1 vs PC2
# sns.kdeplot(data=df_pca_balanced, x='PC1', y='PC2', hue='Posterior_bin', ax=ax[0], alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')

# # PC3 vs PC4
# sns.kdeplot(data=df_pca_balanced, x='PC3', y='PC4', hue='Posterior_bin', ax=ax[1], alpha=0.5)
# ax[1].set_xlabel('PC3')
# ax[1].set_ylabel('PC4')

# # Remove top and right boxes
# for a in ax:
#     a.spines['top'].set_visible(False)
#     a.spines['right'].set_visible(False)
#     a.legend(frameon=False)  # This removes the legend box

# plt.tight_layout()
# plt.savefig(output_file_name + '_pca_balanced.svg', dpi=300)

## Usage: python visualize_pca.py ../../../denisovan_sim_24_08/output_nonghost/true