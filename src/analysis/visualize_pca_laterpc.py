import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import sys
import statsmodels.api as sm
import matplotlib as mpl

font = {'family': 'normal', 'size': 22}
mpl.rc('font', **font)
plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

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
        for i in range(len(proportion_of_coalescing)):
            sum_prop_coal = np.zeros(num_groups)
            for c in range(len(proportion_of_coalescing[i])):
                if epoch_index[i][c] >= start_index and epoch_index[i][c] <= end_index:
                    ratio_prop_of_coal = np.array(proportion_of_coalescing[i][c])/np.sum(proportion_of_coalescing[i][c])
                    sum_prop_coal = sum_prop_coal + ratio_prop_of_coal
            num_overall.append(sum_prop_coal)
            denom_overall.append(np.sum(denom[i,:,start_index:end_index+1], axis=1))

post_overall = np.array(post_overall)
num = np.array(num_overall)
denom = np.array(denom_overall)

num = num[:, np.sum(num, axis=0) != 0]
denom = denom[:, np.sum(denom, axis=0) != 0]

num -= np.mean(num, axis=0)
denom -= np.mean(denom, axis=0)

num /= np.std(num, axis=0)
denom /= np.std(denom, axis=0)

num = num[:, ~np.isnan(num).any(axis=0)]
denom = denom[:, ~np.isnan(denom).any(axis=0)]

X = np.hstack((num, denom))
model = sm.OLS(post_overall[:, 1], sm.add_constant(X)).fit()
print(model.summary())

pca = PCA(n_components=8)
X_reduced = pca.fit_transform(X)

df_pca = pd.DataFrame(X_reduced, columns=['PC' + str(i) for i in range(1, 9)])
for i in range(post_overall.shape[1]):
    df_pca['Posterior_' + str(i)] = post_overall[:, i]

df_pca['Posterior_argmax'] = np.argmax(post_overall, axis=1)
df_pca['Posterior_bin'] = df_pca['Posterior_argmax'].map(lambda x: 'Component ' + str(x + 1))
components = np.sort(df_pca['Posterior_bin'].unique())
if len(df_pca) > 100000:
    df_pca = df_pca.sample(100000)

pc5_min = df_pca['PC5'].quantile(0.02) if 'PC5' in df_pca.columns else None
pc5_max = df_pca['PC5'].quantile(0.98) if 'PC5' in df_pca.columns else None
pc6_min = df_pca['PC6'].quantile(0.02) if 'PC6' in df_pca.columns else None
pc6_max = df_pca['PC6'].quantile(0.98) if 'PC6' in df_pca.columns else None
pc7_min = df_pca['PC7'].quantile(0.02) if 'PC7' in df_pca.columns else None
pc7_max = df_pca['PC7'].quantile(0.98) if 'PC7' in df_pca.columns else None
pc8_min = df_pca['PC8'].quantile(0.02) if 'PC8' in df_pca.columns else None
pc8_max = df_pca['PC8'].quantile(0.98) if 'PC8' in df_pca.columns else None

palette = ['purple', 'green', 'red', 'blue', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

plt.clf()
fig, ax = plt.subplots(2, len(components), figsize=(5 * len(components), 10), dpi=300)

for idx, component in enumerate(components):
    if 'PC5' in df_pca.columns and 'PC6' in df_pca.columns:
        sns.kdeplot(data=df_pca[df_pca['Posterior_bin'] == component], x='PC5', y='PC6', ax=ax[0, idx], fill=True, color=palette[idx], cbar=False, cut=0, clip=((pc5_min, pc5_max), (pc6_min, pc6_max)))
        ax[0, idx].set_xlim(pc5_min, pc5_max)
        ax[0, idx].set_ylim(pc6_min, pc6_max)
        ax[0, idx].set_xlabel('PC5', fontsize=18)
        ax[0, idx].set_ylabel('PC6', fontsize=18)
        ax[0, idx].set_title(f'{component}', fontsize=18, loc='center')

    if 'PC7' in df_pca.columns and 'PC8' in df_pca.columns:
        sns.kdeplot(data=df_pca[df_pca['Posterior_bin'] == component], x='PC7', y='PC8', ax=ax[1, idx], fill=True, color=palette[idx], cbar=False, cut=0, clip=((pc7_min, pc7_max), (pc8_min, pc8_max)))
        ax[1, idx].set_xlim(pc7_min, pc7_max)
        ax[1, idx].set_ylim(pc8_min, pc8_max)
        ax[1, idx].set_xlabel('PC7', fontsize=18)
        ax[1, idx].set_ylabel('PC8', fontsize=18)

plt.tight_layout()
plt.savefig(output_file_name + '_pca_PC5_PC8.svg', dpi=300)
plt.show()
