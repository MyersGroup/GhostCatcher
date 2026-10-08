import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import sys
import statsmodels.api as sm
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import random
import sys
import matplotlib as mpl
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

def calculate_ece(prob_true, prob_pred, n_bins=20):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    ece = 0
    for i in range(n_bins):
        # Find points in each bin
        in_bin = (prob_pred >= bin_boundaries[i]) & (prob_pred < bin_boundaries[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(prob_true[in_bin])  # True accuracy in the bin
            bin_conf = np.mean(prob_pred[in_bin])  # Predicted confidence in the bin
            ece += bin_size * np.abs(bin_acc - bin_conf)  # Weighted by bin size
    ece /= len(prob_pred)  # Normalize by the total number of points
    return ece

def get_pr_calibration(post_file_name, gt_file_name, sample_list):
    post_overall = []
    gt_overall = []
    for sample in sample_list:
        post = pd.read_csv(glob.glob(post_file_name + '_overall_membership_*_sample_id_{0}.csv'.format(sample))[0], sep=r'\s+')
        mean_prop = post[[col for col in post.columns if col.startswith('prob_')]].mean(axis=0)
        post = post.drop(columns=[f'prob_{max_idx}' for max_idx in np.where(mean_prop == mean_prop.max())[0]])
        gt = pd.read_csv(gt_file_name + '_{0}.csv'.format(sample), sep=r'\s+')
        gt = gt.rename(columns={f'prob_{i}': f'gt_{i}' for i in range(len([col for col in gt.columns if col.startswith('prob_')]))})
        gt_components = [col for col in gt.columns if col.startswith('gt_')]
        post_components = [col for col in post.columns if col.startswith('prob_')]
        mdf = pd.merge(post, gt, on=['chr', 'pos'])
        for post_comp in post_components:
            best_r2 = -np.inf
            best_gt_comp = None
            for gt_comp in gt_components:
                if gt_comp in gt.columns:
                    r2 = np.corrcoef(mdf[gt_comp], mdf[post_comp])[0, 1]
                    if r2 > best_r2:
                        best_r2 = r2
                        best_gt_comp = gt_comp
            post_overall.extend(mdf[post_comp].values.tolist())
            gt_overall.extend(mdf[best_gt_comp].values.tolist())
    r2 = np.corrcoef(gt_overall, post_overall)[0, 1]**2
    precision, recall, thresholds = precision_recall_curve(gt_overall, post_overall)
    sampled_pairs = random.sample(list(zip(precision, recall)), 200)
    precision_sampled, recall_sampled = zip(*sampled_pairs)
    sorted_pairs = sorted(zip(precision_sampled, recall_sampled), key=lambda x: x[1])
    precision, recall = zip(*sorted_pairs)
    ap = average_precision_score(gt_overall, post_overall)
    prob_true, prob_pred = calibration_curve(gt_overall, post_overall, n_bins=20)
    ece = calculate_ece(prob_true, prob_pred)
    return r2, precision, recall, ap, prob_pred, prob_true, ece

def plot_results(gt_file_name, relate_file_name, true_file_name=None, skov_file_name=None):
    sample_list = []
    for file in glob.glob(relate_file_name + "_overall_membership_*_sample_id_*.csv"):
        sample = int(file.split("sample_id_")[-1].split(".")[0])
        sample_list.append(sample)

    sample_list = np.unique(sample_list)
    
    r2_relate, precision_relate, recall_relate, ap_relate, prob_pred_relate, prob_true_relate, ece_relate = get_pr_calibration(relate_file_name, gt_file_name, sample_list)
    results = [("Relate trees", r2_relate, precision_relate, recall_relate, ap_relate, prob_pred_relate, prob_true_relate, ece_relate)]
    
    if true_file_name:
        r2_true, precision_true, recall_true, ap_true, prob_pred_true, prob_true_true, ece_true = get_pr_calibration(true_file_name, gt_file_name, sample_list)
        results.append(("True trees", r2_true, precision_true, recall_true, ap_true, prob_pred_true, prob_true_true, ece_true))

    if skov_file_name:
        r2_skov, precision_skov, recall_skov, ap_skov, prob_pred_skov, prob_true_skov, ece_skov = get_pr_calibration(skov_file_name, gt_file_name, sample_list)
        results.append(("HMMIX", r2_skov, precision_skov, recall_skov, ap_skov, prob_pred_skov, prob_true_skov, ece_skov))
    
    # Sort results by R²
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Plot PR Curve
    plt.clf()
    plt.figure(figsize=(8, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Relate, True, Skov et al.
    for idx, (label, r2, precision, recall, ap, prob_pred, prob_true, ece) in enumerate(results):
        plt.plot(recall, precision, color=colors[idx], linestyle='-', linewidth=4, marker='o', label=f"{label} (AP = {ap:.2f})")
    plt.xlabel('Recall (1 - FNR)', fontsize=22)
    plt.ylabel('Precision (1 - FDR)', fontsize=22)
    plt.legend(ncol=1, fontsize=18, frameon=False)
    plt.tight_layout()
    plt.savefig(relate_file_name + '_pr.svg', dpi=300, transparent=True)
    plt.show()

    # Plot Calibration Curve
    plt.clf()
    plt.figure(figsize=(8, 8))
    for idx, (label, r2, precision, recall, ap, prob_pred, prob_true, ece) in enumerate(results):
        plt.plot(prob_pred, prob_true, marker='o', color=colors[idx], label=f"{label} (ECE = {ece:.2f})", lw=4)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted Probability', fontsize=22)
    plt.ylabel('True Probability', fontsize=22)
    plt.legend(fontsize=18, ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig(relate_file_name + '_calibration.svg', dpi=300, transparent=True)
    plt.show()

    # Plot R² Bar Plot
    plt.clf()
    plt.figure(figsize=(5, 8))
    data = {'Scenario': [res[0] for res in results], 'R2': [res[1] for res in results]}
    r2_values = pd.DataFrame(data)
    ### bootstrap this to get SE with R2 
    print("R2 = " + str(r2_values.loc[:, 'R2'].values))
    ax = sns.barplot(x='Scenario', y='R2', data=r2_values, palette=colors[:len(results)], width=0.4, dodge=False, edgecolor='black')

    # Adjust the positioning of the tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)

    # Annotate the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.01), 
                    ha='center', va='bottom', fontsize=20, color='black')

    plt.ylabel(r'$R^2$', fontsize=20)
    plt.xlabel(None)
    plt.ylim(0, 1.1)  # Adjust y-axis limit for better appearance
    plt.tight_layout()
    plt.savefig(relate_file_name + '_r2.svg', dpi=300, transparent=True)
    plt.show()

# Command-line usage
if __name__ == "__main__":
    gt_file_name = sys.argv[1]
    relate_file_name = sys.argv[2]
    true_file_name = sys.argv[3] if len(sys.argv) > 3 else None
    skov_file_name = sys.argv[4] if len(sys.argv) > 4 else None
    plot_results(gt_file_name, relate_file_name, true_file_name, skov_file_name)


## Usage: python ../clean/RelateLocalAncestry/plotting/visualize_r2.py ground_truth output_ghost/relate output_ghost/true skov/skov