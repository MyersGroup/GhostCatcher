import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

font = {'family': 'normal', 'size': 22}
mpl.rc('font', **font)
plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
palette = ['purple','green','red','blue','orange','brown','pink','gray','olive','cyan']

def plot_histogram_from_csv(df, output):
    prob_cols = np.sort([col for col in df.columns if col.startswith('prob_')])
    if len(prob_cols) < 1:
        print("No columns starting with 'prob_' found.")
        return
    num_components = len(prob_cols)
    plt.figure(figsize=(7 * num_components, 5))  # Adjust width dynamically
    for i, col in enumerate(prob_cols):
        plt.subplot(1, num_components, i + 1)
        plt.hist(df[col], bins=50, alpha=0.7, color=palette[i])
        comp = 'comp. ' + str(int(col.split('_')[-1])+1)
        plt.title(f'Histogram for {comp}', fontsize=24)
        plt.xlabel('Probability', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)

    plt.tight_layout()
    plt.savefig(output + '_histogram.svg', dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":
    post_file_name = sys.argv[1]
    dfc = []
    for file in glob.glob(post_file_name + "_overall_membership_*_sample_id_*.csv"):
        df = pd.read_csv(file, sep=r'\s+')
        dfc.append(df)
    combined_df = pd.concat(dfc, ignore_index=True)
    plot_histogram_from_csv(combined_df, post_file_name)
