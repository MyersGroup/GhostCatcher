import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
from scipy import interpolate
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline, BSpline
import glob
import argparse
import os
from src.utils import boolean
import numba
from tqdm import tqdm

import matplotlib as mpl

font = {"size": 22}
mpl.rc("font", **font)
plt.rc("axes.spines", **{"bottom": True, "left": True, "right": False, "top": False})
mpl.rcParams["xtick.labelsize"] = 20  # Set global font size for x-tick labels
mpl.rcParams["ytick.labelsize"] = 20  # Set global font size for y-tick labels
mpl.rcParams["xtick.major.size"] = 10  # Set global length for major x-ticks
mpl.rcParams["ytick.major.size"] = 10  # Set global length for major y-ticks
mpl.rcParams["axes.linewidth"] = 2  # Set global thickness for axis lines
mpl.rcParams["xtick.major.size"] = 10  # Set global length for major x-ticks
mpl.rcParams["ytick.major.size"] = 10  # Set global length for major y-ticks
mpl.rcParams["xtick.major.width"] = 2  # Set global width for major x-ticks
mpl.rcParams["ytick.major.width"] = 2  # Set global width for major y-ticks
mpl.rcParams["font.family"] = "DejaVu Sans"
MASK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "real_data_mask"
)


def make_interp_spline_(x, y):
    nan_mask = ~np.isnan(y) & ~np.isinf(y) & ~np.isnan(x) & ~np.isinf(x)
    return make_interp_spline(x[nan_mask], y[nan_mask])


def func(sum_of_exp, c, a):
    return c * sum_of_exp + a


def get_exp(dist, t_admix):
    return np.exp(-t_admix * dist / 100)


def transform_admixtime(t_admix):
    return 1 + t_admix**2


def transform_weight(weight):
    return 1e-8 + np.exp(weight) / (1 + np.exp(weight))


def get_sum_of_exp(dist, t_admix1, t_admix2, weight):
    sum_of_exp1 = get_exp(dist, t_admix1)
    sum_of_exp2 = get_exp(dist, t_admix2)
    sum_of_exp = weight * sum_of_exp1 + (1 - weight) * sum_of_exp2
    return sum_of_exp


def get_sum_of_exp_continuous(dist, t_admix1, t_admix2, mu):
    t_admix1 = int(t_admix1)
    t_admix2 = int(t_admix2)

    if t_admix1 > t_admix2:
        return np.zeros(len(dist))
    j_vals = np.arange(t_admix1, t_admix2 + 1)
    power_term = (1 - mu) ** (t_admix2 - j_vals)
    normalization_factor = 1 - (1 - mu) ** (t_admix2 - t_admix1 + 1)
    weights = mu * power_term / normalization_factor
    weights = weights / np.sum(weights)
    if np.isnan(weights).any():
        return np.zeros(len(dist))
    exp_terms = np.exp(-np.outer(j_vals, dist) / 100)
    output = np.dot(weights, exp_terms)
    return output


def plot_ld_curve_comp1(
    dist,
    means_all,
    output_prefix,
    negloglike=None,
    t_admix1=None,
    t_admix2=None,
    weight=None,
    admixtimes=None,
    mode="1date",
):
    num_sam = len(means_all)
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    for sam in range(num_sam):
        spl = make_interp_spline_(dist, means_all[sam][0, 0])
        ax.plot(dist, spl(dist), color="gray", alpha=0.3, linewidth=0.5)
    mean_of_all_sam = np.mean(means_all, axis=0)
    spl = make_interp_spline_(dist, mean_of_all_sam[0, 0])
    ax.plot(dist, spl(dist), color="black", alpha=1, linewidth=1)
    if admixtimes is not None:
        # Single-date fit
        sum_of_exp = get_exp(dist, admixtimes)
        nan_mask = ~np.isnan(mean_of_all_sam[0, 0]) & ~np.isinf(mean_of_all_sam[0, 0])
        popt, pcov = curve_fit(
            func, sum_of_exp[nan_mask], mean_of_all_sam[0, 0][nan_mask], maxfev=5000
        )
        ax.plot(dist, func(sum_of_exp, *popt), "--", color="green", linewidth=1)
        ax.text(
            0.4,
            0.8,
            "{:.1f} gens.".format(admixtimes),
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            color="green",
        )
        ax.set_title("One-date fit, logl = {:.2f}".format(-negloglike), fontsize=16)
    elif t_admix1 is not None and t_admix2 is not None and weight is not None:
        # Two-date fit with weight
        if mode == "2date":
            sum_of_exp = get_sum_of_exp(dist, t_admix1, t_admix2, weight)
        elif mode == "continuous":
            sum_of_exp = get_sum_of_exp_continuous(dist, t_admix1, t_admix2, weight)
        nan_mask = ~np.isnan(mean_of_all_sam[0, 0]) & ~np.isinf(mean_of_all_sam[0, 0])
        popt, pcov = curve_fit(
            func, sum_of_exp[nan_mask], mean_of_all_sam[0, 0][nan_mask], maxfev=5000
        )
        ax.plot(dist, func(sum_of_exp, *popt), "--", color="green", linewidth=1)

        ax.text(
            0.3,
            0.8,
            r"$\lambda_1 = {:.1f}$".format(t_admix1),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            color="green",
        )
        ax.text(
            0.3,
            0.75,
            r"$\lambda_2 = {:.1f}$".format(t_admix2),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            color="green",
        )
        if mode == "2date":
            ax.text(
                0.3,
                0.7,
                r"$\mathrm{{weight}}_1 = {:.2f}$".format(weight),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                color="green",
            )
            ax.text(
                0.3,
                0.65,
                r"$\mathrm{{weight}}_2 = {:.2f}$".format(1 - weight),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                color="green",
            )
            ax.set_title("Two-date fit, logl = {:.2f}".format(-negloglike), fontsize=16)
        else:
            ax.text(
                0.3,
                0.7,
                r"$\mu = {:.1e}$".format(weight),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                color="green",
            )
            ax.set_title(
                "Continuous admix. fit, logl = {:.2f}".format(-negloglike), fontsize=16
            )
    else:
        raise ValueError(
            "Invalid input: provide either 'admixtimes' for single-date or 't_admix1', 't_admix2', and 'weight' for two-date."
        )
    ax.set_xlabel("Genetic distance (cM)")
    ax.set_ylabel("Relative probability")
    plt.tight_layout()
    plt.savefig(output_prefix + "ld_curve_comp1.svg", dpi=300, transparent=True)
    plt.show()


def plot_ld_curves(
    dist,
    means_all,
    output_prefix,
    negloglike=None,
    t_admix1=None,
    t_admix2=None,
    weight=None,
    admixtimes=None,
    refit=False,
    mode="1date",
):
    num_sam = len(means_all)
    num_clusters = len(means_all[0])
    fig, ax = plt.subplots(num_clusters, num_clusters, figsize=(13, 13), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for sam in range(num_sam):
        for i in range(num_clusters):
            for j in range(num_clusters):
                spl = make_interp_spline_(dist, means_all[sam][i, j])
                ax[i, j].plot(dist, spl(dist), color="gray", alpha=0.3, linewidth=0.5)
    mean_of_all_sam = np.mean(means_all, axis=0)
    for i in range(num_clusters):
        for j in range(num_clusters):
            spl = make_interp_spline_(dist, mean_of_all_sam[i, j])
            ax[i, j].plot(dist, spl(dist), color="black", alpha=1, linewidth=1)
            if refit:
                if t_admix1 is not None and t_admix2 is not None and weight is not None:
                    (t_admix1, t_admix2, weight), _ = get_admixtimes(
                        dist, np.array(means_all)[:, i : i + 1, j : j + 1], mode="2date"
                    )
                elif admixtimes is not None:
                    admixtimes, _ = get_admixtimes(
                        dist, np.array(means_all)[:, i : i + 1, j : j + 1], mode="1date"
                    )
            if admixtimes is not None:
                sum_of_exp = get_exp(dist, admixtimes)
                nan_mask = ~np.isnan(mean_of_all_sam[i, j]) & ~np.isinf(
                    mean_of_all_sam[i, j]
                )
                popt, pcov = curve_fit(
                    func,
                    sum_of_exp[nan_mask],
                    mean_of_all_sam[i, j][nan_mask],
                    maxfev=5000,
                )
                ax[i, j].plot(
                    dist, func(sum_of_exp, *popt), "--", color="green", linewidth=1
                )
                if refit:
                    ax[i, j].text(
                        0.35,
                        0.5,
                        "{0:.1f} gens".format(admixtimes),
                        transform=ax[i, j].transAxes,
                        fontsize=14,
                        verticalalignment="bottom",
                    )
            elif t_admix1 is not None and t_admix2 is not None and weight is not None:
                if mode == "2date":
                    sum_of_exp = get_sum_of_exp(dist, t_admix1, t_admix2, weight)
                elif mode == "continuous":
                    sum_of_exp = get_sum_of_exp_continuous(
                        dist, t_admix1, t_admix2, weight
                    )
                nan_mask = ~np.isnan(mean_of_all_sam[i, j]) & ~np.isinf(
                    mean_of_all_sam[i, j]
                )
                popt, pcov = curve_fit(
                    func,
                    sum_of_exp[nan_mask],
                    mean_of_all_sam[i, j][nan_mask],
                    maxfev=5000,
                )
                ax[i, j].plot(
                    dist, func(sum_of_exp, *popt), "--", color="green", linewidth=1
                )
                if refit:
                    ax[i, j].text(
                        0.35,
                        0.5,
                        "{:.1f}, {:.1f}, {:.2f}, {:.2f}".format(
                            t_admix1, t_admix2, weight, 1 - weight
                        ),
                        transform=ax[i, j].transAxes,
                        fontsize=14,
                        verticalalignment="bottom",
                    )
    fig.text(0.5, 0.04, "Genetic distance (cM)", ha="center", va="center")
    if admixtimes is not None:
        fig.suptitle(
            "Co-ancestry curve for one-date fit, log(MSE) = {:.2f}".format(negloglike),
            fontsize=18,
        )
    elif t_admix1 is not None and t_admix2 is not None and weight is not None:
        fig.suptitle(
            "Co-ancestry curve for two-date fit, log(MSE) = {:.2f}".format(negloglike),
            fontsize=18,
        )
    plt.savefig(output_prefix + "ld_curve.pdf", dpi=300)
    plt.show()
    if not refit:
        if admixtimes is not None:
            plot_ld_curve_comp1(
                dist,
                means_all,
                output_prefix,
                admixtimes=admixtimes,
                negloglike=negloglike,
                mode=mode,
            )
        else:
            plot_ld_curve_comp1(
                dist,
                means_all,
                output_prefix,
                t_admix1=t_admix1,
                t_admix2=t_admix2,
                weight=weight,
                negloglike=negloglike,
                mode=mode,
            )


def get_admixtime_lhood(params, dist, means_all, mode="1date"):
    if mode == "1date":
        admixtimes = transform_admixtime(params)
        sum_of_exp = get_exp(dist, admixtimes)
    elif mode == "2date" or mode == "continuous":
        t_admix1, t_admix2, weight = params
        t_admix1 = transform_admixtime(t_admix1)
        t_admix2 = transform_admixtime(t_admix2)
        weight = transform_weight(weight)
        if mode == "2date":
            sum_of_exp = get_sum_of_exp(dist, t_admix1, t_admix2, weight)
        else:
            sum_of_exp = get_sum_of_exp_continuous(dist, t_admix1, t_admix2, weight)
    else:
        raise ValueError("Invalid mode. Choose either '1 date' or '2 date'.")
    lhood = 0
    for means in means_all:
        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                nan_mask = ~np.isnan(means[i, j]) & ~np.isinf(means[i, j])
                sum_of_exp_ = sum_of_exp[nan_mask]
                means_i_j = means[i, j][nan_mask]
                popt, pcov = curve_fit(func, sum_of_exp_, means_i_j, maxfev=5000)
                output = func(sum_of_exp_, *popt)
                res = (means_i_j - output) ** 2
                lhood += len(res) * (np.log(np.mean(res)) + 1 + np.log(2 * np.pi)) / 2
    return lhood


def get_admixtimes(dist, means, mode="1date"):
    res_x_min = None
    res_fun_min = np.inf
    if mode == "1date":
        for _ in range(10):
            initial_guess = np.sqrt(
                np.power(10, (np.random.uniform(np.log10(20), np.log10(2000))))
            )
            res = minimize(
                get_admixtime_lhood,
                initial_guess,
                method="Nelder-Mead",
                args=(dist, means, mode),
            )
            if res.fun < res_fun_min:
                res_x_min = res.x[0]
                res_fun_min = res.fun
        res_x_min = transform_admixtime(res_x_min)
    elif mode == "2date" or mode == "continuous":
        for _ in range(10):
            t_admix1 = np.sqrt(
                np.power(10, (np.random.uniform(np.log10(20), np.log10(2000))))
            )
            t_admix2 = t_admix1 + np.sqrt(
                np.power(10, np.random.uniform(np.log10(20), np.log10(2000)))
            )
            weight = np.random.uniform(-5, 5)
            initial_guess = [t_admix1, t_admix2, weight]
            res = minimize(
                get_admixtime_lhood,
                initial_guess,
                method="Nelder-Mead",
                args=(dist, means, mode),
            )
            if res.fun < res_fun_min:
                res_x_min = res.x
                res_fun_min = res.fun
        res_x_min[0] = transform_admixtime(res_x_min[0])
        res_x_min[1] = transform_admixtime(res_x_min[1])
        res_x_min[2] = transform_weight(res_x_min[2])
    else:
        raise ValueError(
            "Invalid mode. Choose either '1 date' or '2 date' or 'continuous'."
        )
    return res_x_min, res_fun_min


def simulate_local_ancestry_markov(
    output, generations, n_samples=10, total_length_cm=250, bin_size_cm=0.05, a=0.5
):
    ### Simulates local ancestry under the model
    ### Used for testing if the dating works or not
    num_bins = int(total_length_cm / bin_size_cm)
    chrom_positions = np.arange(0, total_length_cm, bin_size_cm)
    """
    # Random genomic position grid
    random_increments = np.random.uniform(low=0, high=2*bin_size_cm, size=num_bins)
    chrom_positions = np.cumsum(random_increments)
    chrom_positions = chrom_positions[chrom_positions <= total_length_cm]
    """
    for sample in range(n_samples):
        local_ancestry = np.zeros(num_bins)
        current_ancestry = 0 if np.random.rand() < a else 1
        local_ancestry[0] = current_ancestry
        for i in range(1, num_bins):
            r = (chrom_positions[i] - chrom_positions[i - 1]) * generations / 100.0
            if current_ancestry == 0:
                P_same = np.exp(-r) + a * (1 - np.exp(-r))
                if np.random.rand() < P_same:
                    current_ancestry_next = 0
                else:
                    current_ancestry_next = 1
            else:
                P_same = (1 - a) * (1 - np.exp(-r)) + np.exp(-r)
                if np.random.rand() < P_same:
                    current_ancestry_next = 1
                else:
                    current_ancestry_next = 0
            local_ancestry[i] = current_ancestry_next
            current_ancestry = current_ancestry_next
        prob_0 = 1 - local_ancestry
        prob_1 = local_ancestry
        df_sample = pd.DataFrame(
            {
                "chr": [1] * num_bins,
                "pos": 0.01 * chrom_positions,  # Physical positions (in M)
                "genpos": 0.01 * chrom_positions,
                "prob_0": prob_0,
                "prob_1": prob_1,
            }
        )
        mask = np.random.rand(len(df_sample)) > 0.8
        df_sample.loc[mask, ["prob_0", "prob_1"]] = np.nan
        output_file = f"{output}_overall_membership_{sample + 1}_sample_id_{sample}.csv"
        df_sample.to_csv(output_file, sep="\t", index=False)
        print(f"Sample {sample + 1} saved to {output_file}")


def get_coancestry_per_sample(
    df_hap1, bin_size, bin_max, bin_min, num_clusters, prob_labels
):
    num_bins = int(bin_max / bin_size) - int(bin_min / bin_size)
    means_num = np.zeros((num_bins, num_clusters, num_clusters))
    means_denom = np.zeros((num_bins, num_clusters, num_clusters))
    assert len(np.unique(df_hap1.chr)) == 1
    for chr in np.unique(df_hap1.chr):
        df_chr = df_hap1[df_hap1.chr == chr]
        f = interpolate.interp1d(
            df_chr.genpos.values, df_chr[prob_labels].values, axis=0, kind="nearest"
        )
        interp_genpos = np.arange(
            df_chr.genpos.values[0], df_chr.genpos.values[-1], bin_size
        )
        prob_values = f(interp_genpos)

        genpos_diffs = np.diff(df_chr.genpos.values)
        interp_genpos_nearest = np.searchsorted(df_chr.genpos.values, interp_genpos)
        too_far_mask = (
            np.abs(df_chr.genpos.values[interp_genpos_nearest] - interp_genpos)
            > bin_size
        )
        prob_values[too_far_mask] = np.nan
        props = np.nansum(prob_values, axis=0)
        lens = np.sum(~np.isnan(prob_values[:, 0]))
        for count, bin in enumerate(
            np.arange(int(bin_min / bin_size), int(bin_max / bin_size), 1)
        ):
            for cluster1 in range(num_clusters):
                for cluster2 in range(num_clusters):
                    if bin == 0:
                        cross = prob_values[:, cluster1] * prob_values[:, cluster2]
                    else:
                        cross = (
                            prob_values[0:-bin, cluster1] * prob_values[bin:, cluster2]
                        )
                    means_num[count, cluster1, cluster2] += np.nansum(cross)
                    means_denom[count, cluster1, cluster2] += np.sum(~np.isnan(cross))

    means_num = np.array(means_num).transpose(1, 2, 0)
    means_denom = np.array(means_denom).transpose(1, 2, 0)
    dist = np.arange(
        int(bin_min / bin_size) * bin_size, int(bin_max / bin_size) * bin_size, bin_size
    )
    return means_num, means_denom, dist, props, lens


def get_coancestry_per_pair_sample(
    df_hap1, df_hap2, bin_size, bin_max, bin_min, num_clusters, prob_labels
):
    assert len(df_hap1) == len(df_hap2)
    assert np.allclose(df_hap1.genpos, df_hap2.genpos)
    num_bins = int(bin_max / bin_size) - int(bin_min / bin_size)
    means_num = np.zeros((num_bins, num_clusters, num_clusters))
    means_denom = np.zeros((num_bins, num_clusters, num_clusters))
    assert len(np.unique(df_hap1.chr)) == 1
    for chr in np.unique(df_hap1.chr):
        df_chr1 = df_hap1[df_hap1.chr == chr]
        f1 = interpolate.interp1d(
            df_chr1.genpos.values, df_chr1[prob_labels].values, axis=0, kind="nearest"
        )
        interp_genpos1 = np.arange(
            df_chr1.genpos.values[0], df_chr1.genpos.values[-1], bin_size
        )
        prob_values1 = f1(interp_genpos1)

        df_chr2 = df_hap2[df_hap2.chr == chr]
        f2 = interpolate.interp1d(
            df_chr2.genpos.values, df_chr2[prob_labels].values, axis=0, kind="nearest"
        )
        interp_genpos2 = np.arange(
            df_chr2.genpos.values[0], df_chr2.genpos.values[-1], bin_size
        )
        prob_values2 = f2(interp_genpos2)

        genpos_diffs = np.diff(df_chr1.genpos.values)
        interp_genpos_nearest = np.searchsorted(df_chr1.genpos.values, interp_genpos1)
        too_far_mask = (
            np.abs(df_chr1.genpos.values[interp_genpos_nearest] - interp_genpos1)
            > bin_size
        )
        prob_values1[too_far_mask] = np.nan
        prob_values2[too_far_mask] = np.nan
        props = np.nansum(prob_values1, axis=0)
        lens = np.sum(~np.isnan(prob_values1[:, 0]))
        for count, bin in enumerate(
            np.arange(int(bin_min / bin_size), int(bin_max / bin_size), 1)
        ):
            for cluster1 in range(num_clusters):
                for cluster2 in range(num_clusters):
                    if bin == 0:
                        cross = prob_values1[:, cluster1] * prob_values2[:, cluster2]
                    else:
                        cross = (
                            prob_values1[0:-bin, cluster1]
                            * prob_values2[bin:, cluster2]
                        )
                    means_num[count, cluster1, cluster2] += np.nansum(cross)
                    means_denom[count, cluster1, cluster2] += np.sum(~np.isnan(cross))

    means_num = np.array(means_num).transpose(1, 2, 0)
    means_denom = np.array(means_denom).transpose(1, 2, 0)
    dist = np.arange(
        int(bin_min / bin_size) * bin_size, int(bin_max / bin_size) * bin_size, bin_size
    )
    return means_num, means_denom, dist, props, lens


def read_df_per_sam(df, len_all_cumsum, sam):
    if sam >= 1:
        df_sam = df.iloc[len_all_cumsum[sam - 1] : len_all_cumsum[sam]]
    else:
        df_sam = df.iloc[0 : len_all_cumsum[0]]
    return df_sam


def get_means_whole_genome(
    jn_block, jn_blocks, cent_telo_hla, output_prefix, bin_size, bin_max, bin_min
):
    num_hap = 0
    len_all = []
    file_list = glob.glob(output_prefix + "_overall_membership_*.csv")
    sorted_hap_no = []
    for file in file_list:
        hap_no = int(
            file.split(output_prefix + "_overall_membership_")[1]
            .split(".csv")[0]
            .split("sample_id_")[1]
        )
        sorted_hap_no.append(hap_no)
    num_hap += len(sorted_hap_no)
    df = pd.DataFrame()
    for hap_no in np.sort(sorted_hap_no):
        file = glob.glob(
            output_prefix + "_overall_membership_*" + str(hap_no) + ".csv"
        )[0]
        dfc = pd.read_csv(file, sep="\t")
        dfc["genpos"] = 100 * dfc["genpos"]  ## converting M to cM
        dfc = dfc.drop_duplicates(subset=["genpos", "chr"])
        dfc = dfc.sort_values(by=["chr", "genpos"])
        ## remove jack-knife block (removing 5% from each chromosome)
        for chr in dfc.chr.unique():
            dfc.loc[dfc.chr == chr, "block"] = pd.cut(
                dfc[dfc.chr == chr].pos, bins=jn_blocks, labels=False
            )
            dfc.loc[
                (dfc.chr == chr) & (dfc.block == jn_block),
                ["prob_" + str(i) for i in range(dfc.shape[1] - 4)],
            ] = np.nan
            # dfc = dfc.drop(dfc[(dfc.chr == chr) & (dfc.block == jn_block)].index)
        ## remove cent_telo_hla
        if cent_telo_hla is not None:
            for chr in np.unique(dfc.chr):
                for start, end in zip(
                    cent_telo_hla[cent_telo_hla.chr == str(chr)].start,
                    cent_telo_hla[cent_telo_hla.chr == str(chr)].end,
                ):
                    dfc.loc[
                        (dfc.chr == chr)
                        & (dfc.pos >= (start - 5e5))
                        & (dfc.pos <= (end + 5e5)),
                        ["prob_" + str(i) for i in range(dfc.shape[1] - 4)],
                    ] = np.nan
        len_all.append(len(dfc))
        df = pd.concat([df, dfc], axis=0)
    num_clusters = df.shape[1] - 4
    prob_labels = ["prob_" + str(i) for i in range(num_clusters)]
    len_all_cumsum = np.cumsum(len_all)
    if args.haploid:
        sam_range = range(0, num_hap, 1)
    else:
        sam_range = range(0, num_hap, 2)

    ## First calculate the normalization
    if num_hap > 3 and args.normalize:
        means_whole_genome_num_norm = np.zeros(
            (
                num_clusters,
                num_clusters,
                int(bin_max / bin_size) - int(bin_min / bin_size),
            )
        )
        means_whole_genome_denom_norm = np.zeros(
            (
                num_clusters,
                num_clusters,
                int(bin_max / bin_size) - int(bin_min / bin_size),
            )
        )
        ### only calculate normalization based on the first 100 samples
        for sam in tqdm(sam_range[0:100]):
            for sam2 in sam_range[0:100]:
                if sam - sam2 > 1 or sam2 - sam > 1:
                    df_sam_hap1 = read_df_per_sam(df, len_all_cumsum, sam)
                    df_sam2_hap1 = read_df_per_sam(df, len_all_cumsum, sam2)
                    if args.haploid:
                        df_sam_hap2 = df_sam_hap1
                        df_sam2_hap2 = df_sam2_hap1
                    else:
                        df_sam_hap2 = read_df_per_sam(df, len_all_cumsum, sam + 1)
                        df_sam2_hap2 = read_df_per_sam(df, len_all_cumsum, sam2 + 1)
                    for prob_col in prob_labels:
                        df_sam_hap1.loc[:, prob_col] = (
                            df_sam_hap1[prob_col] + df_sam_hap2[prob_col]
                        ) / 2
                        df_sam2_hap1.loc[:, prob_col] = (
                            df_sam2_hap1[prob_col] + df_sam2_hap2[prob_col]
                        ) / 2
                    for chr in np.unique(df_sam_hap1.chr):
                        if (
                            np.mean(df_sam_hap1[df_sam_hap1["chr"] == chr][prob_labels])
                            == 0
                        ).any() or (
                            np.mean(
                                df_sam_hap1[df_sam2_hap1["chr"] == chr][prob_labels]
                            )
                            == 1
                        ).any():
                            continue
                        means_num, means_denom, dist, props_num, props_denom = (
                            get_coancestry_per_pair_sample(
                                df_sam_hap1[df_sam_hap1["chr"] == chr],
                                df_sam2_hap1[df_sam2_hap1["chr"] == chr],
                                bin_size,
                                bin_max,
                                bin_min,
                                num_clusters,
                                prob_labels,
                            )
                        )
                        means_whole_genome_num_norm += means_num
                        means_whole_genome_denom_norm += means_denom
    else:
        means_whole_genome_num_norm = np.ones(
            (
                num_clusters,
                num_clusters,
                int(bin_max / bin_size) - int(bin_min / bin_size),
            )
        )
        means_whole_genome_denom_norm = np.ones(
            (
                num_clusters,
                num_clusters,
                int(bin_max / bin_size) - int(bin_min / bin_size),
            )
        )

    means_whole_genome_norm = (
        means_whole_genome_num_norm / means_whole_genome_denom_norm
    )
    means_whole_genome_num = np.zeros(
        (num_clusters, num_clusters, int(bin_max / bin_size) - int(bin_min / bin_size))
    )
    means_whole_genome_denom = np.zeros(
        (num_clusters, num_clusters, int(bin_max / bin_size) - int(bin_min / bin_size))
    )
    props_whole_genome_num = np.zeros(num_clusters)
    props_whole_genome_denom = np.zeros(num_clusters)
    for sam in sam_range:
        df_sam_hap1 = read_df_per_sam(df, len_all_cumsum, sam)
        if args.haploid:
            df_sam_hap2 = df_sam_hap1
        else:
            df_sam_hap2 = read_df_per_sam(df, len_all_cumsum, sam + 1)
        for prob_col in prob_labels:
            df_sam_hap1.loc[:, prob_col] = (
                df_sam_hap1[prob_col] + df_sam_hap2[prob_col]
            ) / 2
        for chr in np.unique(df_sam_hap1.chr):
            if (
                np.mean(df_sam_hap1[df_sam_hap1["chr"] == chr][prob_labels]) == 0
            ).any():
                continue
            means_num, means_denom, dist, props_num, props_denom = (
                get_coancestry_per_sample(
                    df_sam_hap1[df_sam_hap1["chr"] == chr],
                    bin_size,
                    bin_max,
                    bin_min,
                    num_clusters,
                    prob_labels,
                )
            )
            means_whole_genome_num += means_num
            means_whole_genome_denom += means_denom
            props_whole_genome_num += props_num
            props_whole_genome_denom += props_denom
    means_whole_genome = means_whole_genome_num / means_whole_genome_denom
    means_whole_genome = means_whole_genome / means_whole_genome_norm
    props_whole_genome = props_whole_genome_num / props_whole_genome_denom
    if num_hap <= 3 or not args.normalize:
        for i in range(means_whole_genome.shape[0]):
            for j in range(means_whole_genome.shape[1]):
                means_whole_genome[i, j] = means_whole_genome[i, j] / (
                    props_whole_genome[i] * props_whole_genome[j]
                )
    return means_whole_genome, dist


def run_ld_curve_dating(args):
    bin_size = args.bin_size
    bin_max = args.bin_max
    bin_min = args.bin_min
    output_prefix = args.output
    mode = args.mode
    jn_blocks = 20

    means_all_blocks = []
    admixture_params_all_blocks = []
    if args.genome_build is not None:
        cent_telo_hla = pd.read_csv(
            os.path.join(MASK_DIR, str(args.genome_build) + "_real_data_mask.txt"),
            sep="\t",
        )
    else:
        cent_telo_hla = None
    for jn_block in range(jn_blocks):  #### caution!!
        means_whole_genome, dist = get_means_whole_genome(
            jn_block,
            jn_blocks,
            cent_telo_hla,
            output_prefix,
            bin_size,
            bin_max,
            bin_min,
        )
        if mode == "1date":
            t_admix1, _ = get_admixtimes(dist, [means_whole_genome], mode=mode)
            print("Block: " + str(jn_block) + ", t_admix1: " + str(t_admix1))
            admixture_params_all_blocks.append(t_admix1)
        elif mode == "2date" or mode == "continuous":
            (t_admix1, t_admix2, weight), _ = get_admixtimes(
                dist, [means_whole_genome], mode=mode
            )
            admixture_params_all_blocks.append((t_admix1, t_admix2, weight))
            if mode == "2date":
                print(
                    "Block: "
                    + str(jn_block)
                    + ", t_admix1: "
                    + str(t_admix1)
                    + " t_admix2: "
                    + str(t_admix2)
                    + ", weight: "
                    + str(weight)
                )
            else:
                print(
                    "Block: "
                    + str(jn_block)
                    + ", t_admix1: "
                    + str(t_admix1)
                    + " t_admix2: "
                    + str(t_admix2)
                    + ", mu: "
                    + str(weight)
                )
        means_all_blocks.append(means_whole_genome)

    if mode == "1date":
        t_admix1, negloglike = get_admixtimes(dist, means_all_blocks, mode=mode)
        print(
            "Overall, t_admix1: " + str(t_admix1) + ", negloglike: " + str(negloglike)
        )
        plot_ld_curves(
            dist,
            means_all_blocks,
            negloglike=negloglike,
            admixtimes=t_admix1,
            output_prefix=output_prefix + "_" + mode,
            refit=False,
            mode=mode,
        )
        plot_ld_curves(
            dist,
            means_all_blocks,
            negloglike=negloglike,
            admixtimes=t_admix1,
            output_prefix=output_prefix + "_" + mode + "_refit",
            refit=True,
            mode=mode,
        )
    elif mode == "2date" or mode == "continuous":
        (t_admix1, t_admix2, weight), negloglike = get_admixtimes(
            dist, means_all_blocks, mode=mode
        )
        plot_ld_curves(
            dist,
            means_all_blocks,
            negloglike=negloglike,
            t_admix1=t_admix1,
            t_admix2=t_admix2,
            weight=weight,
            output_prefix=output_prefix + "_" + mode,
            refit=False,
            mode=mode,
        )
        plot_ld_curves(
            dist,
            means_all_blocks,
            negloglike=negloglike,
            t_admix1=t_admix1,
            t_admix2=t_admix2,
            weight=weight,
            output_prefix=output_prefix + "_" + mode + "_refit",
            refit=True,
            mode=mode,
        )
        if mode == "2date":
            print(
                "Overall, t_admix1: "
                + str(t_admix1)
                + " t_admix2: "
                + str(t_admix2)
                + ", weight: "
                + str(weight)
                + ", negloglike: "
                + str(negloglike)
            )
        else:
            print(
                "Overall, t_admix1: "
                + str(t_admix1)
                + " t_admix2: "
                + str(t_admix2)
                + ", mu: "
                + str(weight)
                + ", negloglike: "
                + str(negloglike)
            )


def run_get_segment_length(args):
    bin_size = args.bin_size
    bin_max = args.bin_max
    bin_min = 0
    output_prefix = args.output
    jn_blocks = 20
    if args.genome_build is not None:
        cent_telo_hla = pd.read_csv(
            os.path.join(MASK_DIR, str(args.genome_build) + "_real_data_mask.txt"),
            sep="\t",
        )
    else:
        cent_telo_hla = None

    means_whole_genome, dist = get_means_whole_genome(
        jn_blocks + 1,
        jn_blocks,
        cent_telo_hla,
        output_prefix,
        bin_size,
        bin_max,
        bin_min,
    )  ## no removing of blocks
    for comp in range(means_whole_genome.shape[0]):
        co_ancestry = means_whole_genome[comp, comp]
        pdf = -np.diff(co_ancestry)
        # pdf = np.maximum(pdf,0)
        pdf = pdf / np.sum(pdf)
        df = pd.DataFrame(
            {"dist_start": dist[:-1], "dist_end": dist[1:], "number_segments": pdf}
        )
        mean_segment_length = np.sum(pdf * (dist[:-1] + dist[1:]) / 2)
        print(
            "Mean segment length for component "
            + str(comp)
            + ": "
            + str(mean_segment_length)
        )
        df.to_csv(
            output_prefix + "_segment_length_comp" + str(comp) + ".csv",
            sep="\t",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin_size", help="Bin size in ld curves (in cM)", default=0.05, type=float
    )
    parser.add_argument(
        "--bin_max",
        help="Maximum bin location in ld curves (in cM)",
        default=10,
        type=float,
    )
    parser.add_argument(
        "--bin_min",
        help="Maximum bin location in ld curves (in cM)",
        default=0.02,
        type=float,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output prefix (same as what was used when calling ghost_catcher.py)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--genome_build",
        help="Which genome build to use for filtering centromere/telomere/hla (hg38/hg37/None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test",
        help="Only for debugging purpose, test with simulated local ancestry",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "--mode",
        help="Single-date or two-date fit",
        type=str,
        default=None,
        choices=["1date", "2date", "continuous"],
    )
    parser.add_argument(
        "--normalize",
        help="Perform pseudo individual based normalization of coancestry curves",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "--haploid", help="Use haploid data", type=boolean, default=False
    )

    args = parser.parse_args()
    print(args)
    # run_get_segment_length(args)
    if args.test:
        print("Simulating local ancestry under the model...")
        simulate_local_ancestry_markov(
            args.output,
            generations=500,
            n_samples=10,
            total_length_cm=250,
            bin_size_cm=args.bin_size,
            a=0.05,
        )
    if args.mode is None:
        args.mode = "1date"
        print("Running one-date fit...")
        run_ld_curve_dating(args)
        args.mode = "2date"
        print("Running two-date fit...")
        run_ld_curve_dating(args)
        args.mode = "continuous"
        print("Running continuous admixture fit...")
        run_ld_curve_dating(args)
    else:
        print("Running " + args.mode + " fit...")
        run_ld_curve_dating(args)

    # python -W ignore RelateLocalAncestry/ld_curve_dating.py --output ground_truth_genpos
