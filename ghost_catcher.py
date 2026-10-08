import pandas as pd
import numpy as np
import math
import time
import tskit
import argparse
import copy
from tqdm import tqdm
import random
from functools import partial
from scipy.stats import hmean
import scipy.stats as stats
from joblib import Parallel, delayed
import gc
import pdb
import warnings
from numba import jit
import numba as nb
import glob
import os
import msprime
import bisect
import statsmodels.api as sm

from src.calc_tree_stats import load_tree_stats
from src.calc_fixed_params import load_fixed_params
from src.utils import (
    filter_recomb_rate,
    load_mask_csv,
    write_coal,
    boolean,
    compute_gamma_num,
    compute_gamma_denom,
    load_gamma,
    load_props,
    load_tadmix,
    get_target_branch_length,
)
from src.hmm_decode import Decode_grid

warnings.filterwarnings("ignore")


def load_trees(args):
    chrs = list(map(int, args.chrs.split(",")))
    ts_list = []
    if args.trees != None:
        for chr in chrs:
            ts = tskit.load(args.trees + str(chr) + ".trees")  ## relate trees
            ts_list.append(ts)

    return ts_list


def update_membership_eventwise_denom(
    gamma_arr, denom, ignore_first_epoch, ignore_last_epoch, n_epochs
):
    log_denom_em = np.zeros((gamma_arr.shape[0], denom.shape[0]), dtype="float64")
    start_end_mask = np.ones(n_epochs - 1, dtype=bool)
    start_end_mask[0] = False if ignore_first_epoch else True
    start_end_mask[-1] = False if ignore_last_epoch else True
    for j in range(gamma_arr.shape[0]):
        ### Caution: This hack removes epochs which have any one reference population without coal. rate
        ### We instead want to remove the reference population not the entire epoch
        nan_mask = np.isnan(gamma_arr[j]).sum(axis=0) == 0
        combined_mask = np.logical_and(start_end_mask, nan_mask)
        gamma_arr_nan_removed = gamma_arr[j][:, combined_mask]
        denom_nan_removed = denom[:, :, combined_mask]
        log_denom_em[j] = -np.sum(
            gamma_arr_nan_removed * denom_nan_removed, axis=(1, 2)
        )
    return log_denom_em


@jit(nopython=True, fastmath=True)
def update_membership_eventwise_numba(
    proportion_of_coalescing_all,
    epoch_index_all,
    gamma_arr,
    ignore_first_epoch,
    ignore_last_epoch,
    n_epochs,
    target_branch_length,
    num_clusters,
):
    log_num_em = np.zeros(
        (num_clusters, len(proportion_of_coalescing_all)), dtype="float64"
    )
    for n_site in range(len(proportion_of_coalescing_all)):
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[n_site]
        epoch_index_in_tree = epoch_index_all[n_site]
        target_branch_length_in_tree = target_branch_length[n_site]
        for j in range(num_clusters):
            log_num_em_j = 0.0
            for i in range(len(proportion_of_coalescing_in_tree)):
                if (not ignore_first_epoch) or epoch_index_in_tree[i] >= 1:
                    if (not ignore_last_epoch) or epoch_index_in_tree[i] < n_epochs - 2:
                        if not (
                            gamma_arr[j][:, epoch_index_in_tree[i]]
                            == -9223372036854775808
                        ).any():
                            log_num_em_j += (
                                np.log(
                                    sum(
                                        gamma_arr[j][:, epoch_index_in_tree[i]]
                                        * proportion_of_coalescing_in_tree[i]
                                    )
                                    / sum(proportion_of_coalescing_in_tree[i]),
                                )
                            ) / target_branch_length_in_tree[i]
            log_num_em[j, n_site] = log_num_em_j
    return log_num_em


def update_membership_eventwise_numpy(
    proportion_of_coalescing_all,
    epoch_index_all,
    gamma_arr,
    ignore_first_epoch,
    ignore_last_epoch,
    n_epochs,
    target_branch_length,
    num_clusters,
):
    log_num_em = np.zeros(
        (num_clusters, len(proportion_of_coalescing_all)), dtype="float64"
    )

    for n_site in range(len(proportion_of_coalescing_all)):
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[n_site]
        epoch_index_in_tree = epoch_index_all[n_site]
        target_branch_length_in_tree = target_branch_length[n_site]
        for j in range(num_clusters):
            log_num_em_j = 0.0
            for i in range(len(proportion_of_coalescing_in_tree)):
                if (not ignore_first_epoch) or epoch_index_in_tree[i] >= 1:
                    if (not ignore_last_epoch) or epoch_index_in_tree[i] < n_epochs - 2:
                        if np.isnan(gamma_arr[j][:, epoch_index_in_tree[i]]).any():
                            continue
                        log_num_em_j += (
                            np.log(
                                np.sum(
                                    gamma_arr[j][:, epoch_index_in_tree[i]]
                                    * proportion_of_coalescing_in_tree[i]
                                )
                                / sum(proportion_of_coalescing_in_tree[i]),
                            )
                        ) / target_branch_length_in_tree[i]
            log_num_em[j, n_site] = log_num_em_j
    return log_num_em


def e_m_step(
    args,
    own_membership,  ##now is n_clusters x n_sites
    trans_prop,
    tau,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    denom,
    unique_groups,
    epoch_intervals,
    n_sites,
    n_samples,
    iter,
    target_branch_length_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
):
    st = time.time()
    n_epochs = len(epoch_intervals)
    n_unique_groups = len(unique_groups)

    if args.load_gamma is not None and (not (iter == args.num_iters - 1 and (not args.force_relate_gammas))):
        print("Using initial gamma specified in file: " + str(args.load_gamma))
        gamma_arr = load_gamma(args.load_gamma, args.groups, unique_groups)
        for epoch in range(gamma_arr.shape[2]):
            if (
                args.ignore_first_epoch
                and (args.start_time - math.log(args.ypg, 10)) > epoch_intervals[epoch]
            ):
                gamma_arr[:, :, epoch] = np.nan
            if (
                args.ignore_last_epoch
                and (args.end_time - math.log(args.ypg, 10)) <= epoch_intervals[epoch]
            ):
                gamma_arr[:, :, epoch] = np.nan

    if args.load_gamma is None or (iter == args.num_iters - 1 and (not args.force_relate_gammas)):
        n = np.zeros(
            (args.num_clusters, n_unique_groups, n_epochs - 1),
            dtype="float64",
        )
        d = np.zeros(
            (args.num_clusters, n_unique_groups, n_epochs - 1),
            dtype="float64",
        )
        if prev_gamma is None:
            prev_gamma = np.ones_like(n)
        prev_gamma = np.nan_to_num(prev_gamma, nan=0)

        ### caution: for finding nea ancestry in eurasian segments
        # eurasian_ancestry = np.load(args.output + '_eurasian_ancestry.npy')
        # # eurasian_ancestry = np.load('nea_in_afr_eurasian_segments/bantukenya_eurasian_ancestry.npy')[0:1]
        # eurasian_ancestry = (eurasian_ancestry > 0.5).flatten()
        # for j in range(args.num_clusters):
        #     own_membership[j] = own_membership[j] * eurasian_ancestry

        for sample_no in range(n_samples):
            start = int(np.sum(n_sites[0:sample_no]))
            end = int(np.sum(n_sites[0 : sample_no + 1]))
            own_membership_sample = own_membership[:, start:end]
            for j in range(len(own_membership_sample)):
                n_j = compute_gamma_num(
                    own_membership_sample[j],
                    prev_gamma[j],
                    proportion_of_coalescing_all[sample_no],
                    epoch_index_all[sample_no],
                    n_unique_groups,
                    n_epochs,
                    target_branch_length_masked[sample_no],
                    args.ignore_first_epoch,
                    args.ignore_last_epoch,
                )
                d_j = compute_gamma_denom(
                    own_membership_sample[j],
                    denom[sample_no],
                )
                n[j] += n_j
                d[j] += d_j

        gamma_arr = n / d

        ### run with inferred gamma for the very last iteration
        if args.load_gamma is not None:
            for epoch in range(gamma_arr.shape[2]):
                if (
                    args.ignore_first_epoch
                    and (args.start_time - math.log(args.ypg, 10))
                    > epoch_intervals[epoch]
                ):
                    gamma_arr[:, :, epoch] = np.nan
                if (
                    args.ignore_last_epoch
                    and (args.end_time - math.log(args.ypg, 10))
                    <= epoch_intervals[epoch]
                ):
                    gamma_arr[:, :, epoch] = np.nan

    if args.load_props != None:
        print("Using initial props specified in file: " + str(args.load_props))
        tau = load_props(
            args.load_props
        )  ### load taus only works for not(props_per_chrs)

    # print("M-step time: " + str(time.time() - st))
    st = time.time()

    if args.t_admix_guess is not None:
        trans_prop = load_tadmix(args.t_admix_guess)

    gamma_arr = np.maximum(gamma_arr, 0)
    prev_gamma = copy.deepcopy(gamma_arr)
    # if np.isnan(gamma_arr).any():
    #     print(gamma_arr)

    log_num_em = np.zeros((args.num_clusters, np.sum(n_sites)), dtype="float64")
    log_denom_em = np.zeros((args.num_clusters, np.sum(n_sites)), dtype="float64")
    count_masked_trees = 0

    update_membership_eventwise = update_membership_eventwise_numba
    for sample_no in range(n_samples):
        log_num_em_sam = update_membership_eventwise(
            proportion_of_coalescing_all[sample_no],
            epoch_index_all[sample_no],
            gamma_arr,
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
            target_branch_length_masked[sample_no],
            args.num_clusters,
        )
        log_denom_em_sam = update_membership_eventwise_denom(
            gamma_arr,
            denom[sample_no],
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
        )
        start = int(np.sum(n_sites[0:sample_no]))
        end = int(np.sum(n_sites[0 : sample_no + 1]))
        log_num_em[:, start:end] = log_num_em_sam
        log_denom_em[:, start:end] = log_denom_em_sam
    loglikelihood_per_comp = log_num_em + log_denom_em
    # print("log-like time: " + str(time.time() - st))

    ### HMM smoothing
    st = time.time()
    own_membership_hmm = np.zeros((args.num_clusters, np.sum(n_sites)), dtype="float64")
    for sample_no in range(n_samples):
        start = int(np.sum(n_sites[0:sample_no]))
        end = int(np.sum(n_sites[0 : sample_no + 1]))
        (
            own_membership_sam,
            trans_num_sam,
            trans_denom_sam,
            tau_sam,
            log_likelihood_sam,
        ) = Decode_grid(
            tree_left_bp[sample_no],
            tree_right_bp[sample_no],
            gen_grid_all[sample_no],
            trans_prop,
            loglikelihood_per_comp[:, start:end],
            tau,
            window_size=args.force_build,
        )
        own_membership_hmm[:, start:end] = own_membership_sam
        if sample_no == 0:
            trans_num = trans_num_sam
            trans_denom = trans_denom_sam
            tau_update = tau_sam
            log_likelihood_hmm = log_likelihood_sam
        else:
            trans_num += trans_num_sam
            trans_denom += trans_denom_sam
            tau_update += tau_sam
            log_likelihood_hmm += log_likelihood_sam

    trans_prop = trans_num / trans_denom
    tau = tau_update / np.sum(tau_update)
    tau = np.minimum(np.maximum(tau, 1e-11), 1 - 1e-11)

    print("props: " + str(np.mean(own_membership_hmm, axis=1)))
    print("admix time: " + str(trans_prop))
    # print("HMM time: " + str(time.time() - st))
    if np.isnan(log_likelihood_hmm):
        pdb.set_trace()

    ## when load_gamma this plots coal. rates for all epochs
    if iter == args.num_iters - 1 and (not args.force_relate_gammas):
        gamma_arr = n / d

    return own_membership_hmm, trans_prop, gamma_arr, tau, log_likelihood_hmm


def random_sweep_iter(
    args,
    n_clusters,
    unique_groups,
    n_epochs,
    num_trees_per_sample,
    n_iters,
    ts_list,
    poplabels,
    mask_dodgy,
    epoch_intervals_pow,
    num,
    denom,
    proportion_of_coalescing_all,
    epoch_index_all,
    target_branch_length_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    chr_map,
    n_sites,
):
    epoch_intervals = np.log10(epoch_intervals_pow)

    n_unique_groups = len(unique_groups)
    if args.load_gamma is not None:
        gamma_arr = load_gamma(args.load_gamma, args.groups, unique_groups)
        ## CAUTION: recheck this, removing gamma_arr outside the range
        for epoch in range(gamma_arr.shape[2]):
            if (
                args.ignore_first_epoch
                and (args.start_time - math.log(args.ypg, 10)) > epoch_intervals[epoch]
            ):
                gamma_arr[:, :, epoch] = np.nan
            if (
                args.ignore_last_epoch
                and (args.end_time - math.log(args.ypg, 10)) <= epoch_intervals[epoch]
            ):
                gamma_arr[:, :, epoch] = np.nan
    else:
        gamma_arr = np.power(
            np.e,
            np.random.uniform(
                -16.11, -5.3, (n_clusters, n_unique_groups, n_epochs - 1)
            ),
        )
        gamma_arr = np.array(gamma_arr, dtype="float64")

    if args.load_props is not None:
        tau = load_props(args.load_props)

    else:
        tau = np.random.uniform(0.01, 0.99, n_clusters)
        tau /= np.sum(tau)

    trans_prop = (
        np.power(10, np.random.uniform(np.log10(20), np.log10(2000)))
        if args.t_admix_guess is None
        else load_tadmix(args.t_admix_guess)
    )
    print((tau, trans_prop))
    n_samples = len(args.sample_id)

    own_membership_trial = [[] for _ in range(n_clusters)]
    # update_membership_eventwise = update_membership_eventwise_numpy if np.isnan(gamma_arr).any() else update_membership_eventwise_numba
    update_membership_eventwise = update_membership_eventwise_numba
    for sample_no, sample in enumerate(args.sample_id):
        log_num_em_sam = update_membership_eventwise(
            proportion_of_coalescing_all[sample_no],
            epoch_index_all[sample_no],
            gamma_arr,
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
            target_branch_length_masked[sample_no],
            args.num_clusters,
        )
        log_denom_em_sam = update_membership_eventwise_denom(
            gamma_arr,
            denom[sample_no],
            args.ignore_first_epoch,
            args.ignore_last_epoch,
            n_epochs,
        )
        (
            own_membership_sam,
            trans_num_sam,
            trans_denom_sam,
            tau_sam,
            log_likelihood_sam,
        ) = Decode_grid(
            tree_left_bp[sample_no],
            tree_right_bp[sample_no],
            gen_grid_all[sample_no],
            trans_prop,
            log_num_em_sam + log_denom_em_sam,
            tau,
            window_size=args.force_build,
        )
        for clust_j in range(n_clusters):
            own_membership_trial[clust_j].extend(own_membership_sam[clust_j].tolist())
        if sample_no == 0:
            trans_num = trans_num_sam
            trans_denom = trans_denom_sam
            tau_update = tau_sam
            log_likelihood = log_likelihood_sam
        else:
            trans_num += trans_num_sam
            trans_denom += trans_denom_sam
            tau_update += tau_sam
            log_likelihood += log_likelihood_sam

    trans_prop = trans_num / trans_denom
    tau = tau_update / np.sum(tau_update)
    own_membership_trial = np.array(own_membership_trial, dtype="float64")

    for epoch in range(args.sweep_num_iters):
        own_membership_trial, trans_prop, gamma_arr, tau, log_likelihood = e_m_step(
            args,
            own_membership_trial,
            trans_prop,
            tau,
            gamma_arr,
            proportion_of_coalescing_all,
            epoch_index_all,
            denom,
            unique_groups,
            epoch_intervals,
            n_sites,
            n_samples,
            epoch,
            target_branch_length_masked,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
        )

    if args.load_membership is not None:
        own_membership_trial = np.load(args.load_membership)
        # trans_prop = load_tadmix(args.t_admix_guess)
        # tau = np.load(args.load_props)

    return (
        log_likelihood,
        own_membership_trial,
        trans_prop,
        tau,
        unique_groups,
    )


def random_sweep(
    args,
    n_clusters,
    unique_groups,
    n_epochs,
    num_trees_per_sample,
    n_repeats,
    ts_list,
    poplabels,
    mask_dodgy,
    epoch_intervals,
    target_branch_length_masked,
    mutscale_masked,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    chr_map,
    gt_ref=None,
    exact_pos=None,
):
    print("Performing a random sweep for better initialization")
    best_loglikelihood = -np.inf
    epoch_intervals_pow = np.power(10, epoch_intervals)

    st = time.time()
    num, denom, denom_unscaled, proportion_of_coalescing_all, epoch_index_all = (
        [],
        [],
        [],
        [],
        [],
    )
    for sample_no, sample in enumerate(args.sample_id):
        (
            num1,
            denom1,
            denom1_unscaled,
            proportion_of_coalescing_all1,
            epoch_index_all1,
        ) = load_fixed_params(
            args,
            ts_list,
            sample,
            poplabels,
            mask_dodgy[sample_no],
            chr_map,
            epoch_intervals,
            target_branch_length_masked[sample_no],
            mutscale_masked[sample_no],
            gt_ref,
            unique_groups,
            exact_pos,
        )
        num.append(num1)
        denom.append(denom1)
        denom_unscaled.append(denom1_unscaled)
        proportion_of_coalescing_all.append(proportion_of_coalescing_all1)
        epoch_index_all.append(epoch_index_all1)
    del ts_list
    gc.collect()
    ts_list = []

    print("fixed params:" + str(time.time() - st))
    n_sites = []

    count_matching_mask = 0
    for sample_no in range(len(args.sample_id)):
        if (mask_dodgy[sample_no] == mask_dodgy[0]).all():
            count_matching_mask += 1

    if count_matching_mask == len(args.sample_id):
        effective_number_samples = 1
        print("All samples have the same mask")
    else:
        effective_number_samples = len(args.sample_id)

    for sample_no in range(effective_number_samples):
        n_sites_sam = 0
        chr_map_masked = np.array(chr_map)[mask_dodgy[sample_no]]
        for chr in np.unique(chr_map_masked):
            if exact_pos is not None:
                exact_pos_chr = exact_pos[exact_pos["chr"] == chr]
            for i, (l, r) in enumerate(
                zip(
                    tree_left_bp[sample_no][chr_map_masked == chr],
                    tree_right_bp[sample_no][chr_map_masked == chr],
                )
            ):
                if exact_pos is None:
                    n_sites_sam += int(np.ceil(r / args.force_build)) - int(
                        np.ceil(l / args.force_build)
                    )
                else:
                    n_sites_sam += np.searchsorted(
                        exact_pos_chr["pos"], r
                    ) - np.searchsorted(exact_pos_chr["pos"], l)
        n_sites.append(copy.deepcopy(n_sites_sam))

    if count_matching_mask == len(args.sample_id):
        n_sites = [n_sites[0]] * len(args.sample_id)

    st = time.time()
    out = Parallel(n_jobs=1)(
        delayed(random_sweep_iter)(
            args,
            n_clusters,
            unique_groups,
            n_epochs,
            num_trees_per_sample,
            n_iters,
            ts_list,
            poplabels,
            mask_dodgy,
            epoch_intervals_pow,
            num,
            denom,
            proportion_of_coalescing_all,
            epoch_index_all,
            target_branch_length_masked,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
            chr_map,
            n_sites,
        )
        for n_iters in range(n_repeats)
    )
    print("random sweep: " + str(time.time() - st))
    for i in range(len(out)):
        (
            log_likelihood,
            own_membership_trial,
            trans_prop_trial,
            tau_trial,
            unique_groups,
        ) = out[i]
        if log_likelihood > best_loglikelihood or i == 0:
            best_loglikelihood = log_likelihood
            own_membership = own_membership_trial
            trans_prop = trans_prop_trial
            tau = tau_trial

    return (
        own_membership,
        trans_prop,
        tau,
        num,
        denom,
        denom_unscaled,
        proportion_of_coalescing_all,
        epoch_index_all,
        unique_groups,
        n_sites,
    )


def get_expected_r2(own_membership, num_samples):
    if num_samples % 2 == 0:
        num_clusters = len(own_membership)
        own_membership = own_membership.reshape(num_clusters, num_samples, -1)
        num_sites = own_membership.shape[2]
        diploid_membership = np.zeros((num_clusters, num_samples // 2, num_sites))
        cross_haploid_membership = np.zeros((num_clusters, num_samples // 2, num_sites))
        for sample in range(0, num_samples, 2):
            hap1 = own_membership[:, sample, :]
            hap2 = own_membership[:, sample + 1, :]
            diploid_membership[:, sample // 2, :] = hap1 + hap2
            cross_haploid_membership[:, sample // 2, :] = hap1 * hap2
        sum_diploid = diploid_membership.sum(axis=(1, 2))
        sum_diploid_squared = (diploid_membership**2).sum(axis=(1, 2))
        sum_cross_haploid = cross_haploid_membership.sum(axis=(1, 2))
        num = sum_diploid_squared - (sum_diploid**2) / num_sites / (num_samples // 2)
        denom = (
            sum_diploid
            + 2 * sum_cross_haploid
            - (sum_diploid**2) / num_sites / (num_samples // 2)
        )
        expected_r2 = num / denom
        print("Final expected R2 = " + str(np.sum(num) / np.sum(denom)))
        print("Per ancestry expected R2 = " + str(expected_r2))
    else:
        print("Not calculating expected R2 as both haploids not present....")


def write_membership_grid(
    args,
    own_membership,
    chr_map,
    mask_dodgy,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    n_clusters,
    sample_name_list,
    sample_id_list,
    output,
    window_size=1e3,
    exact_pos=None,
):
    assert len(tree_left_bp) == len(tree_right_bp)
    count_i = 0
    for k, (sample_name, sample_id) in enumerate(zip(sample_name_list, sample_id_list)):
        if exact_pos is None:
            chr_map_ = chr_map[mask_dodgy[k]]
            res = []
            for i, (c, l, r) in enumerate(
                zip(chr_map_, tree_left_bp[k], tree_right_bp[k])
            ):
                for j in range(
                    int(np.ceil(l / window_size)), int(np.ceil(r / window_size))
                ):
                    res.append(
                        [c, j * window_size] + own_membership[:, count_i].tolist()
                    )
                    count_i += 1
            df = pd.DataFrame(
                data=np.array(res),
                columns=["chr", "pos"] + ["prob_" + str(i) for i in range(n_clusters)],
            )
        else:
            df = exact_pos.copy()
            df[["prob_" + str(i) for i in range(n_clusters)]] = own_membership[
                :, k * len(df) : (k + 1) * len(df)
            ].T

        df["genpos"] = gen_grid_all[k]  ## unit in morgans
        df.to_csv(
            output
            + "_overall_membership_"
            + str(sample_name)
            + "_sample_id_"
            + str(sample_id)
            + ".csv",
            index=False,
            sep="\t",
        )


def write_membership_gamma(
    args,
    own_membership,
    gamma_arr,
    tau,
    trans_prop,
    mask_dodgy,
    chr_map,
    tree_left_bp,
    tree_right_bp,
    gen_grid_all,
    epoch_intervals,
    unique_groups,
    sample_id_label,
    sample_name_list,
    exact_pos,
    output=None,
):
    ## gamma and membership plots
    filename = (
        "overall_membership_" + sample_id_label + ".npy"
    )  ## this saves membership for all the trees (without the filtering)
    filename = args.output + "_" + filename
    with open(filename, "wb") as f:
        np.save(f, own_membership)

    write_membership_grid(
        args,
        own_membership,
        np.array(chr_map),
        mask_dodgy,
        tree_left_bp,
        tree_right_bp,
        gen_grid_all,
        args.num_clusters,
        sample_name_list,
        args.sample_id,
        args.output if output is None else output,
        args.force_build,
        exact_pos,
    )

    write_coal(
        gamma_arr,
        sample_id_label + ".coal",
        unique_groups,
        args.output,
        epoch_intervals,
    )

    with open(
        args.output + "_gamma_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, gamma_arr)

    with open(
        args.output + "_props_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, tau)

    with open(
        args.output + "_tadmix_" + sample_id_label + ".npy",
        "wb",
    ) as f:
        np.save(f, trans_prop)


def main(args):
    if args.load_mask is not None or args.cm_grid is not None:
        force_build_orig = args.force_build
        args.force_build = 1
        print("Exact positions can only be used with force_build=1")

    epoch_intervals = np.array(
        [-np.inf]
        + np.linspace(
            args.start_time - math.log(args.ypg, 10),
            args.end_time - math.log(args.ypg, 10),
            args.num_epochs - 1,
        ).tolist()
        + [np.inf],
        dtype="float64",
    )
    if (
        args.load_gamma is not None
        and args.load_props is not None
        and (args.t_admix_guess is not None or args.hmm is False)
    ):
        args.num_iters = 1
        args.sweep_num_iters = 0
        args.n_repeats = 1
        print(
            "Setting num_iters, num_iters and n_repeats to 1 as initial gamma and props are provided"
        )
    if args.only_make_pickle_files:
        args.num_iters = 1
        args.sweep_num_iters = 1
        args.n_repeats = 1
        print(
            "only_make_pickle_files=True: setting num_iters=1, sweep_num_iters=1, n_repeats=1"
        )

    if args.load_gamma is not None:
        if ".coal" in args.load_gamma:
            second_line = (
                open(args.load_gamma).readlines()[1].strip("\n").split(" ")[:-1]
            )
            epoch_intervals = np.array(
                np.log10(np.array(second_line, dtype="float64")).tolist() + [np.inf],
                dtype="float64",
            )
            args.num_epochs = len(epoch_intervals) - 1

    poplabels = pd.read_csv(args.poplabels, sep="\s+")
    chrs = list(map(int, args.chrs.split(",")))
    print("Considering chromosomes: " + str(chrs))

    ### Load all the trees in a list
    ts_list = load_trees(args)
    if len(ts_list) > 0:
        if len(poplabels) == ts_list[0].num_samples // 2:
            poplabels = pd.DataFrame(
                np.repeat(poplabels.values, 2, axis=0), columns=poplabels.columns
            )
        if len(poplabels) != ts_list[0].num_samples:
            raise ValueError(
                "Number of samples in trees doesnt match number of samples in poplabels.txt"
            )
    else:
        poplabels = pd.DataFrame(
            np.repeat(poplabels.values, 2, axis=0), columns=poplabels.columns
        )

    ### Setting the target sample ids
    sample_id = []
    for i in range(len(args.sample_id)):
        if "-" in args.sample_id[i]:
            sample_id.extend(
                np.arange(
                    int(args.sample_id[i].split("-")[0]),
                    int(args.sample_id[i].split("-")[1]) + 1,
                ).tolist()
            )
        else:
            try:
                sample_id.append(int(args.sample_id[i]))
            except:
                sample_id.extend(
                    poplabels[
                        (
                            (poplabels.INCLUDE == 1)
                            & (poplabels.GROUP == args.sample_id[i])
                        )
                    ].index.tolist()
                )
    sample_id = np.unique(sample_id).tolist()
    print(sample_id)
    args.sample_id = sample_id
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    if len(sample_id_label) > 100:
        print("Truncating sample_id_label to 100 characters to avoid long file names")
        sample_id_label = sample_id_label[0:100]
    sample_name_list = poplabels.loc[args.sample_id].ID.tolist()
    print(sample_name_list)

    poplabels.loc[args.sample_id, "INCLUDE"] = 1
    unique_groups = np.unique(poplabels[poplabels.INCLUDE == 1].GROUP)

    num_samples = len(poplabels)
    for sample in args.sample_id:
        if sample >= num_samples or sample < 0:
            raise ValueError("The sample ids are out of range")
        else:
            print(str(sample) + " is: " + str(poplabels.GROUP.iloc[sample]))

    ### Load tree stats
    (
        recomb_rates,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        chr_map,
        frac_branches_with_snp,
    ) = load_tree_stats(args, ts_list, poplabels, args.tree_stats_file_prefix)

    ## cM grid option
    if args.cm_grid is not None:
        df_all = pd.DataFrame()
        chrs = list(map(int, args.chrs.split(",")))
        for chr_no, chr in enumerate(chrs):
            rec_file = args.rec + str(chr) + ".txt"
            rec_file_gz = args.rec + str(chr) + ".txt.gz"
            if os.path.isfile(rec_file):
                recomb_map_msprime = msprime.RateMap.read_hapmap(rec_file)
            elif os.path.isfile(rec_file_gz):
                recomb_map_msprime = msprime.RateMap.read_hapmap(rec_file_gz)
            else:
                print(f"Recombination map for chromosome {chr} not found!")
                continue
            df = pd.DataFrame(
                np.arange(
                    np.array(tree_left_bp)[np.array(chr_map) == chr][0],
                    np.array(tree_right_bp)[np.array(chr_map) == chr][-1],
                    1,
                ),
                columns=["pos"],
            )
            df["genpos"] = recomb_map_msprime.get_cumulative_mass(
                np.minimum(
                    np.maximum(df["pos"].values, recomb_map_msprime.position.min()),
                    recomb_map_msprime.position.max(),
                )
            )
            m_grid = args.cm_grid / 100
            df["genpos_rounded"] = (df["genpos"] / m_grid).astype("int") * m_grid
            df = df.groupby("genpos_rounded").first().reset_index()
            df = df.drop(columns=["genpos_rounded"])
            df["chr"] = chr
            df_kb = pd.DataFrame(
                np.arange(
                    np.array(tree_left_bp)[np.array(chr_map) == chr][0],
                    np.array(tree_right_bp)[np.array(chr_map) == chr][-1],
                    force_build_orig,
                ),
                columns=["pos"],
            )
            df_kb["chr"] = chr
            df_union = (
                pd.concat([df[["chr", "pos"]], df_kb[["chr", "pos"]]])
                .drop_duplicates()
                .sort_values(by="pos")
                .reset_index(drop=True)
            )
            df_all = pd.concat([df_all, df_union], ignore_index=True)
        print("Saving the mask file corresponding to the cM grid...")
        df_all.to_csv(args.output + ".cm.mask", sep="\t", index=None)
        args.load_mask = args.output + ".cm.mask"

    ### Filter based on recombination rates
    if args.load_mask is None:
        mask_dodgy = []
        for sample_id in args.sample_id:
            mask_dodgy_sam = filter_recomb_rate(
                args.masking_threshold,
                tree_left_bp,
                tree_right_bp,
                recomb_rates,
                frac_branches_with_snp,
                chr_map,
            )
            mask_dodgy.append(mask_dodgy_sam)
        exact_pos = None
    else:
        exact_pos = pd.read_csv(args.load_mask, sep="\s+")
        exact_pos = exact_pos[
            exact_pos["chr"].isin(list(map(int, args.chrs.split(","))))
        ]
        exact_pos = exact_pos.sort_values(by=["chr", "pos"])
        exact_pos = exact_pos.reset_index(drop=True)
        mask_dodgy_recomb = filter_recomb_rate(
            args.masking_threshold,
            tree_left_bp,
            tree_right_bp,
            recomb_rates,
            frac_branches_with_snp,
            chr_map,
        )
        count = 0
        for chr_count, chr in enumerate(np.unique(chr_map)):
            for tree_left_i, tree_right_i in zip(
                np.array(tree_left_bp)[chr_map == chr],
                np.array(tree_right_bp)[chr_map == chr],
            ):
                if not mask_dodgy_recomb[count]:
                    exact_pos.loc[
                        (exact_pos.pos >= tree_left_i)
                        & (exact_pos.pos < tree_right_i)
                        & (exact_pos.chr == chr),
                        "pos",
                    ] = np.nan
                count += 1
            tree_first_pos = np.array(tree_left_bp)[chr_map == chr][0]
            tree_last_pos = np.array(tree_right_bp)[chr_map == chr][-1]
            exact_pos[(exact_pos.chr == chr) & (exact_pos.pos > tree_last_pos)] = np.nan
            exact_pos[(exact_pos.chr == chr) & (exact_pos.pos < tree_first_pos)] = (
                np.nan
            )
        exact_pos = exact_pos.dropna()
        load_mask = load_mask_csv(
            args, exact_pos, tree_left_bp, tree_right_bp, recomb_rates, chr_map
        )
        mask_dodgy = []
        for sample_id in args.sample_id:
            mask_dodgy.append(load_mask)

    num_trees_per_sample = [
        np.sum(mask_dodgy[sam]) for sam in range(len(args.sample_id))
    ]
    poplabels_included = poplabels[poplabels.INCLUDE == 1]
    tree_left_bp = np.array(
        [
            np.array(tree_left_bp)[mask_dodgy[sam]].tolist()
            for sam in range(len(args.sample_id))
        ]
    )
    tree_right_bp = np.array(
        [
            np.array(tree_right_bp)[mask_dodgy[sam]].tolist()
            for sam in range(len(args.sample_id))
        ]
    )
    tree_left_bp_gen = np.array(
        [
            np.array(tree_left_bp_gen)[mask_dodgy[sam]].tolist()
            for sam in range(len(args.sample_id))
        ]
    )
    tree_right_bp_gen = np.array(
        [
            np.array(tree_right_bp_gen)[mask_dodgy[sam]].tolist()
            for sam in range(len(args.sample_id))
        ]
    )

    ## generate gen_grid for each sample
    gen_grid_kb = {}
    for chr in list(map(int, args.chrs.split(","))):
        if os.path.isfile(args.rec + str(chr) + ".txt"):
            recomb_map_msprime = msprime.RateMap.read_hapmap(
                args.rec + str(chr) + ".txt"
            )
        elif os.path.isfile(args.rec + str(chr) + ".txt.gz"):
            recomb_map_msprime = msprime.RateMap.read_hapmap(
                args.rec + str(chr) + ".txt.gz"
            )
        else:
            raise "Recomb map format not identified"

        if args.load_mask is not None:
            exact_pos_chr = exact_pos[exact_pos["chr"] == chr]
            gen_grid_kb[chr] = recomb_map_msprime.get_cumulative_mass(
                np.minimum(
                    np.maximum(
                        exact_pos_chr["pos"].values, recomb_map_msprime.position.min()
                    ),
                    recomb_map_msprime.position.max(),
                )
            )
        else:
            grid_ = np.arange(
                0,
                int(
                    np.ceil(
                        tree_right_bp[0][np.array(chr_map)[mask_dodgy[0]] == chr].max()
                    )
                ),
                args.force_build,
            )
            gen_grid_kb[chr] = recomb_map_msprime.get_cumulative_mass(
                np.minimum(
                    np.maximum(grid_, recomb_map_msprime.position.min()),
                    recomb_map_msprime.position.max(),
                )
            )

    gen_grid_all = []
    for sample_no in range(len(args.sample_id)):
        gen_grid_sam = []
        if args.load_mask is None:
            for i, (c, l, r) in enumerate(
                zip(
                    np.array(chr_map)[mask_dodgy[sample_no]],
                    tree_left_bp[sample_no],
                    tree_right_bp[sample_no],
                )
            ):
                for j in range(
                    int(np.ceil(l / args.force_build)),
                    int(np.ceil(r / args.force_build)),
                ):
                    if args.hmm is False:
                        gen_grid_sam.append(j * 1e3)
                    else:
                        gen_grid_sam.append(gen_grid_kb[c][j])
        else:
            for c in list(map(int, args.chrs.split(","))):
                if args.hmm is False:
                    gen_grid_sam.extend(np.arange(0, len(gen_grid_kb[c])) * 1e3)
                else:
                    gen_grid_sam.extend(gen_grid_kb[c])
        gen_grid_all.append(np.array(gen_grid_sam))

    ### Load gt_ref is specified
    if args.gt_ref is not None:
        ## Currently assuming all target samples have same masking
        print("Loading the local ancestry information of the reference panel")
        with open(args.gt_ref) as f:
            unique_groups = f.readline().strip("\n").split(" ")
        gt_ref_df = pd.read_csv(args.gt_ref, sep=" ", skiprows=[0], header=None)
        gt_ref_df = gt_ref_df.rename(columns={0: "chr", 1: "pos"})

        ## convert from bp to trees
        gt_ref = np.zeros((len(poplabels), num_trees_per_sample[0]), dtype="float")
        for n_t in range(num_trees_per_sample[0]):
            chr_ = np.array(chr_map)[mask_dodgy[0]][n_t]
            tree_pos = (tree_left_bp[0][n_t] + tree_right_bp[0][n_t]) / 2
            gt_ref_chr = gt_ref_df[gt_ref_df["chr"] == chr_]
            gt_ref_nt = gt_ref_chr.iloc[
                (gt_ref_chr["pos"] - tree_pos).abs().argsort()[:1]
            ][gt_ref_chr.columns[2:]].values[0]
            gt_ref[:, n_t] = gt_ref_nt
    else:
        gt_ref = None

    st = time.time()
    target_branch_length_masked, mutscale_masked = get_target_branch_length(
        args,
        poplabels,
        ts_list,
        chrs,
        chr_map,
        mask_dodgy,
        args.sample_id,
        gt_ref=gt_ref,
        exact_pos=exact_pos,
    )
    print("target branch length: " + str(time.time() - st))

    (
        own_membership,
        trans_prop,
        tau,
        num,
        denom,
        denom_unscaled,
        proportion_of_coalescing_all,
        epoch_index_all,
        unique_groups,
        n_sites,
    ) = random_sweep(
        args,
        args.num_clusters,
        unique_groups,
        len(epoch_intervals),
        num_trees_per_sample,
        args.n_repeats,
        ts_list,
        poplabels,
        mask_dodgy,
        epoch_intervals,
        target_branch_length_masked,
        mutscale_masked,
        tree_left_bp,
        tree_right_bp,
        gen_grid_all,
        chr_map,
        gt_ref=gt_ref,
        exact_pos=exact_pos,
    )

    if args.only_make_pickle_files:
        print("only_make_pickle_files=True: stopping after generating pickle files.")
        return

    if args.load_gamma:
        gamma_arr = load_gamma(args.load_gamma, args.groups, unique_groups)
        ## CAUTION: recheck this, removing gamma_arr outside the range
        for epoch in range(gamma_arr.shape[2]):
            if (
                args.ignore_first_epoch
                and (args.start_time - math.log(args.ypg, 10)) > epoch_intervals[epoch]
            ):
                gamma_arr[:, :, epoch] = np.nan
            if (
                args.ignore_last_epoch
                and (args.end_time - math.log(args.ypg, 10)) <= epoch_intervals[epoch]
            ):
                gamma_arr[:, :, epoch] = np.nan
    else:
        gamma_arr = None

    if args.load_props:
        tau = load_props(args.load_props)

    ### EM
    filename = "overall_membership_iter0_" + sample_id_label + ".npy"
    filename = args.output + "_" + filename
    with open(filename, "wb") as f:
        np.save(f, own_membership)

    log_likelihood_arr = []

    filename_logl = args.output + "_" + sample_id_label + ".logl"
    filename_tau = args.output + "_" + sample_id_label + ".tau"
    f_logl = open(filename_logl, "w")
    f_tau = open(filename_tau, "w")

    st = time.time()
    for epoch in range(args.num_iters):
        own_membership, trans_prop, gamma_arr, tau, log_likelihood = e_m_step(
            args,
            own_membership,
            trans_prop,
            tau,
            gamma_arr,
            proportion_of_coalescing_all,
            epoch_index_all,
            denom,
            unique_groups,
            epoch_intervals,
            n_sites,
            len(args.sample_id),
            epoch,
            target_branch_length_masked,
            tree_left_bp,
            tree_right_bp,
            gen_grid_all,
        )
        if epoch == 0:
            write_coal(
                gamma_arr,
                sample_id_label + "_iter0.coal",
                unique_groups,
                args.output,
                epoch_intervals,
            )
        for i in range(np.shape(tau)[0]):
            f_tau.write(str(tau[i]) + " ")
        f_tau.write("\n")
        log_likelihood_arr.append(log_likelihood)

        f_logl.write(str(log_likelihood_arr[-1]) + "\n")

    print("HMM admix date = " + str(trans_prop))
    print("em iters: " + str(time.time() - st))
    print("Final log-likelihood = " + str(log_likelihood_arr[-1]))
    get_expected_r2(own_membership, len(args.sample_id))

    write_membership_gamma(
        args,
        own_membership,
        gamma_arr,
        tau,
        trans_prop,
        mask_dodgy,
        chr_map,
        tree_left_bp,
        tree_right_bp,
        gen_grid_all,
        epoch_intervals,
        unique_groups,
        sample_id_label,
        sample_name_list,
        exact_pos,
    )

    print("Saving the coal. rates and proportions without using HMM")
    # target_branch_length_masked_unity = target_branch_length_masked
    # for i in range(len(target_branch_length_masked_unity)):
    #     for j in range(len(target_branch_length_masked_unity[i])):
    #         for k in range(len(target_branch_length_masked_unity[i][j])):
    #             target_branch_length_masked_unity[i][j][k] = 1
    gen_grid_all_unscaled = copy.deepcopy(gen_grid_all)
    for sample_no in range(len(args.sample_id)):
        gen_grid_all_unscaled[sample_no] = 1e3 * np.arange(
            0, len(gen_grid_all[sample_no])
        )

    own_membership_nohmm, _, gamma_arr_nohmm, tau_nohmm, _ = e_m_step(
        args,
        own_membership,
        trans_prop,
        tau,
        gamma_arr,
        proportion_of_coalescing_all,
        epoch_index_all,
        denom_unscaled,
        unique_groups,
        epoch_intervals,
        n_sites,
        len(args.sample_id),
        epoch,  ## should this be 0 or epoch (last iteration)?
        mutscale_masked,
        tree_left_bp,
        tree_right_bp,
        gen_grid_all_unscaled,
    )
    write_membership_gamma(
        args,
        own_membership_nohmm,
        gamma_arr_nohmm,
        tau_nohmm,
        trans_prop,
        mask_dodgy,
        chr_map,
        tree_left_bp,
        tree_right_bp,
        gen_grid_all,
        epoch_intervals,
        unique_groups,
        "nohmm_" + sample_id_label,
        sample_name_list,
        exact_pos,
        output=args.output + "_nohmm",
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sample_id",
        "--sample_id",
        help="Enter space seperated list of the indices of haplotype you wish local ancestry for",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-fb",
        "--force_build",
        help="force build size to subsample the trees in bp",
        type=float,
        default=10000,
    )
    parser.add_argument("-r", "--rec", help="Filename of rec maps.", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="Output prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "-chrs",
        "--chrs",
        help="Comma-seperated list of chromosomes to be considered",
        type=str,
        default="1,2",
    )
    parser.add_argument(
        "-masking_thresh",
        "--masking_threshold",
        help="Remove top x cent of high recombination regions",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--n_repeats",
        help="Number of restarts when randomly initializing local ancestry, we choose best out of n_trails",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-load_gamma",
        "--load_gamma",
        help="Starting gamma values written in a file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-force_relate_gammas",
        "--force_relate_gammas",
        help="Force using relate gammas until the end (completely supervied GB)",
        type=boolean,
        default=False,
    )
    parser.add_argument(
        "-load_mask",
        "--load_mask",
        help="Load mask csv file with chr, tree_position_left",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-load_props",
        "--load_props",
        help="Starting taU values written in a file or space seperated list",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--t_admix_guess",
        help="Guess for the time of admixture",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-trees",
        "--trees",
        help="Location to trees in tskit format",
        type=str,
        default=None,
    )
    parser.add_argument("--poplabels", help="Location to poplabels file", type=str)
    parser.add_argument(
        "-i",
        "--num_iters",
        help="Number of iterations for EM",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--only_make_pickle_files",
        action="store_true",
        help="Generate tree/fixed/branch pickle files only; skip EM and force num_iters=1, n_repeats=1, sweep_num_iters=1",
    )
    parser.add_argument(
        "-k",
        "--num_clusters",
        help="Number of clusters to find using the EM",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-start_time",
        "--start_time",
        help="Starting time for the population size plots, measured in log-scale",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "-end_time",
        "--end_time",
        help="Ending time for the population size plots, measured in log-scale",
        type=float,
        default=6,
    )
    parser.add_argument(
        "-num_epochs",
        "--num_epochs",
        help="Num epochs",
        type=int,
        default=9,
    )
    parser.add_argument(
        "-ignore_first_epoch",
        "--ignore_first_epoch",
        help="Ignore first epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "-ignore_last_epoch",
        "--ignore_last_epoch",
        help="Ignore last epoch while calculating the local ancestry in the EM",
        type=boolean,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--sweep_num_iters",
        type=int,
        default=10,
        help="Number of iterations to run EM for in random sweep",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        nargs="+",
        help="space seperated list of source group, example Nea CHB where Nea and CHB are population names in poplabels file",
    )
    parser.add_argument(
        "--hmm",
        help="Run HMM or treat each window as independent",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "--mut_scaling",
        help="Scale the likelihood based on presence of mutation on the lineage, improves robustness to bottlenecks",
        type=boolean,
        default=True,
    )
    parser.add_argument(
        "--ypg", type=float, default=28, help="years per generation, 28 years default"
    )
    parser.add_argument(
        "--tree_stats_file_prefix",
        type=str,
        default=None,
        help="file prefix for the tree stats file",
    )
    parser.add_argument(
        "--branch_persistence_file_prefix",
        type=str,
        default=None,
        help="file prefix for the branch persistence file",
    )
    parser.add_argument(
        "--fixed_params_file_prefix",
        type=str,
        default=None,
        help="file prefix for the fixed params file",
    )
    parser.add_argument(
        "--load_membership",
        help="Load the membership from a .npy file",
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
        "--gt_ref", help="Local ancestry of the reference panel", type=str, default=None
    )
    parser.add_argument(
        "--node_persist_thresh",
        type=float,
        default=0.5,
        help="Correlation coefficient threshold above which nodes are considered equivalent",
    )
    parser.add_argument(
        "--cm_grid",
        type=float,
        default=None,
        help="Store local ancestry information per cM instead",
    )
    parser.add_argument(
        "--ignore_coal_between_targets",
        help="Whether to ignore coalescene events between target samples",
        type=boolean,
        default=False,
    )
    args = parser.parse_args()
    np.random.seed(args.seed)  ## fix the random seed
    random.seed(args.seed)
    print(args)
    main(args)
