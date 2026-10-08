"""
utils.py: contains basic helper function used throughout the code
"""

import argparse
import numpy as np
from sklearn.calibration import calibration_curve
import pandas as pd
import copy
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
import pdb
import numba as nb
from numba import jit
from numba.typed import List
import pickle
import time
from src.infer_node_persistence import get_coal_descendants, get_approx_node_persistence


def get_mut_scaling_grid():
    mut_scaling = {}
    for i in range(1, 1000):
        sum_ = 0
        for j in range(1, i + 1):
            sum_ += 2 / (j + 1)
        mut_scaling[i] = i / sum_
        # mut_scaling[i] = i
    return mut_scaling


mut_scaling_grid = get_mut_scaling_grid()


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_numba_nested_list(arr_list):
    ## careful with empty lists
    arr_list_nb = List()
    for i in arr_list:
        inner = List()
        for j in i:
            inner.append(j)
        arr_list_nb.append(inner)
    return arr_list_nb


def make_one_hot(X, max_X=None):
    X = np.array(X, dtype="int")
    classes = np.arange(0, max_X, 1) if max_X is not None else np.sort(np.unique(X))
    # Y = []
    if len(X.shape) == 2:
        Y = np.zeros((len(classes), X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        Y = np.zeros((len(classes), X.shape[0]))
    for c in range(len(classes)):
        # Y.append(scipy.sparse.csr_matrix(np.array(X == c, dtype='int')))
        Y[c] = np.array(X == classes[c], dtype="int")
    return Y


def mask_for_dodgy_trees(recomb_rates, masking_thresh):
    recomb_rates = np.array(recomb_rates)
    mask = recomb_rates <= np.nanpercentile(recomb_rates, (masking_thresh) * 100)
    return mask


def write_coal(gamma_arr, filename, labs, output, epoch_intervals):
    if type(labs) == dict:
        labs = list(labs.keys())
    if len(labs) > 10:
        epoch_intervals_pow = np.power(10, epoch_intervals)
        f = open(output + "_" + filename + ".all", "w")
        f.write(" ".join(labs) + "\n")
        for val in epoch_intervals_pow:
            f.write("%s " % val)
        f.write("\n")
        for i in range(gamma_arr.shape[0]):
            for j in range(gamma_arr.shape[1]):
                f.write(str(i) + " " + str(j) + " ")
                for e in range(gamma_arr.shape[2]):
                    f.write(str(gamma_arr[i][j][e]) + " ")
                f.write("\n")

        f.close()

        min_avg_coal = []
        mean_popsize = -np.log10(np.nanmean(gamma_arr[:, :, 1:-1], axis=2))
        for i in range(gamma_arr.shape[1]):
            min_diff_arr = []
            for j in range(gamma_arr.shape[0]):
                min_diff_with_other_comp = -np.inf
                for k in range(gamma_arr.shape[0]):
                    if k != j:
                        min_diff_with_other_comp = max(
                            min_diff_with_other_comp,
                            mean_popsize[j, i] - mean_popsize[k, i],
                        )
                min_diff_arr.append(min_diff_with_other_comp)
            min_avg_coal.append(min_diff_arr)

        min_avg_coal = np.array(min_avg_coal)  ## N_ref x N_clusters
        top_groups = []
        for j in range(gamma_arr.shape[0]):
            top_groups.extend(
                np.argsort(min_avg_coal[:, j])[0 : 10 // gamma_arr.shape[0]]
            )

        labs = np.array(labs)[top_groups].tolist()
        gamma_arr = gamma_arr[:, top_groups]

    epoch_intervals_pow = np.power(10, epoch_intervals)
    filename = output + "_" + filename
    f = open(filename, "w")
    f.write(" ".join(labs) + "\n")
    for val in epoch_intervals_pow:
        f.write("%s " % val)
    f.write("\n")
    for i in range(gamma_arr.shape[0]):
        for j in range(gamma_arr.shape[1]):
            f.write(str(i) + " " + str(j) + " ")
            for e in range(gamma_arr.shape[2]):
                f.write(str(gamma_arr[i][j][e]) + " ")
            f.write("\n")

    f.close()


def filter_recomb_rate(
    masking_threshold,
    tree_left_bp,
    tree_right_bp,
    recomb_rates,
    frac_branches_with_snp_all,
    chr_map,
):
    recomb_rates = np.array(recomb_rates)
    frac_branches_with_snp_all = np.array(frac_branches_with_snp_all)
    tree_interval = np.abs(np.array(tree_right_bp) - np.array(tree_left_bp))
    mask_dodgy = np.ones_like(recomb_rates, dtype="bool")
    for chr in np.unique(chr_map):
        recomb_rates_chr = recomb_rates[chr_map == chr]
        frac_branches_with_snp_all_chr = frac_branches_with_snp_all[chr_map == chr]
        tree_interval_chr = tree_interval[chr_map == chr]
        for i, r in enumerate(recomb_rates_chr):
            if np.isnan(r):
                recomb_rates_chr[
                    np.abs(
                        np.array(tree_left_bp)[chr_map == chr]
                        - np.array(tree_left_bp)[chr_map == chr][i]
                    )
                    < 500000
                ] = np.inf
        recomb_rates_chr = np.nan_to_num(recomb_rates_chr, posinf=np.nan)
        mask_dodgy[chr_map == chr] = ~np.isnan(recomb_rates_chr)

        recomb_0_thresh = np.sum(np.array(recomb_rates_chr) <= 0) / len(
            recomb_rates_chr
        )
        mask_dodgy[chr_map == chr] *= ~mask_for_dodgy_trees(
            recomb_rates_chr,
            recomb_0_thresh,
        )
        mask_dodgy[chr_map == chr] *= mask_for_dodgy_trees(
            recomb_rates_chr,
            1 - masking_threshold,
        )
        print(
            f"After recomb rate filter: {np.sum(mask_dodgy[chr_map == chr])} trees remain"
        )
        # frac_branched_with_snp_cutoff = np.nanpercentile(
        #     frac_branches_with_snp_all_chr, 80
        # )
        # print(
        #     "Filtering trees with fraction of branches with mutation < "
        #     + str(frac_branched_with_snp_cutoff)
        # )
        # mask_dodgy[chr_map == chr] *= (
        #     frac_branches_with_snp_all_chr >= frac_branched_with_snp_cutoff
        # )
        # print(
        #     f"After mut on branch filter: {np.sum(mask_dodgy[chr_map == chr])} trees remain"
        # )

        # print("Filtering trees with interval > 1MB")
        # mask_dodgy[chr_map == chr] *= tree_interval_chr < 1e6
        # print(
        #     f"After tree interval filter: {np.sum(mask_dodgy[chr_map == chr])} trees remain"
        # )

        ### removing coldspots
        # mask_dodgy[chr_map == chr] *= ~mask_for_dodgy_trees(
        #     recomb_rates_chr,
        #     0.1,
        # )

    mask_dodgy = np.array(mask_dodgy)

    print(
        "Filtering based on recombination rate, trees remaining: "
        + str(sum(mask_dodgy))
        + " average recomb. rate: "
        + str(np.mean(np.array(recomb_rates)[mask_dodgy]))
    )
    return mask_dodgy


def load_mask_csv(
    args, membership_mask, tree_left_bp, tree_right_bp, recomb_rates, chr_map
):
    mask_dodgy = np.zeros(len(tree_left_bp), dtype="bool")
    # mask_dodgy_nan = np.zeros(len(tree_left_bp), dtype="bool")
    recomb_rates = np.array(recomb_rates)
    count = 0
    for chr_count, chr in enumerate(np.unique(chr_map)):
        # recomb_rates_chr = recomb_rates[chr_map == chr]
        # for i, r in enumerate(recomb_rates_chr):
        #     if np.isnan(r):
        #         recomb_rates_chr[
        #             np.abs(np.array(tree_left_bp)[chr_map == chr] - np.array(tree_left_bp)[chr_map == chr][i]) < 500000
        #         ] = np.inf
        # recomb_rates_chr = np.nan_to_num(recomb_rates_chr, posinf=np.nan)
        # mask_dodgy_nan[chr_map == chr] = ~np.isnan(recomb_rates_chr)
        membership_mask_chr = membership_mask[membership_mask["chr"] == chr]
        membership_mask_chr["pos"] = membership_mask_chr["pos"]
        membership_mask_chr["pos"] = membership_mask_chr["pos"].astype(int)
        membership_mask_chr = membership_mask_chr.sort_values(by="pos")
        for tree_left_i, tree_right_i in zip(
            np.array(tree_left_bp)[chr_map == chr],
            np.array(tree_right_bp)[chr_map == chr],
        ):
            if (
                membership_mask_chr[
                    (membership_mask_chr["pos"] >= tree_left_i)
                    & (membership_mask_chr["pos"] < tree_right_i)
                ].shape[0]
                > 0
            ):
                mask_dodgy[count] = True
            count += 1
    print("Number of trees = " + str(np.sum(mask_dodgy)))
    return mask_dodgy


@jit(nopython=True, fastmath=True)
def compute_gamma_num(
    own_membership,
    prev_gamma,
    proportion_of_coalescing_all,
    epoch_index_all,
    num_ref_groups,
    n_epochs,
    target_branch_length,
    ignore_first_epoch,
    ignore_last_epoch,
):
    num_full_tree = np.zeros((num_ref_groups, n_epochs - 1), dtype="float64")
    for n_site in range(len(own_membership)):
        proportion_of_coalescing_in_tree = proportion_of_coalescing_all[n_site]
        epoch_index_in_tree = epoch_index_all[n_site]
        target_branch_length_tree = target_branch_length[n_site]
        for i in range(len(proportion_of_coalescing_in_tree)):
            if (
                (
                    ignore_first_epoch
                    and not ignore_last_epoch
                    and epoch_index_in_tree[i] >= 1
                )
                or (
                    ignore_last_epoch
                    and not ignore_first_epoch
                    and epoch_index_in_tree[i] < n_epochs - 2
                )
                or (
                    ignore_first_epoch
                    and ignore_last_epoch
                    and epoch_index_in_tree[i] >= 1
                    and epoch_index_in_tree[i] < n_epochs - 2
                )
                or (not ignore_first_epoch and not ignore_last_epoch)
            ):
                epoch = epoch_index_in_tree[i]
                prev_gamma_e = prev_gamma[:, epoch]
                num = prev_gamma_e * proportion_of_coalescing_in_tree[i]
                sum_of_num = np.sum(num)
                if (
                    sum_of_num != 0
                ):  ## sometimes the num are less than python float64 precision, we ignore those coal events while calculating
                    num = num / sum_of_num
                common_term = own_membership[n_site] / target_branch_length_tree[i]
                num_full_tree[:, epoch] += common_term * num
    return num_full_tree


def compute_gamma_denom(own_membership, denom):
    eps = 1e-200
    denom_1 = np.sum((own_membership * denom.T).T, axis=0)
    return denom_1 + eps


def load_gamma(path, groups, ref_groups):
    if ".npy" in path:
        return np.load(path)
    elif ".coal" in path:
        with open(path) as f:
            header = f.readline().strip("\n").split(" ")
        header = np.array(header)
        if len(np.intersect1d(ref_groups, header)) != len(ref_groups):
            print(
                "Groups in header do not match groups in input, groups in header are: "
                + str(header)
            )
            import pdb

            pdb.set_trace()
            raise ValueError
        groups_to_index = []
        ref_groups_to_index = []
        if groups is None:
            groups_to_index = np.arange(len(header))
        else:
            for g in groups:
                try:
                    groups_to_index.append(np.where(header == g)[0][0])
                except:
                    groups_to_index.append(np.nan)
        for g in ref_groups:
            try:
                ref_groups_to_index.append(np.where(header == g)[0][0])
            except:
                ref_groups_to_index.append(np.nan)
        df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=[0, 1])
        gamma_arr = np.nan * np.ones((len(groups), len(ref_groups), df.shape[1] - 2))
        for i, gid1 in enumerate(groups_to_index):
            for j, gid2 in enumerate(ref_groups_to_index):
                if not np.isnan(gid2):
                    gamma_arr[i, j] = df[(df[0] == gid1) & (df[1] == gid2)].values[
                        :, 2:
                    ]

        return gamma_arr
    else:
        print("Unsupported file format for gamma files")


# def load_gamma(path, groups, ref_groups):
#     if ".npy" in path:
#         return np.load(path)
#     elif ".coal" in path:
#         with open(path) as f:
#             header = f.readline().strip("\n").split(" ")
#         header = np.array(header)
#         if len(np.intersect1d(ref_groups, header)) != len(ref_groups):
#             print(
#                 "Groups in header do not match groups in input, groups in header are: "
#                 + str(header)
#             )
#             raise ValueError
#         groups_to_index = [np.where(header == g)[0][0] for g in groups]
#         ref_groups_to_index = [np.where(header == g)[0][0] for g in ref_groups]
#         df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=[0, 1])
#         gamma_arr = np.zeros((len(groups), len(ref_groups), df.shape[1] - 2))
#         for i, gid1 in enumerate(groups_to_index):
#             for j, gid2 in enumerate(ref_groups_to_index):
#                 gamma_arr[i, j] = df[(df[0] == gid1) & (df[1] == gid2)].values[:, 2:]
#         print(np.nan_to_num(gamma_arr, nan=0))
#         return np.nan_to_num(gamma_arr, nan=0)
#     else:
#         print("Unsupported file format for gamma files")


def load_props(path):
    if ".npy" in path:
        return np.load(path)
    elif ".txt" in path:
        return np.loadtxt(path)
    else:
        return np.array(path.split(" "), dtype="float")


def load_tadmix(path):
    try:
        return np.load(path)
    except:
        return float(path)


def scale_number_window_list(number_window_list, num_muts_list):
    ### caution!!!!!!
    number_window_list = np.array(number_window_list)  # *(1-1/math.e)
    prev_node_with_mutation = 0
    for i in range(len(number_window_list)):
        if num_muts_list[i] > 0:
            number_window_list[prev_node_with_mutation : i + 1] *= mut_scaling_grid[
                i + 1 - prev_node_with_mutation
            ]
            prev_node_with_mutation = i + 1
        elif i == len(number_window_list) - 1:
            number_window_list[prev_node_with_mutation:] *= mut_scaling_grid[
                i + 1 - prev_node_with_mutation
            ]
    return number_window_list.tolist()


def get_target_branch_length(
    args,
    poplabels,
    ts_list,
    chrs,
    chr_map,
    mask_dodgy,
    sample_list,
    gt_ref=None,
    exact_pos=None,
):
    """
    Calculates the branch length of the target population.
    exact_pos is a dataframe with chr, pos where we want the target branch length exactly.
    """
    target_branch_length = []
    mutscale = []

    for sample_no, sample in enumerate(sample_list):
        target_branch_length_sample = List()
        mutscale_sample = List()
        leave_one_sample_out = list(set(sample_list) - set([sample]))

        for chr_no, chr in enumerate(chrs):
            if args.branch_persistence_file_prefix is not None:
                branch_persistence_file_name = (
                    args.branch_persistence_file_prefix
                    + "_chr"
                    + str(chr)
                    + "_sample"
                    + str(sample)
                    + ".pkl"
                )
            else:
                branch_persistence_file_name = (
                    args.output
                    + "_branch_persistence_chr"
                    + str(chr)
                    + "_sample"
                    + str(sample)
                    + ".pkl"
                )

            try:
                if args.ignore_coal_between_targets:
                    with open(branch_persistence_file_name, "rb") as f_pkl:
                        (
                            sample_id_file,
                            mut_scaling_file,
                            hmm_file,
                            force_build_file,
                            start_time,
                            end_time,
                            ignore_first_epoch,
                            ignore_last_epoch,
                            masking_threshold,
                            poplabels_file,
                            target_branch_length_sample_chr,
                            mutscale_sample_chr,
                            gt_ref_na_sum,
                            exact_pos_file,
                        ) = pickle.load(f_pkl)
                else:
                    with open(branch_persistence_file_name, "rb") as f_pkl:
                        (
                            mut_scaling_file,
                            hmm_file,
                            force_build_file,
                            start_time,
                            end_time,
                            ignore_first_epoch,
                            ignore_last_epoch,
                            masking_threshold,
                            poplabels_file,
                            target_branch_length_sample_chr,
                            mutscale_sample_chr,
                            gt_ref_na_sum,
                            exact_pos_file,
                        ) = pickle.load(f_pkl)
                        sample_id_file = args.sample_id
                if (
                    ((sample_id_file == args.sample_id) | (args.ignore_coal_between_targets))
                    & (mut_scaling_file == args.mut_scaling)
                    & (hmm_file == args.hmm)
                    & (force_build_file == args.force_build)
                    &
                    # (start_time == args.start_time) &
                    # (end_time == args.end_time) &
                    # (ignore_first_epoch == args.ignore_first_epoch) &
                    # (ignore_last_epoch == args.ignore_last_epoch) &
                    (masking_threshold == args.masking_threshold)
                    & (np.all(poplabels_file == poplabels.values) | (args.ignore_coal_between_targets))
                ):
                    if gt_ref is not None:
                        gt_ref_chr = gt_ref[
                            :, np.array(chr_map)[mask_dodgy[sample_no]] == chr
                        ]
                        if gt_ref_na_sum != np.sum(np.isnan(gt_ref_chr)):
                            print(
                                "Branch persistence statistics file does not match the current settings, recomputing..."
                            )
                            raise Exception
                    if exact_pos is not None:
                        if len(exact_pos[(exact_pos["chr"] == chr)]) != len(
                            exact_pos_file[exact_pos_file[:, 0] == chr]
                        ):
                            print(
                                "Branch persistence statistics file does not match the current settings, recomputing..."
                            )
                            raise Exception
                        if (
                            exact_pos[(exact_pos["chr"] == chr)].values
                            != exact_pos_file[exact_pos_file[:, 0] == chr]
                        ).any():
                            print(
                                "Branch persistence statistics file does not match the current settings, recomputing..."
                            )
                            raise Exception
                    for i in target_branch_length_sample_chr:
                        numba_i = List().empty_list(nb.types.float64)
                        for j in i:
                            numba_i.append(j)
                        target_branch_length_sample.append(numba_i)
                    for i in mutscale_sample_chr:
                        numba_i = List().empty_list(nb.types.float64)
                        for j in i:
                            numba_i.append(j)
                        mutscale_sample.append(numba_i)
                    print(
                        f"Loaded branch persistence statistics from: {branch_persistence_file_name}"
                    )
                    continue
                else:
                    print(
                        "Branch persistence statistics file does not match the current settings, recomputing..."
                    )
                    raise Exception
            except:
                count_all_tree, count_all_tree2, count_all_tree3 = 0, 0, 0
                print(
                    f"Saving branch persistence statistics to: {branch_persistence_file_name}"
                )
                mask_dodgy_sam_chr = mask_dodgy[sample_no][np.array(chr_map) == chr]
                if gt_ref is not None:
                    gt_ref_chr = gt_ref[
                        :, np.array(chr_map)[mask_dodgy[sample_no]] == chr
                    ]
                else:
                    gt_ref_chr = None

                target_branch_length_sample_chr = []
                mutscale_sample_chr = []
                ts = ts_list[chr_no]
                ts_edges = ts.edges()
                if exact_pos is not None:
                    exact_pos_chr = exact_pos[exact_pos["chr"] == chr]

                tree_left_bp_chr, tree_right_bp_chr = [], []
                for tree in ts.trees():
                    if (
                        np.ceil(tree.interval[1] / args.force_build)
                        - np.ceil(tree.interval[0] / args.force_build)
                        > 0
                    ):
                        if mask_dodgy_sam_chr[count_all_tree]:
                            tree_left_bp_chr.append(tree.interval[0])
                            tree_right_bp_chr.append(tree.interval[1])
                        count_all_tree += 1

                bp_grid = []
                num_sites_per_tree = []
                for i, (l, r) in enumerate(zip(tree_left_bp_chr, tree_right_bp_chr)):
                    if exact_pos is not None:
                        num_sites = np.searchsorted(
                            exact_pos_chr["pos"].values, r
                        ) - np.searchsorted(exact_pos_chr["pos"].values, l)
                    else:
                        num_sites = np.ceil(r / args.force_build) - np.ceil(
                            l / args.force_build
                        )
                    num_sites_per_tree.append(num_sites)
                    for j in range(
                        int(np.ceil(l / args.force_build)),
                        int(np.ceil(r / args.force_build)),
                    ):
                        bp_grid.append(j * args.force_build)  ## caution
                if exact_pos is not None:
                    bp_grid = exact_pos[(exact_pos["chr"] == chr)]["pos"].values
                else:
                    bp_grid = np.array(bp_grid)
                num_sites_per_tree = np.array(num_sites_per_tree, dtype="int")

                ### Might be faster than previous implementation
                # if exact_pos is not None:
                #     left_indices = np.searchsorted(exact_pos_chr['pos'].values, tree_left_bp_chr)
                #     right_indices = np.searchsorted(exact_pos_chr['pos'].values, tree_right_bp_chr)
                #     num_sites_per_tree = right_indices - left_indices
                #     bp_grid = exact_pos_chr['pos'].values
                # else:
                #     num_sites_per_tree = (tree_right_bp_chr // args.force_build) - (tree_left_bp_chr // args.force_build)
                #     ranges = [np.arange(int(l / args.force_build) + 1, int(r / args.force_build) + 1) * args.force_build for l, r in zip(tree_left_bp_chr, tree_right_bp_chr)]
                #     bp_grid = np.concatenate(ranges)
                # num_sites_per_tree = np.array(num_sites_per_tree, dtype='int')

                # df_coal_time_matrix = get_coal_times(ts, sample, bp_grid)
                df_coal_descendants = get_coal_descendants(
                    ts, sample, bp_grid, ts.num_samples
                )

                tree = ts.first()
                poplabels_included = poplabels[poplabels.INCLUDE == 1].index.values
                for tid in tqdm(range(ts.num_trees)):
                    if (
                        np.ceil(tree.interval[1] / args.force_build)
                        - np.ceil(tree.interval[0] / args.force_build)
                        > 0
                    ):
                        if mask_dodgy_sam_chr[count_all_tree2]:
                            number_of_overlaps_list = get_approx_node_persistence(
                                df_coal_descendants,
                                (tree.interval[0] + tree.interval[1]) / 2,
                                args.node_persist_thresh,
                            )
                            poplabels_included_pos = poplabels_included.copy()
                            number_window_list = (
                                []
                            )  # List().empty_list(nb.types.float64)
                            num_muts_list = []
                            parent = copy.deepcopy(sample)
                            if gt_ref is not None:
                                gt_ref_nt = gt_ref_chr[:, count_all_tree3]
                                new_mask = (poplabels.INCLUDE == 1) & (
                                    ~np.isnan(gt_ref_nt) | (poplabels.index == sample)
                                )
                                poplabels_included_pos = poplabels[
                                    new_mask
                                ].index.values.copy()
                                count_all_tree3 += 1

                            edge_count = 0
                            while parent != tree.root:
                                edge_id = tree.edge(parent)
                                edge = ts_edges[edge_id]
                                edgep = ts_edges[tree.edge(tree.parent(parent))]
                                parent = tree.parent(parent)
                                tree_childrens = tree.children(parent)
                                tree_leaves_left = list(tree.leaves(tree_childrens[0]))
                                tree_leaves_right = list(tree.leaves(tree_childrens[1]))
                                left_check = np.intersect1d(
                                    tree_leaves_left, poplabels_included_pos
                                )
                                right_check = np.intersect1d(
                                    tree_leaves_right, poplabels_included_pos
                                )
                                if args.ignore_coal_between_targets:
                                    left_check_size = (
                                        left_check.size
                                        - np.intersect1d(
                                            left_check, leave_one_sample_out
                                        ).size
                                    )
                                    right_check_size = (
                                        right_check.size
                                        - np.intersect1d(
                                            right_check, leave_one_sample_out
                                        ).size
                                    )
                                else:
                                    left_check_size = left_check.size
                                    right_check_size = right_check.size

                                if left_check_size > 0 and right_check_size > 0:
                                    num_muts_list.append(
                                        int(
                                            edgep.metadata.decode("utf-8")
                                            .rstrip("\x00")
                                            .split(" ")[2]
                                        )
                                    )
                                    if args.hmm:
                                        # edge_left, edge_right, _ = edge.metadata.decode('utf-8').rstrip('\x00').split(" ")
                                        # edge_left, edge_right = min(tree.interval[0], float(edge_left)), max(tree.interval[1],float(edge_right))
                                        # if parent != tree.root:
                                        #     edgep_right = max(float(edgep.metadata.decode('utf-8').split(" ")[1]), tree.interval[1])
                                        #     edgep_left = min(float(edgep.metadata.decode('utf-8').split(" ")[0]), tree.interval[0])
                                        #     edge_left = min(edge_left, edgep_left)
                                        #     edge_right = max(edge_right, edgep_right)
                                        # number_of_overlaps = np.searchsorted(bp_grid, edge_right) - np.searchsorted(bp_grid, edge_left)
                                        number_of_overlaps = number_of_overlaps_list[
                                            edge_count
                                        ]
                                        number_window_list.append(
                                            1.0 * number_of_overlaps
                                        )
                                    else:
                                        number_window_list.append(1.0)
                                edge_count += 1
                            ## scale number_window_list by the number of mutations in the tree
                            if args.mut_scaling:
                                number_window_list = scale_number_window_list(
                                    number_window_list, num_muts_list
                                )
                                mutscale_sample_chr.append(
                                    scale_number_window_list(
                                        np.ones_like(number_window_list), num_muts_list
                                    )
                                )
                            else:
                                mutscale_sample_chr.append(
                                    np.ones_like(number_window_list)
                                )
                            target_branch_length_sample_chr.append(number_window_list)
                        count_all_tree2 += 1
                    tree.next()

                target_branch_length_sample_chr = [
                    sublist
                    for sublist, count in zip(
                        target_branch_length_sample_chr, num_sites_per_tree
                    )
                    for _ in range(count)
                ]
                mutscale_sample_chr = [
                    sublist
                    for sublist, count in zip(mutscale_sample_chr, num_sites_per_tree)
                    for _ in range(count)
                ]
                if args.ignore_coal_between_targets:
                    with open(branch_persistence_file_name, "wb") as f_pkl:
                        pickle.dump(
                            [
                                args.sample_id,
                                args.mut_scaling,
                                args.hmm,
                                args.force_build,
                                args.start_time,
                                args.end_time,
                                args.ignore_first_epoch,
                                args.ignore_last_epoch,
                                args.masking_threshold,
                                poplabels.values,
                                target_branch_length_sample_chr,
                                mutscale_sample_chr,
                                np.isnan(gt_ref).sum() if gt_ref is not None else None,
                                exact_pos.values if exact_pos is not None else None,
                            ],
                            f_pkl,
                        )
                else:
                    with open(branch_persistence_file_name, "wb") as f_pkl:
                        pickle.dump(
                            [
                                args.mut_scaling,
                                args.hmm,
                                args.force_build,
                                args.start_time,
                                args.end_time,
                                args.ignore_first_epoch,
                                args.ignore_last_epoch,
                                args.masking_threshold,
                                poplabels.values,
                                target_branch_length_sample_chr,
                                mutscale_sample_chr,
                                np.isnan(gt_ref).sum() if gt_ref is not None else None,
                                exact_pos.values if exact_pos is not None else None,
                            ],
                            f_pkl,
                        )
                for i in target_branch_length_sample_chr:
                    numba_i = List().empty_list(nb.types.float64)
                    for j in i:
                        numba_i.append(j)
                    target_branch_length_sample.append(numba_i)
                for i in mutscale_sample_chr:
                    numba_i = List().empty_list(nb.types.float64)
                    for j in i:
                        numba_i.append(j)
                    mutscale_sample.append(numba_i)

        target_branch_length.append(target_branch_length_sample)
        mutscale.append(mutscale_sample)

    return target_branch_length, mutscale  ## num_samples x num_trees x num_branches


if __name__ == "__main__":
    load_gamma(
        "../real_apr23/1000G_sub_aDNA_Mar2023/result/1000G_sub_Nea_pp_v2_chr1.coal",
        ["Vindija", "CHB"],
        ["Vindija", "CHB", "YRI"],
    )
