"""
calc_tree_stats.py: contains code to calculate various tree statistics used for filtering
"""

import numpy as np
import math
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
import pickle
import msprime
import copy
import pdb
import os
import bisect
MASK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_data_mask"
)


def compute_tree_stats(
    args,
    poplabels,
    ts_list,
    chrs,
    rec,
    sample_list=None,
    force_build=1,
):
    tree_size = []
    tree_left_bp = []
    tree_right_bp = []
    tree_left_bp_gen = []
    tree_right_bp_gen = []
    no_of_mutations = []
    recomb_window_size = 50000  ## window size for measure recombination rates
    recomb_rates = []
    chr_map = []
    count = 0
    num_nodes = len(list(ts_list[0].first().nodes()))
    first_tree_nodes = list(ts_list[0].first().nodes())[0:-1]

    if args.genome_build in ["hg37", "hg38"]:
        cent_telo_hla = pd.read_csv(
            os.path.join(MASK_DIR, str(args.genome_build) + "_real_data_mask.txt"),
            sep="\t",
        )
    elif args.genome_build == "hg19":
        cent_telo_hla = pd.read_csv(
            os.path.join(MASK_DIR, "hg37_real_data_mask.txt"),
            sep="\t",
        )
    elif args.genome_build is None:
        print("Caution: Not using any filter to remove HLA/Centromere/Telomere regions")
        cent_telo_hla = pd.DataFrame(columns=["chr", "start", "end", "description"])
    else:
        print("Make sure to use genome_build either hg37 or hg38")
        cent_telo_hla = pd.read_csv(
            os.path.join(MASK_DIR, "real_data_mask.txt"), sep="\t"
        )

    for chr_no, chr in enumerate(chrs):
        if os.path.isfile(rec + str(chr) + ".txt"):
            recomb_map = pd.read_csv(
                rec + str(chr) + ".txt",
                sep=r"\s+",
            )
            recomb_map_msprime = msprime.RateMap.read_hapmap(rec + str(chr) + ".txt")
        elif os.path.isfile(rec + str(chr) + ".txt.gz"):
            recomb_map = pd.read_csv(
                rec + str(chr) + ".txt.gz",
                sep=r"\s+",
            )
            recomb_map_msprime = msprime.RateMap.read_hapmap(rec + str(chr) + ".txt.gz")
        else:
            raise "Recomb map format not identified"
        recomb_map_arr = np.array(recomb_map[recomb_map.columns[1:]])
        recomb_map["Start Position(bp)"] = np.array(
            [recomb_map_arr[0, 0]] + recomb_map_arr[:-1, 0].tolist()
        )
        tree_left_bp_chr, tree_right_bp_chr, bvalues_chr = [], [], []
        ts = ts_list[count]
        count += 1
        tree = ts.first()
        ts_edges = ts.edges()
        for tid in tqdm(range(ts.num_trees)):  # len(list(ts.trees()))
            if (
                np.ceil(tree.interval[1] / force_build)
                - np.ceil(tree.interval[0] / force_build)
                > 0
            ):
                tree_size.append(tree.interval[1] - tree.interval[0])
                tree_left_bp_chr.append(tree.interval[0])
                tree_right_bp_chr.append(tree.interval[1])

                num_muts = []
                for node in tree.nodes():
                    edge_id = tree.edge(node)
                    edge = ts_edges[edge_id]
                    num_muts.append(
                        int(edge.metadata.decode("utf-8").rstrip("\x00").split(" ")[2])
                    )

                num_muts = np.array(num_muts)
                no_of_mutations.append(np.mean(num_muts > 0))
                chr_map.append(chr)
                if (
                    (
                        tree.interval[0]
                        >= cent_telo_hla[cent_telo_hla.chr == str(chr)].start - 500000
                    )
                    & (
                        tree.interval[1]
                        < cent_telo_hla[cent_telo_hla.chr == str(chr)].end + 500000
                    )
                ).any():
                    recomb_rate = np.nan
                else:
                    start_recomb_window = np.clip(
                        tree.interval[0] - recomb_window_size,
                        recomb_map_msprime.position.min(),
                        recomb_map_msprime.position.max(),
                    )
                    end_recomb_window = np.clip(
                        tree.interval[1] + recomb_window_size,
                        recomb_map_msprime.position.min(),
                        recomb_map_msprime.position.max(),
                    )
                    recomb_rate = (
                        100
                        * (
                            recomb_map_msprime.get_cumulative_mass(end_recomb_window)
                            - recomb_map_msprime.get_cumulative_mass(
                                start_recomb_window
                            )
                        )
                        / (end_recomb_window - start_recomb_window)
                    )
                recomb_rates.append(recomb_rate)
            tree.next()
        del tree
        del ts
        tree_left_bp_gen.extend(
            recomb_map_msprime.get_cumulative_mass(
                np.minimum(
                    np.maximum(tree_left_bp_chr, recomb_map_msprime.position.min()),
                    recomb_map_msprime.position.max(),
                )
            ).tolist()
        )
        tree_right_bp_gen.extend(
            recomb_map_msprime.get_cumulative_mass(
                np.minimum(
                    np.maximum(tree_right_bp_chr, recomb_map_msprime.position.min()),
                    recomb_map_msprime.position.max(),
                )
            ).tolist()
        )
        tree_left_bp.extend(tree_left_bp_chr)
        tree_right_bp.extend(tree_right_bp_chr)

    return (
        tree_size,
        tree_left_bp,
        tree_right_bp,
        tree_left_bp_gen,
        tree_right_bp_gen,
        no_of_mutations,
        recomb_rates,
        chr_map,
    )


def load_tree_stats(args, ts_list, poplabels, tree_stats_file_prefix=None):
    chrs = list(map(int, args.chrs.split(",")))
    sample_id_label = "_".join([str(e) for e in args.sample_id])
    recomb_rates_all = []
    tree_left_bp_all = []
    tree_right_bp_all = []
    tree_left_bp_gen_all = []
    tree_right_bp_gen_all = []
    chr_map_all = []
    frac_branches_with_snp_all = []

    for chrom_no, chrom in enumerate(chrs):
        if tree_stats_file_prefix is not None:
            tree_stats_file_name = tree_stats_file_prefix + "_chr" + str(chrom) + ".pkl"
        else:
            tree_stats_file_name = args.output + "_tree_stats_chr" + str(chrom) + ".pkl"
        try:
            f_pkl = open(tree_stats_file_name, "rb")
            (
                tree_size,
                tree_left_bp,
                tree_right_bp,
                tree_left_bp_gen,
                tree_right_bp_gen,
                no_of_mutations,
                recomb_rates,
                chr_map,
            ) = pickle.load(f_pkl)
            f_pkl.close()
            print("Done loading tree statistics from: " + str(tree_stats_file_name))
        except:
            print("Tree statistics file not found, calculating tree statistics..")
            ## mapping samples back to their original names
            (
                tree_size,
                tree_left_bp,
                tree_right_bp,
                tree_left_bp_gen,
                tree_right_bp_gen,
                no_of_mutations,
                recomb_rates,
                chr_map,
            ) = compute_tree_stats(
                args,
                poplabels,
                ts_list[chrom_no : chrom_no + 1],
                [chrom],
                args.rec,
                args.sample_id,
                args.force_build,
            )

            f_pkl = open(tree_stats_file_name, "wb")
            pickle.dump(
                [
                    tree_size,
                    tree_left_bp,
                    tree_right_bp,
                    tree_left_bp_gen,
                    tree_right_bp_gen,
                    no_of_mutations,
                    recomb_rates,
                    chr_map,
                ],
                f_pkl,
            )
            f_pkl.close()
            print("Tree statistics stored in: " + str(tree_stats_file_name))
        recomb_rates_all.extend(recomb_rates)
        tree_left_bp_all.extend(tree_left_bp)
        tree_right_bp_all.extend(tree_right_bp)
        tree_left_bp_gen_all.extend(tree_left_bp_gen)
        tree_right_bp_gen_all.extend(tree_right_bp_gen)
        chr_map_all.extend(chr_map)
        frac_branches_with_snp_all.extend(no_of_mutations)

    return (
        recomb_rates_all,
        tree_left_bp_all,
        tree_right_bp_all,
        tree_left_bp_gen_all,
        tree_right_bp_gen_all,
        chr_map_all,
        frac_branches_with_snp_all,
    )
