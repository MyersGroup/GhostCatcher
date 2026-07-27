"""Estimate the divergence time between two populations from coalescence rates
along Relate/true trees.

For every local tree we count, at each internal node, how many within-pop1
(A-A), within-pop2 (B-B) and cross (A-B) sample pairs have their MRCA there,
from subtree counts (so ALL pairs are used exactly, no pairwise loop).

The coalescence rate in a time epoch is

    R(epoch) = N(epoch) / O(epoch)

    N = number of coalescence events in the epoch        (raw count, NOT span-weighted)
    O = "opportunity" = at-risk lineage-time in the epoch (span-weighted, Eq. 10):
        each (pair, tree) contributes tree_span * (time it is at-risk within the
        epoch) = span*(t-lo) if it coalesces inside, span*(hi-lo) if still
        un-coalesced at hi, 0 if it coalesced before lo.

Rates are averaged over `n_sets` random sets of `n_epochs` epoch boundaries (an
average-shifted-histogram smoother) read off at `n_eval` points. Divergence =
where rCCR(t) = R_AB / (0.5*(R_AA + R_BB)) crosses --rccr_threshold (default 0.9);
95% CI by genomic block bootstrap (contiguous tree blocks within each chromosome).

Optional local-ancestry masking (--mask1/--mask2): restrict a population to the
positions where each of its samples has ancestry > --local_ancestry_threshold, read from
per-sample files <mask>_<sample>.csv. Samples with no such file are dropped from
that population. e.g. OOA-vs-Eurasian divergence:
    --pops afr Eurasian --mask1 ground_truth/ooa_in_afr --mask1_colname prob_0

Usage:
  python estimate_divergence.py --prefix example/relate_chr --chrs 22 \
      --poplabels example/poplabels.txt --pops focal B \
      --output output/div_focal_B --epoch_start 3.5
"""

import argparse
import glob
import os

import matplotlib
import msprime
import numpy as np
import pandas as pd
import tskit
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPES = ("AA", "BB", "AB")


def load_mask(prefix, nodes, chrom, col, thresh, invert=False):
    """Per-sample ancestry mask for sample `nodes` on chromosome `chrom`. The
    per-sample file is either <prefix>_<sample>.csv (ground truth) or a GB
    membership file <prefix>_overall_membership_*sample_id_<sample>.csv (inferred
    local ancestry). Samples with no file are dropped. Returns (kept_nodes,
    grid_positions, active[kept_sample, position] bool), where active =
    (column `col` > thresh), or (< thresh) if invert (complement, e.g. non-OOA)."""
    kept, pos, act = [], None, []
    for s in nodes:
        fn = f"{prefix}_{int(s)}.csv"
        if not os.path.exists(fn):  # fall back to GB membership naming
            g = glob.glob(f"{prefix}_overall_membership_*sample_id_{int(s)}.csv")
            if not g:
                continue
            fn = g[0]
        df = pd.read_csv(fn, sep="\t")  # tab: empty ancestry cells -> NaN (excluded), not shifted
        df = df[df["chr"] == chrom]
        if pos is None:
            pos = df["pos"].to_numpy()
        v = df[col].to_numpy()
        act.append(v < thresh if invert else v > thresh)
        kept.append(s)
    return np.asarray(kept), pos, np.array(act)


def tree_recomb_rates(ts, recomb, window):
    """Mean recombination rate (cM/Mb) over [left-window, right+window] for EVERY
    tree, vectorised off the breakpoints (no tree traversal needed)."""
    bp = ts.breakpoints(as_array=True)
    lo = np.clip(bp[:-1] - window, recomb.position[0], recomb.position[-1])
    hi = np.clip(bp[1:] + window, recomb.position[0], recomb.position[-1])
    mass = recomb.get_cumulative_mass(hi) - recomb.get_cumulative_mass(lo)
    return mass / np.maximum(hi - lo, 1.0) * 1e8  # Morgans/bp -> cM/Mb


def collect_chr(ts, baseA, baseB, maskA, maskB, edges, n_blocks, recomb=None,
                recomb_pct=None, recomb_window=1e5, max_span=None, grid_snap=True):
    """Per contiguous block of trees, build the histograms needed for the rate:
      FW   = span-weighted #coalescences per fine bin   (numerator and opportunity)
      FT   = FW weighted by coalescence time            (for the opportunity)
      UW   = span-weighted #coalescences below t_min    (so #coalesced-by-hi is right)
      TOTW = span-weighted TOTAL #pairs                 (the at-risk pool)
    Blocks are contiguous tree ranges within THIS chromosome only. If `recomb` (an
    msprime RateMap) is given, trees are ranked by their mean recombination rate over
    [left-recomb_window, right+recomb_window] and the highest-rate `recomb_pct` PERCENT
    are removed, keeping the lowest-recombination (100-recomb_pct)%. The cutoff is a
    per-chromosome quantile of the eligible (max_span-passing) trees, so it adapts to
    the map rather than fixing an absolute cM/Mb. If `max_span` is given, trees wider
    than max_span bp are skipped (excludes centromere/telomere/assembly-gap-spanning
    trees, which pass the low-recomb filter and otherwise dominate via span-weighting)."""
    nfine = len(edges) - 1
    N = ts.num_nodes
    tnode = ts.tables.nodes.time
    samp = np.zeros(N, bool)  # which node ids are sample leaves
    samp[ts.samples()] = True

    FW = {k: np.zeros((n_blocks, nfine)) for k in TYPES}
    FT = {k: np.zeros((n_blocks, nfine)) for k in TYPES}
    UW = {k: np.zeros(n_blocks) for k in TYPES}
    TOTW = {k: np.zeros(n_blocks) for k in TYPES}

    bnd = np.linspace(0, ts.num_trees, n_blocks + 1).astype(int)
    blk = np.clip(
        np.searchsorted(bnd, np.arange(ts.num_trees), "right") - 1, 0, n_blocks - 1
    )

    # per-tree recomb rates + the quantile cutoff that removes the top recomb_pct%
    rates = rate_cut = None
    if recomb is not None and recomb_pct is not None:
        rates = tree_recomb_rates(ts, recomb, recomb_window)
        elig = np.ones(len(rates), bool)
        if max_span is not None:  # rank only among trees we'd otherwise use
            elig = np.diff(ts.breakpoints(as_array=True)) <= max_span
        rate_cut = np.percentile(rates[elig], 100.0 - recomb_pct)
        print(f"  recomb: removing top {recomb_pct:g}% of trees by rate "
              f"-> keep rate < {rate_cut:.3f} cM/Mb", flush=True)

    # persistent per-pop buffers for masking: only the masked sample entries change
    # from tree to tree, so copy the (ts.num_nodes-long) base ONCE here instead of
    # every tree -- the per-tree copy was O(num_nodes) and dominated masked runs.
    curA = baseA.copy() if maskA is not None else None
    curB = baseB.copy() if maskB is not None else None

    for ti, tree in tqdm(enumerate(ts.trees()), total=ts.num_trees):
        if tree.num_edges == 0:
            continue
        if max_span is not None and tree.span > max_span:
            continue  # skip centromere/telomere/gap-spanning trees
        if rates is not None and rates[ti] >= rate_cut:
            continue  # drop the highest-recombination trees
        b = blk[ti]
        span = tree.span
        # isA/isB: sample nodes belonging to pop A/B in THIS tree. With a mask, a
        # sample is kept only where its ancestry passes threshold at the tree
        # midpoint (mask = (sample_nodes, grid_positions, active[sample,pos])).
        isA, isB = baseA, baseB
        if grid_snap and (maskA is not None or maskB is not None):
            # only use trees that contain a grid point (10kb multiple); ancestry is
            # constant within a marginal tree, so the grid value there is EXACT --
            # avoids the nearest-grid-point misassignment of the midpoint lookup.
            skip = False
            for mask in (maskA, maskB):
                if mask is not None:
                    gi = np.searchsorted(mask[1], tree.interval.left, "left")
                    if gi >= len(mask[1]) or mask[1][gi] >= tree.interval.right:
                        skip = True
                        break
            if skip:
                continue
        elif maskA is not None or maskB is not None:
            mid = (tree.interval.left + tree.interval.right) / 2
        for mask, cur, flag in ((maskA, curA, "A"), (maskB, curB, "B")):
            if mask is not None:
                if grid_snap:  # the grid point that falls inside this tree
                    g = min(np.searchsorted(mask[1], tree.interval.left, "left"),
                            mask[2].shape[1] - 1)
                else:
                    g = min(max(np.searchsorted(mask[1], mid, "right") - 1, 0),
                            mask[2].shape[1] - 1)
                cur[mask[0]] = mask[2][:, g]  # only rewrite the masked sample entries
                if flag == "A":
                    isA = cur
                else:
                    isB = cur
        # all arrays below are sized to THIS tree's nodes only (a few hundred),
        # not ts.num_nodes; `node` maps local index -> global id, `loc` the inverse
        order = list(tree.nodes(order="postorder"))  # children before parents
        node = np.fromiter(order, np.int64)
        n = node.size
        loc = {u: i for i, u in enumerate(order)}  # global node id -> local index
        nA = isA[node].astype(np.int64)  # local #pop-A samples in subtree (leaf flags)
        nB = isB[node].astype(np.int64)
        ssA = np.zeros(n)  # sum over children of (#A in child)^2  (ditto ssB, sAB)
        ssB = np.zeros(n)
        sAB = np.zeros(n)
        mA, mB = int(nA.sum()), int(nB.sum())  # #active A / B samples in this tree
        TOTW["AA"][b] += span * mA * (mA - 1) / 2  # every pair coalesces somewhere
        TOTW["BB"][b] += span * mB * (mB - 1) / 2
        TOTW["AB"][b] += span * mA * mB
        for li, u in enumerate(order):  # push subtree counts up to parents
            p = tree.parent(u)
            if p != tskit.NULL:
                pl = loc[p]
                au, bu = nA[li], nB[li]
                nA[pl] += au
                nB[pl] += bu
                ssA[pl] += au * au
                ssB[pl] += bu * bu
                sAB[pl] += au * bu
        # #pairs whose MRCA is each node = pairs split across that node's children
        c = {"AA": (nA * nA - ssA) / 2, "BB": (nB * nB - ssB) / 2, "AB": nA * nB - sAB}
        tn, sm = tnode[node], samp[node]
        for k in TYPES:
            cnt = c[k]
            cnt[sm] = 0.0  # leaves are not MRCAs
            nz = cnt > 0
            if not nz.any():
                continue
            t = tn[nz]
            cc = cnt[nz]  # raw #pairs coalescing at each node (numerator)
            ww = cc * span  # span-weighted (opportunity)
            under = t < edges[0]
            UW[k][b] += ww[under].sum()
            inr = (~under) & (t < edges[-1])
            if inr.any():
                idx = np.searchsorted(edges, t[inr], "right") - 1
                np.add.at(FW[k][b], idx, ww[inr])
                np.add.at(FT[k][b], idx, ww[inr] * t[inr])
    return FW, FT, UW, TOTW


def smooth_rates(sel, FW, FT, UW, TOTW, edges, evals, sets):
    """Rate at each eval point, mean of per-set ratios over the epoch `sets`, for
    block subset `sel`."""
    R = {}
    for k in TYPES:
        cw = np.concatenate(
            [[0.0], np.cumsum(FW[k][sel].sum(0))]
        )  # cw[j] = span-weighted #coal with time < edges[j]
        ct = np.concatenate(
            [[0.0], np.cumsum(FT[k][sel].sum(0))]
        )  # span-weighted, *time
        uw = UW[k][sel].sum()
        tot = TOTW[k][sel].sum()  # total span-weighted #pairs
        acc = np.zeros(len(evals))
        for ks in sets:  # ks = fine-edge indices of one epoch set
            lo, hi = edges[ks[:-1]], edges[ks[1:]]  # each epoch [lo, hi)
            n_spn = cw[ks[1:]] - cw[ks[:-1]]  # span-weighted #coal in epoch
            intT = ct[ks[1:]] - ct[ks[:-1]]  # span-weighted sum of times in epoch
            W_hi = uw + cw[ks[1:]]  # span-weighted #pairs coalesced by hi
            # opportunity: (at-risk time for pairs coalescing inside)
            #            + (epoch width)*(#pairs still at risk at hi)
            opp = (intT - lo * n_spn) + (hi - lo) * (tot - W_hi)
            rate = np.where(opp > 0, n_spn / np.maximum(opp, 1e-300), 0.0)
            acc += rate[np.clip(np.searchsorted(hi, evals, "right"), 0, len(rate) - 1)]
        R[k] = acc / len(sets)  # mean of per-set ratios (ASH)
    return R


def crossing(R, evals, rccr_thresh=0.5):
    """First time (log-interpolated) where rCCR rises through 0.5."""
    denom = 0.5 * (R["AA"] + R["BB"])
    rccr = np.where(denom > 0, R["AB"] / np.maximum(denom, 1e-300), np.nan)
    f = rccr - rccr_thresh
    for i in range(1, len(evals)):
        if np.isfinite(f[i - 1]) and np.isfinite(f[i]) and f[i - 1] < 0 <= f[i]:
            x0, x1 = np.log(evals[i - 1]), np.log(evals[i])
            return float(np.exp(x0 - f[i - 1] * (x1 - x0) / (f[i] - f[i - 1]))), rccr
    return np.nan, rccr


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefix", required=True, help="trees prefix; reads <prefix><chr>.trees"
    )
    p.add_argument("--chrs", required=True, nargs="+", type=int)
    p.add_argument("--poplabels", required=True)
    p.add_argument("--pops", required=True, nargs=2, help="GROUP labels pop1 pop2")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--mask1", default=None, help="per-sample ancestry prefix restricting pop1"
    )
    p.add_argument(
        "--mask2", default=None, help="per-sample ancestry prefix restricting pop2"
    )
    p.add_argument("--mask1_colname", default="prob_0")
    p.add_argument("--mask2_colname", default="prob_0")
    p.add_argument("--local_ancestry_threshold", type=float, default=0.5)
    p.add_argument("--mask1_invert", action="store_true", help="keep pop1 where col < thresh")
    p.add_argument("--mask2_invert", action="store_true", help="keep pop2 where col < thresh")
    p.add_argument("--no_grid_snap", action="store_true",
                   help="disable grid-snap (ON by default): grid-snap uses only trees "
                        "containing an ancestry grid point and reads the mask there "
                        "(exact), avoiding nearest-grid-point midpoint leakage")
    p.add_argument("--recomb_map", default=None,
                   help="recomb-map prefix (reads <recomb_map><chr>.txt via read_hapmap); "
                        "restrict to low-recombination trees")
    p.add_argument("--recomb_pct_removed", type=float, default=50.0,
                   help="percent of trees to REMOVE, highest recombination rate first "
                        "(50 = keep the lowest-recomb half). Quantile cutoff, so it "
                        "adapts to the map instead of fixing an absolute cM/Mb.")
    p.add_argument("--recomb_window", type=float, default=5e4, help="bp window each side")
    p.add_argument("--max_span", type=float, default=1e6,
                   help="skip trees wider than this (bp); excludes centromere/gap-spanning trees")
    p.add_argument(
        "--rccr_threshold", default="0.9,0.9",
        help="rCCR crossing threshold as a single value or 'lo,hi' range; a point is "
             "sampled uniformly at random from the range per bootstrap replicate. "
             "e.g. 0.5 or 0.5,0.5 (fixed) or 0.5,0.9 (range).",
    )
    p.add_argument("--gen", type=float, default=28.0)
    p.add_argument(
        "--epoch_start", type=float, default=3.0, help="log10 years (recent epoch edge)"
    )
    p.add_argument(
        "--epoch_end", type=float, default=7.0, help="log10 years (old epoch edge)"
    )
    p.add_argument("--n_epochs", type=int, default=40)
    p.add_argument("--n_sets", type=int, default=1000)
    p.add_argument("--n_fine", type=int, default=100)
    p.add_argument("--n_blocks", type=int, default=1000, help="total bootstrap blocks")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dump_rates", action="store_true",
                   help="also save <output>_rates.npz with the per-bootstrap ICR curves "
                        "(icr_AA/BB/AB, years yr) for overlaying cross-coalescence curves")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    pop1, pop2 = args.pops
    G = args.gen  # years per generation (times are in generations internally)
    # fine grid (n_fine+1 edges) and eval points (n_fine, the fine bins), both
    # uniform in log10(years), converted to generations
    edges = 10 ** np.linspace(args.epoch_start, args.epoch_end, args.n_fine + 1) / G
    evals = 10 ** np.linspace(args.epoch_start, args.epoch_end, args.n_fine) / G
    bpc = max(
        1, round(args.n_blocks / len(args.chrs))
    )  # contiguous blocks per chromosome

    FW = {k: [] for k in TYPES}
    FT = {k: [] for k in TYPES}
    UW = {k: [] for k in TYPES}
    TOTW = {k: [] for k in TYPES}
    for chrom in args.chrs:
        ts = tskit.load(f"{args.prefix}{chrom}.trees")
        groups = pd.read_csv(args.poplabels, sep=r"\s+")["GROUP"].values
        s = ts.samples()
        A_nodes, B_nodes = s[groups == pop1], s[groups == pop2]
        maskA = maskB = None
        if args.mask1:
            maskA = load_mask(
                args.mask1, A_nodes, chrom, args.mask1_colname, args.local_ancestry_threshold,
                args.mask1_invert,
            )
            A_nodes = maskA[0]  # drop pop1 samples lacking ancestry files
        if args.mask2:
            maskB = load_mask(
                args.mask2, B_nodes, chrom, args.mask2_colname, args.local_ancestry_threshold,
                args.mask2_invert,
            )
            B_nodes = maskB[0]
        baseA = np.zeros(ts.num_nodes, bool)
        baseA[A_nodes] = True
        baseB = np.zeros(ts.num_nodes, bool)
        baseB[B_nodes] = True
        recomb = msprime.RateMap.read_hapmap(f"{args.recomb_map}{chrom}.txt") if args.recomb_map else None
        fw, ft, uw, totw = collect_chr(ts, baseA, baseB, maskA, maskB, edges, bpc,
                                       recomb, args.recomb_pct_removed, args.recomb_window,
                                       args.max_span, not args.no_grid_snap)
        for k in TYPES:
            FW[k].append(fw[k])
            FT[k].append(ft[k])
            UW[k].append(uw[k])
            TOTW[k].append(totw[k])
        print(
            f"chr{chrom}: {ts.num_trees} trees, {pop1}={len(A_nodes)} {pop2}={len(B_nodes)}",
            flush=True,
        )
    FW = {k: np.concatenate(FW[k]) for k in TYPES}
    FT = {k: np.concatenate(FT[k]) for k in TYPES}
    UW = {k: np.concatenate(UW[k]) for k in TYPES}
    TOTW = {k: np.concatenate(TOTW[k]) for k in TYPES}
    n_blocks = len(TOTW["AA"])

    # Epoch grids. n_sets > 1: ASH -- random uniform epoch boundaries, averaged
    # (mean of ratios). n_sets == 1: a single grid of `n_epochs` epochs with equal
    # AB opportunity per epoch (quantile-distributed boundaries), giving a stepwise
    # ICR; the fine grid (n_fine) still sets the eval/histogram resolution.
    if args.n_sets == 1:
        cwAB = np.concatenate([[0.0], np.cumsum(FW["AB"].sum(0))])
        opp_ab = (TOTW["AB"].sum() - cwAB[:-1]) * np.diff(edges)  # AB exposure per bin
        cdf = np.concatenate([[0.0], np.cumsum(opp_ab)])
        cdf = cdf / cdf[-1]
        idx = np.clip(np.searchsorted(cdf, np.arange(1, args.n_epochs) / args.n_epochs),
                      1, args.n_fine - 1)
        sets = [np.unique(np.concatenate([[0], idx, [args.n_fine]]))]
    else:
        sets = [
            np.concatenate([[0], np.sort(rng.choice(np.arange(1, args.n_fine),
                            args.n_epochs - 1, replace=False)), [args.n_fine]])
            for _ in range(args.n_sets)
        ]

    # rCCR crossing threshold: a single value or a [lo, hi] range from which a point
    # is drawn uniformly at random per bootstrap replicate.
    parts = [float(x) for x in args.rccr_threshold.split(",")]
    thr_lo, thr_hi = parts[0], parts[-1]

    div_boot = np.empty(args.n_boot)
    icr_boot = {k: np.empty((args.n_boot, len(evals))) for k in TYPES}
    for bi in range(args.n_boot):
        sel = rng.integers(0, n_blocks, n_blocks)  # resample blocks with replacement
        Rb = smooth_rates(sel, FW, FT, UW, TOTW, edges, evals, sets)
        thr = rng.uniform(thr_lo, thr_hi)  # sample a threshold in the range
        div_boot[bi], _ = crossing(Rb, evals, thr)
        for k in TYPES:
            icr_boot[k][bi] = np.where(
                Rb[k] > 0, 0.5 / np.maximum(Rb[k], 1e-300), np.nan
            )
    div0 = np.nanmean(div_boot)  # point estimate = mean over bootstraps
    lo, hi = np.nanpercentile(div_boot, [2.5, 97.5])
    if args.dump_rates:  # per-bootstrap ICR curves for cross-coalescence overlays
        np.savez(f"{args.output}_rates.npz", yr=evals * G, div_boot=div_boot,
                 **{f"icr_{k}": icr_boot[k] for k in TYPES})
        print(f"saved {args.output}_rates.npz")
    print(
        f"\nDivergence({pop1},{pop2}) = {div0 * G:,.0f} years (mean of bootstrap)  "
        f"rCCR threshold in [{thr_lo}, {thr_hi}]  "
        f"95% CI [{lo * G:,.0f}, {hi * G:,.0f}]  (n={np.isfinite(div_boot).sum()}/{args.n_boot})"
    )


if __name__ == "__main__":
    main()
