#!/usr/bin/env python
"""LDA-based silhouette + PCA plot for one GhostBuster run.

Per site, from the fixed_params pickles: num = coalescence composition across
reference groups (summed over the analysis epochs), denom = opportunity.
X = z([num | denom]); sites labeled by k=2 posterior argmax. A cross-validated
Fisher-LDA projection of X (which auto-weights the informative references) is
scored by silhouette + held-out AUC.

Prints: SILHOUETTE_LDA  LDA_CV_AUC  NSITES
"""
import pickle, glob, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

p = argparse.ArgumentParser(description="LDA-based silhouette + PCA plot for one GhostBuster run.")
p.add_argument("prefix", help="path prefix for *_fixed_params_chr*_sample*.pkl and *_overall_membership_*.csv")
p.add_argument("-o", "--plot_out", help="output PCA plot (default: <prefix>_pca.svg)")
args = p.parse_args()
prefix = args.prefix
plot_out = args.plot_out or prefix + "_pca.svg"
rng = np.random.default_rng(0)

samples = sorted({int(f.split("sample")[-1].split(".")[0])
                  for f in glob.glob(prefix + "_fixed_params_chr*_sample*.pkl")})
chrs = sorted({int(f.split("chr")[-1].split("_sample")[0])
               for f in glob.glob(prefix + "_fixed_params_chr*_sample*.pkl")})

post_all, num_all, den_all = [], [], []
for s in samples:
    mf = glob.glob(prefix + f"_overall_membership_*_sample_id_{s}.csv")
    if not mf: continue
    post = pd.read_csv(mf[0], sep=r"\s+")
    P = post[[f"prob_{i}" for i in range(post.shape[1]-3)]].values
    num_s, den_s = [], []
    for c in chrs:
        with open(prefix + f"_fixed_params_chr{c}_sample{s}.pkl", "rb") as f: data = pickle.load(f)
        off = len(data) - 17
        denom, prop, eidx = data[10+off], data[12+off], data[13+off]
        _, ng, ne = denom.shape
        si = 1 if data[5+off] else 0
        ei = ne-2 if data[6+off] else ne-1
        for i in range(len(prop)):
            sp = np.zeros(ng)
            for cc in range(len(prop[i])):
                if si <= eidx[i][cc] <= ei: sp += np.array(prop[i][cc]) / np.sum(prop[i][cc])
            num_s.append(sp); den_s.append(np.sum(denom[i, :, si:ei+1], axis=1))
    if len(num_s) != len(P): continue
    idx = np.arange(len(num_s))
    if len(idx) > 8000: idx = rng.choice(idx, 8000, replace=False)
    num_all += [num_s[i] for i in idx]; den_all += [den_s[i] for i in idx]; post_all += list(P[idx])

def z(a):
    a = np.array(a, float); a = a[:, a.std(0) > 0]
    a = (a - a.mean(0)) / a.std(0); return a[:, ~np.isnan(a).any(0)]
X = np.hstack((z(num_all), z(den_all)))
labels = np.argmax(np.array(post_all), axis=1)

# subsample, then cross-validated LDA projection -> silhouette + AUC
n = len(labels); ii = rng.choice(n, min(n, 10000), replace=False)
X, labels = X[ii], labels[ii]
if len(np.unique(labels)) < 2 or np.bincount(labels).min() < 10:
    print("SILHOUETTE_LDA nan"); print("LDA_CV_AUC nan"); print(f"NSITES {n}"); sys.exit(0)
scores = np.zeros(len(labels))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, labels):
    scores[te] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], labels[tr]).decision_function(X[te])
sil = silhouette_score(scores.reshape(-1, 1), labels)
auc = roc_auc_score(labels, scores)
print(f"SILHOUETTE_LDA {sil:.4f}")
print(f"LDA_CV_AUC {auc:.4f}")
print(f"NSITES {n}")

# PCA KDE plot
nc = min(4, X.shape[1]); Xr = PCA(nc).fit_transform(X)
pcs = [f"PC{i}" for i in range(1, nc+1)]; d = pd.DataFrame(Xr, columns=pcs)
d["comp"] = [f"Component {l+1}" for l in labels]
q = {c: (d[c].quantile(.02), d[c].quantile(.98)) for c in pcs}
pairs = [(pcs[i], pcs[i+1]) for i in range(0, nc-1, 2)]
comps = np.sort(d["comp"].unique()); pal = ["purple", "green"]
fig, ax = plt.subplots(len(pairs), len(comps), figsize=(5*len(comps), 5*len(pairs)), squeeze=False)
for r, (px, py) in enumerate(pairs):
    for j, c in enumerate(comps):
        sub = d[d["comp"] == c]
        sns.kdeplot(data=sub, x=px, y=py, ax=ax[r, j], fill=True, color=pal[j % 2], cut=0, clip=(q[px], q[py]))
        ax[r, j].set(xlim=q[px], ylim=q[py])
        if r == 0: ax[r, j].set_title(f"{c} (sil_LDA={sil:.2f})")
plt.tight_layout(); plt.savefig(plot_out, dpi=200); plt.close()
print("wrote", plot_out)
