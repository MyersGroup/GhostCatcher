#!/usr/bin/env python
"""PCA + silhouette on the coal-count matrix for one GB run prefix.

Adapted from clean/RelateLocalAncestry/plotting/visualize_pca.py. Builds the
per-site feature matrix from the fixed_params pickles:
  num   = proportion_of_coalescing (coal counts) per reference group, summed over
          the analysis epochs [start_index, end_index]
  denom = opportunity, summed over the same epochs
Labels each site by its k=2 posterior argmax component. Then:
  1. silhouette_score on the COAL-COUNT matrix (num) with the k=2 labels
     -> how separated the two components are (high = real structure, ~0 = noise)
  2. PCA(4) KDE plot of PC1-PC2 / PC3-PC4 per component -> <out>_pca.svg
Usage: python pca_silhouette.py <output_prefix> <plot_out.svg>
Prints: SILHOUETTE_NUM <v>  SILHOUETTE_X <v>  NSITES <n>
"""
import pickle, glob, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

prefix = sys.argv[1]
plot_out = sys.argv[2] if len(sys.argv) > 2 else prefix + "_pca.svg"

samples = sorted({int(f.split("sample")[-1].split(".")[0])
                  for f in glob.glob(prefix + "_fixed_params_chr*_sample*.pkl")})
chrs = sorted({int(f.split("chr")[-1].split("_sample")[0])
               for f in glob.glob(prefix + "_fixed_params_chr*_sample*.pkl")})

post_overall, num_overall, denom_overall = [], [], []
for sample in samples:
    post = pd.read_csv(glob.glob(prefix + f"_overall_membership_*_sample_id_{sample}.csv")[0], sep=r"\s+")
    post_overall.extend(post[[f"prob_{i}" for i in range(post.shape[1]-3)]].values)
    for chrom in chrs:
        with open(prefix + f"_fixed_params_chr{chrom}_sample{sample}.pkl", "rb") as f:
            data = pickle.load(f)
        off = len(data) - 17   # 1 if sample_id is the leading field, else 0
        ignore_first, ignore_last = data[5+off], data[6+off]
        denom       = data[10+off]
        prop_coal   = data[12+off]
        epoch_index = data[13+off]
        _, ng, ne = denom.shape
        si = 1 if ignore_first else 0
        ei = ne-2 if ignore_last else ne-1
        for i in range(len(prop_coal)):
            sp = np.zeros(ng)
            for c in range(len(prop_coal[i])):
                if si <= epoch_index[i][c] <= ei:
                    sp = sp + np.array(prop_coal[i][c])/np.sum(prop_coal[i][c])
            num_overall.append(sp)
            denom_overall.append(np.sum(denom[i,:,si:ei+1], axis=1))

post_overall = np.array(post_overall)
num = np.array(num_overall, dtype=float); denom = np.array(denom_overall, dtype=float)
num = num[:, np.sum(num,axis=0)!=0]; denom = denom[:, np.sum(denom,axis=0)!=0]
def z(a):
    a = a - np.mean(a,axis=0); s = np.std(a,axis=0); s[s==0]=1; a = a/s
    return a[:, ~np.isnan(a).any(axis=0)]
num, denom = z(num), z(denom)
X = np.hstack((num, denom))
labels = np.argmax(post_overall, axis=1)

# silhouette (subsample for speed) on coal-count matrix and on full feature matrix
rng = np.random.default_rng(0)
n = len(labels)
idx = rng.choice(n, min(n, 10000), replace=False)
def sil(M):
    lab = labels[idx]
    if len(np.unique(lab)) < 2: return np.nan
    return silhouette_score(M[idx], lab)
s_num, s_X = sil(num), sil(X)
print(f"SILHOUETTE_NUM {s_num:.4f}")
print(f"SILHOUETTE_X {s_X:.4f}")
print(f"NSITES {n}")

# PCA KDE plot. Feature count can be <4 (e.g. ghost=1: only one reference group
# survives, so X has 2 cols) -> cap n_components and plot only the PC-pairs we have.
nc = min(4, X.shape[1])
print(f"NFEATURES {X.shape[1]}")
pca = PCA(n_components=nc); Xr = pca.fit_transform(X)
pcs = [f"PC{i}" for i in range(1,nc+1)]
d = pd.DataFrame(Xr, columns=pcs)
d["comp"] = ["Component "+str(l+1) for l in labels]
if len(d) > 100000: d = d.sample(100000, random_state=0)
comps = np.sort(d["comp"].unique())
q = {c: (d[c].quantile(.02), d[c].quantile(.98)) for c in pcs}
pairs = [(pcs[i],pcs[i+1]) for i in range(0,nc-1,2)]  # (PC1,PC2),(PC3,PC4)...
pal = ["purple","green","red","blue","orange"]
fig, ax = plt.subplots(len(pairs), len(comps), figsize=(5*len(comps),5*len(pairs)), squeeze=False)
for r,(px,py) in enumerate(pairs):
    for j,c in enumerate(comps):
        sub = d[d["comp"]==c]
        sns.kdeplot(data=sub,x=px,y=py,ax=ax[r,j],fill=True,color=pal[j%len(pal)],cut=0,
                    clip=(q[px],q[py]))
        ax[r,j].set(xlim=q[px],ylim=q[py])
        if r==0: ax[r,j].set_title(f"{c}  (sil_num={s_num:.2f})")
plt.tight_layout(); plt.savefig(plot_out, dpi=200); plt.close()
print("wrote", plot_out)
