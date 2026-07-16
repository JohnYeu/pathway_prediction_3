#!/usr/bin/env python3
"""
PathwayML-Ath: Negative-sampling ratio robustness analysis
Evaluates XGBoost performance across pos:neg ratios 1:1 to 1:5

Usage:
    python robustness_analysis.py

Outputs:
    data_real/robustness_results.json   - per-ratio metrics
    figures_final/Fig7_robustness.png   - 6-panel figure
    figures_final/Fig7_robustness.pdf
"""

import json
import os
import random
import warnings
from itertools import combinations

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ── Paths and data loading ────────────────────────────────────────────────────
DATA_DIR = 'data_real'
FIG_DIR  = 'figures_final'
os.makedirs(FIG_DIR, exist_ok=True)

# Load the cached pathway definitions, gene-GO annotations, and feature config
all_pw   = json.load(open(f'{DATA_DIR}/all_pathways.json'))
gene_go  = json.load(open(f'{DATA_DIR}/gene_go.json'))
fi       = json.load(open(f'{DATA_DIR}/feature_info.json'))
SELECTED_GO   = fi['selected_go']       # 60 GO terms surviving feature selection
all_gene_pool = list(gene_go.keys())     # all GO-annotated genes as negative pool


def jaccard(s1, s2):
    """Jaccard similarity between two GO term sets."""
    u = s1 | s2
    return len(s1 & s2) / len(u) if u else 0.0


def build_vec(genes):
    """Build the 69-dimensional feature vector for one gene set.

    Layout: [60 GO frequencies, 4 Jaccard stats, gene_count, log_count,
    GO entropy, GO-size std, mean GO per gene].
    Pairwise Jaccard is capped at 15 genes for efficiency.
    """
    genes = [g for g in genes if g in gene_go]
    if not genes:
        return np.zeros(len(SELECTED_GO) + 9, dtype=np.float32)
    gs = [set(gene_go.get(g, [])) for g in genes]
    n  = len(genes)
    freq  = [sum(1 for s in gs if go in s) / n for go in SELECTED_GO]
    pairs = [jaccard(gs[i], gs[j])
             for i, j in combinations(range(min(n, 15)), 2)]
    sims  = pairs if pairs else [0.0]
    go_fa = np.array(freq) + 1e-9; go_fa /= go_fa.sum()
    entropy   = float(-np.sum(go_fa * np.log(go_fa + 1e-12)))
    go_sizes  = [len(s) for s in gs]
    return np.array(freq + [
        np.mean(sims), np.min(sims), np.max(sims), np.std(sims),
        float(n), float(np.log1p(n)), entropy,
        float(np.std(go_sizes)), float(np.mean(go_sizes))
    ], dtype=np.float32)


def make_negs(n, seed=42):
    """Generate n mixed hard-negative gene sets (4 types in equal proportions)."""
    random.seed(seed); np.random.seed(seed)
    pw_g = [list(info['genes'])
            for info in all_pw.values() if len(info['genes']) >= 5]
    out  = []
    for i in range(n):
        t = i % 4
        if t == 0:   # random
            out.append(random.sample(all_gene_pool, random.randint(5, 30)))
        elif t == 1:  # shuffled pathway (same size, random genes)
            out.append(random.sample(all_gene_pool, len(random.choice(pw_g))))
        elif t == 2:  # partial pathway (50-80% subsample)
            src = random.choice(pw_g)
            k   = max(3, int(len(src) * random.uniform(0.5, 0.8)))
            out.append(random.sample(src, min(k, len(src))))
        else:          # cross-pathway mixture
            s1, s2 = random.choice(pw_g), random.choice(pw_g)
            out.append(
                random.sample(s1, max(2, len(s1) // 2)) +
                random.sample(s2, max(2, len(s2) // 2)))
    return out


# ── Build positive feature vectors (fixed across all ratios) ─────────────────
pos_list = [(pid, list(info['genes']))
            for pid, info in all_pw.items() if len(info['genes']) >= 5]
Xp = np.array([build_vec(g) for pid, g in pos_list])
n_pos = len(Xp)
print(f"Positives: {n_pos}")

# ── Cross-validation folds (3 x 5-fold for robust estimation) ────────────────
cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cv3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

ratios  = [1, 2, 3, 4, 5]
results = {}

print(f"\n{'Ratio':6s} {'n_neg':7s} {'CV AUROC':14s} "
      f"{'Test AUROC':11s} {'AUPRC':7s} {'F1':7s} {'Prec':7s} {'Rec':7s}")
print("-" * 75)

for ratio in ratios:
    # Generate negatives at this ratio; scale_pos_weight compensates imbalance
    neg_lists = make_negs(n_pos * ratio, seed=42)
    Xn = np.array([build_vec(g) for g in neg_lists])
    X  = np.vstack([Xp, Xn])
    y  = np.array([1] * n_pos + [0] * len(Xn))

    XGB_params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=ratio,   # compensate imbalance
        min_child_weight=3, reg_alpha=0.1,
        eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)

    # 3 × 5-fold CV: 15 folds total for stable AUROC estimation
    all_cv_aucs = []
    for fold_cv in [cv1, cv2, cv3]:
        for tr_idx, vl_idx in fold_cv.split(X, y):
            m = XGBClassifier(**XGB_params)
            m.fit(X[tr_idx], y[tr_idx])
            yp = m.predict_proba(X[vl_idx])[:, 1]
            if len(np.unique(y[vl_idx])) > 1:
                all_cv_aucs.append(roc_auc_score(y[vl_idx], yp))

    cv_mean = np.mean(all_cv_aucs)
    cv_std  = np.std(all_cv_aucs)

    # Fixed 80:20 held-out test split (same seed across ratios for fair comparison)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    m_final = XGBClassifier(**XGB_params)
    m_final.fit(X_tr, y_tr)
    yp_te    = m_final.predict_proba(X_te)[:, 1]
    te_auroc = roc_auc_score(y_te, yp_te)
    te_auprc = average_precision_score(y_te, yp_te)
    te_f1    = f1_score(y_te, (yp_te >= 0.5).astype(int))
    te_prec  = precision_score(y_te, (yp_te >= 0.5).astype(int))
    te_rec   = recall_score(y_te, (yp_te >= 0.5).astype(int))

    print(f"1:{ratio}    {len(Xn):6d}  {cv_mean:.3f}±{cv_std:.3f}    "
          f"{te_auroc:.3f}       {te_auprc:.3f}  {te_f1:.3f}  "
          f"{te_prec:.3f}  {te_rec:.3f}")

    results[f'1:{ratio}'] = {
        'ratio':    f'1:{ratio}',
        'n_pos':    n_pos,
        'n_neg':    int(len(Xn)),
        'cv_aucs':  all_cv_aucs,
        'cv_mean':  float(cv_mean),
        'cv_std':   float(cv_std),
        'te_auroc': float(te_auroc),
        'te_auprc': float(te_auprc),
        'te_f1':    float(te_f1),
        'te_prec':  float(te_prec),
        'te_rec':   float(te_rec),
    }

json.dump(results, open(f'{DATA_DIR}/robustness_results.json', 'w'), indent=2)
print(f"\n✓ Saved {DATA_DIR}/robustness_results.json")


# ── Generate Figure 7 ─────────────────────────────────────────────────────────
ratio_labels = ['1:1', '1:2', '1:3', '1:4', '1:5']
n_negs       = [results[r]['n_neg']    for r in ratio_labels]
cv_means     = [results[r]['cv_mean']  for r in ratio_labels]
cv_stds      = [results[r]['cv_std']   for r in ratio_labels]
te_aurocs    = [results[r]['te_auroc'] for r in ratio_labels]
te_auprcs    = [results[r]['te_auprc'] for r in ratio_labels]
te_f1s       = [results[r]['te_f1']   for r in ratio_labels]
te_precs     = [results[r]['te_prec'] for r in ratio_labels]
te_recs      = [results[r]['te_rec']  for r in ratio_labels]
cv_all       = [results[r]['cv_aucs'] for r in ratio_labels]

B = '#2171B5'; G = '#238B45'; O = '#D94801'
x = np.arange(len(ratio_labels))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'legend.fontsize': 8})

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

# (A) Dataset composition
pos_bar = [results[r]['n_pos'] for r in ratio_labels]
ax0.bar(x, pos_bar, color=B, alpha=0.85, edgecolor='white', label='Positives (543)')
ax0.bar(x, n_negs, bottom=pos_bar, color=O, alpha=0.65, edgecolor='white', label='Negatives')
ax0.set_xticks(x); ax0.set_xticklabels(ratio_labels)
ax0.set_xlabel('Positive : Negative ratio'); ax0.set_ylabel('Number of samples')
ax0.set_title('(A) Training Dataset Composition', fontweight='bold')
ax0.legend(framealpha=0.9)
for i, (p, n) in enumerate(zip(pos_bar, n_negs)):
    ax0.text(i, p + n + 30, str(p + n), ha='center', va='bottom', fontsize=8, fontweight='bold')

# (B) CV AUROC
ax1.plot(x, cv_means, 'o-', color=B, lw=2.2, ms=8, zorder=5, label='CV AUROC (3×5-fold)')
ax1.fill_between(x, np.array(cv_means) - np.array(cv_stds),
                  np.array(cv_means) + np.array(cv_stds), alpha=0.18, color=B)
ax1.errorbar(x, cv_means, yerr=cv_stds, fmt='none', color=B, capsize=4, lw=1.5, zorder=6)
ax1.axvline(1, color=G, lw=1.5, ls='--', alpha=0.7, label='Selected 1:2')
ax1.axhline(0.5, color='#ccc', lw=1, ls=':', label='Random')
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax1.text(i, m + s + 0.003, f'{m:.3f}', ha='center', va='bottom', fontsize=8, color=B, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(ratio_labels)
ax1.set_xlabel('Ratio'); ax1.set_ylabel('AUROC'); ax1.set_ylim([0.78, 0.92])
ax1.set_title('(B) Cross-Validation AUROC (3×5-fold)', fontweight='bold')
ax1.legend(framealpha=0.9)

# (C) Test AUROC + AUPRC
ax2.plot(x, te_aurocs, 's-', color=B, lw=2.2, ms=8, label='Test AUROC')
ax2.plot(x, te_auprcs, '^--', color=O, lw=2.0, ms=8, label='Test AUPRC')
ax2.axvline(1, color=G, lw=1.5, ls='--', alpha=0.7, label='Selected 1:2')
ax2.axhline(0.5, color='#ccc', lw=1, ls=':')
for i, (a, p) in enumerate(zip(te_aurocs, te_auprcs)):
    ax2.text(i, a + 0.004, f'{a:.3f}', ha='center', va='bottom', fontsize=7.5, color=B, fontweight='bold')
    ax2.text(i, p - 0.022, f'{p:.3f}', ha='center', va='top',    fontsize=7.5, color=O, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(ratio_labels)
ax2.set_xlabel('Ratio'); ax2.set_ylabel('Score'); ax2.set_ylim([0.55, 0.93])
ax2.set_title('(C) Test AUROC and AUPRC', fontweight='bold')
ax2.legend(framealpha=0.9)

# (D) F1 / Precision / Recall
ax3.plot(x, te_f1s,   'o-',  color=B,  lw=2.0, ms=7, label='F1')
ax3.plot(x, te_precs, 's--', color=G,  lw=2.0, ms=7, label='Precision')
ax3.plot(x, te_recs,  '^:',  color=O,  lw=2.0, ms=7, label='Recall')
ax3.axvline(1, color='#7B5EA7', lw=1.5, ls='--', alpha=0.6, label='Selected 1:2')
ax3.set_xticks(x); ax3.set_xticklabels(ratio_labels)
ax3.set_xlabel('Ratio'); ax3.set_ylabel('Score'); ax3.set_ylim([0.40, 0.88])
ax3.set_title('(D) F1 / Precision / Recall', fontweight='bold')
ax3.legend(framealpha=0.9, fontsize=7.5)

# (E) CV fold distribution (violin)
parts = ax4.violinplot(cv_all, positions=x, widths=0.55,
                        showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor(B); pc.set_alpha(0.45)
parts['cmedians'].set_color(B); parts['cmedians'].set_linewidth(2)
ax4.scatter(x, cv_means, color=B, s=50, zorder=5, label='Mean')
ax4.axhline(0.5, color='#ccc', lw=1, ls=':')
ax4.axvline(1, color=G, lw=1.5, ls='--', alpha=0.7, label='Selected 1:2')
ax4.set_xticks(x); ax4.set_xticklabels(ratio_labels)
ax4.set_xlabel('Ratio'); ax4.set_ylabel('CV fold AUROC')
ax4.set_title('(E) CV Fold AUROC Distribution', fontweight='bold')
ax4.legend(framealpha=0.9, fontsize=7.5)

# (F) Heatmap summary
metrics = np.array([te_aurocs, te_auprcs, te_f1s, te_precs, te_recs, cv_means])
metric_labels = ['Test AUROC', 'Test AUPRC', 'Test F1',
                  'Test Precision', 'Test Recall', 'CV AUROC']
im = ax5.imshow(metrics, aspect='auto', cmap='YlOrRd', vmin=0.45, vmax=0.90)
ax5.set_xticks(x); ax5.set_xticklabels(ratio_labels, fontsize=9)
ax5.set_yticks(range(6)); ax5.set_yticklabels(metric_labels, fontsize=8.5)
ax5.set_xlabel('Ratio')
ax5.set_title('(F) Performance Heatmap', fontweight='bold')
plt.colorbar(im, ax=ax5, shrink=0.75, label='Score', pad=0.02)
for i in range(6):
    for j in range(5):
        ax5.text(j, i, f'{metrics[i,j]:.3f}', ha='center', va='center',
                 fontsize=8, fontweight='bold',
                 color='white' if metrics[i, j] > 0.72 else '#333')

plt.suptitle(
    'Figure 7.  Robustness Analysis: Effect of Positive-to-Negative Sampling Ratio\n'
    '(543 positives; ratios 1:1--1:5; 3\\u00d75-fold CV; '
    'XGBoost with scale_pos_weight compensation)',
    fontweight='bold', fontsize=10, y=1.01)

plt.savefig(f'{FIG_DIR}/Fig7_robustness.pdf', bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/Fig7_robustness.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved {FIG_DIR}/Fig7_robustness.{{pdf,png}}")
