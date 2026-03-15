"""
============================================================
  K-MEANS MASTERY — STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
  Dataset: CC GENERAL.csv  |  8,950 credit card customers
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Nice plot style ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3f5c',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#b0b0b0',
    'ytick.color':      '#b0b0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3e',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
PALETTE = ['#7C83FD', '#96FAFA', '#FF6B6B', '#FFD93D',
           '#6BCB77', '#FF922B', '#CC5DE8', '#74C0FC']

# ════════════════════════════════════════════════════════════
# 1. LOAD & BASIC INSPECTION
# ════════════════════════════════════════════════════════════
df = pd.read_csv('CC GENERAL.csv')

print("=" * 60)
print("  DATASET OVERVIEW")
print("=" * 60)
print(f"  Shape          : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"  Memory Usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"\n  Column Data Types:")
for col, dtype in df.dtypes.items():
    print(f"    {col:<40} {dtype}")

print("\n" + "=" * 60)
print("  STATISTICAL SUMMARY (numeric features)")
print("=" * 60)
desc = df.describe().T
desc['range'] = desc['max'] - desc['min']
print(desc[['count','mean','std','min','50%','max','range']].to_string())

# ════════════════════════════════════════════════════════════
# 2. MISSING VALUES
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mv = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
mv = mv[mv['Missing Count'] > 0]
print(mv.to_string())

# ════════════════════════════════════════════════════════════
# 3. PLOT A — Feature Distributions (Histograms)
# ════════════════════════════════════════════════════════════
numeric_cols = df.select_dtypes(include='number').columns.tolist()
n_cols = 3
n_rows = -(-len(numeric_cols) // n_cols)   # ceiling division

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(18, n_rows * 3.5),
                         facecolor='#0f1117')
fig.suptitle('STEP 2 — Feature Distributions\nCC General Credit Card Dataset',
             fontsize=16, fontweight='bold', color='#96FAFA', y=1.01)

axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    color = PALETTE[i % len(PALETTE)]
    ax.hist(df[col].dropna(), bins=50, color=color, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_title(col, fontsize=9, fontweight='bold', color=color)
    ax.set_xlabel('Value', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.grid(True, alpha=0.3)
    # overlay median line
    median_val = df[col].median()
    ax.axvline(median_val, color='white', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.text(0.97, 0.95, f'median={median_val:.1f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, color='white', alpha=0.8)

# hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\n  ✅ Saved: eda_distributions.png")

# ════════════════════════════════════════════════════════════
# 4. PLOT B — Correlation Heatmap
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0f1117')
corr = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
cmap = sns.diverging_palette(250, 15, s=75, l=40, as_cmap=True)

sns.heatmap(corr,
            mask=mask,
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            annot=True, fmt='.2f',
            annot_kws={'size': 7},
            linewidths=0.4,
            linecolor='#0f1117',
            square=True,
            ax=ax,
            cbar_kws={'shrink': 0.8})

ax.set_title('STEP 2 — Correlation Heatmap\n(lower triangle only)',
             fontsize=14, fontweight='bold', color='#96FAFA', pad=15)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✅ Saved: eda_correlation.png")

# ════════════════════════════════════════════════════════════
# 5. PLOT C — Boxplots (Outlier Detection)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(18, n_rows * 3.5),
                         facecolor='#0f1117')
fig.suptitle('STEP 2 — Boxplots (Outlier Detection)',
             fontsize=16, fontweight='bold', color='#FFD93D', y=1.01)

axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    color = PALETTE[i % len(PALETTE)]
    data = df[col].dropna()
    bp = ax.boxplot(data,
                    patch_artist=True,
                    vert=True,
                    widths=0.4,
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='#aaaaaa', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='#aaaaaa', linewidth=1)
    for flier in bp['fliers']:
        flier.set(marker='o', color=color, alpha=0.3, markersize=2)

    ax.set_title(col, fontsize=9, fontweight='bold', color=color)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3)

    # count outliers using IQR rule
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
    ax.set_xlabel(f'Outliers: {n_out:,}', fontsize=7, color='#FF6B6B')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  ✅ Saved: eda_boxplots.png")

# ════════════════════════════════════════════════════════════
# 6. KEY INSIGHTS SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  KEY EDA INSIGHTS")
print("=" * 60)

high_corr = []
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) >= 0.5:
            high_corr.append((corr.columns[i], corr.columns[j], round(corr.iloc[i,j],2)))

print(f"\n  📌 High correlations (|r| ≥ 0.5):")
for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
    direction = "↑↑" if r > 0 else "↑↓"
    print(f"     {direction}  {a}  ↔  {b}  (r = {r})")

print(f"\n  📌 Right-skewed features (need scaling):")
for col in numeric_cols:
    skew = df[col].skew()
    if abs(skew) > 1:
        print(f"     {col:<40}  skewness = {skew:.2f}")

print(f"\n  📌 Features with outliers (IQR rule):")
for col in numeric_cols:
    data = df[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
    pct = n_out / len(data) * 100
    if pct > 1:
        print(f"     {col:<40}  {n_out:>4} outliers  ({pct:.1f}%)")

print("\n" + "=" * 60)
print("  EDA COMPLETE ✅")
print("  Plots saved: eda_distributions.png | eda_correlation.png | eda_boxplots.png")
print("=" * 60)
