import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── 1. LOAD ──────────────────────────────────────────────────
df = pd.read_csv(Path(__file__).parent.parent / "transformed_listings.csv")

numeric_features = [
    "rent", "area", "floor", "building_age",
    "building_size", "distance_to_nearest_station",
    "avg_distance_to_stations", "management_fee"
]

# ── 2. PREPARE DATA ──────────────────────────────────────────
def prepare(df):
    df_model = df[numeric_features + ["floor_plan"]].copy()
    df_model["management_fee"] = df_model["management_fee"].fillna(
        df_model["management_fee"].median()
    )
    top_plans = df_model["floor_plan"].value_counts().nlargest(8).index
    df_model["floor_plan"] = df_model["floor_plan"].where(
        df_model["floor_plan"].isin(top_plans), "Other"
    )
    df_encoded = pd.get_dummies(df_model, columns=["floor_plan"], drop_first=True)
    scaler = StandardScaler()
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
    return df_encoded.values

# ── 3. ISOLATION FOREST (wrapped for looping) ────────────────
def run_isolation_forest(X, contamination, random_state=42):
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state
    )
    labels = iso.fit_predict(X)   # 1 = normal, -1 = anomaly
    scores = iso.decision_function(X)
    return labels, scores

# ── 4. PREPARE & PCA (once — same for all contamination runs) ─
X = prepare(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

# ── 5. SWEEP CONTAMINATION & PLOT 2x2 ───────────────────────
contamination_values = [0.02, 0.05, 0.08, 0.10]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#0f1117")
axes = axes.flatten()

for ax, c in zip(axes, contamination_values):
    ax.set_facecolor("#0f1117")

    labels, scores = run_isolation_forest(X, contamination=c)
    normal  = labels == 1
    anomaly = labels == -1

    ax.scatter(X_pca[normal, 0],  X_pca[normal, 1],
        c="steelblue", alpha=0.35, s=14, edgecolors="none", zorder=2)
    ax.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1],
        c="crimson", alpha=0.85, s=22, edgecolors="white",
        linewidths=0.25, zorder=3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
            label=f"Normal  (n={normal.sum():,})", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
            label=f"Anomaly (n={anomaly.sum():,})", markersize=8),
    ]
    ax.legend(handles=legend_elements, fontsize=9,
        facecolor="#1e1e2e", edgecolor="#444", labelcolor="white")
    ax.set_title(f"contamination = {c}  ({anomaly.sum()} flagged)",
        color="white", fontsize=12, pad=8)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})", color="white", fontsize=9)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

fig.suptitle("IsolationForest — Contamination Sweep (Tokyo Housing)",
    color="white", fontsize=15, y=1.01)
fig.tight_layout()
plt.savefig(Path(__file__).parent.parent / "contamination_sweep.png", dpi=150,
    facecolor=fig.get_facecolor(), bbox_inches="tight")