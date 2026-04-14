import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import pi
np.random.seed(42)

CSV_PATH = "/Users/benrothenberg/Documents/INST 414/ASA All PGA Raw Data - Tourn Level.csv"
CLUSTER_COLORS = {
    "A: Elite":       "#185FA5",
    "B: Bombers":     "#0F6E56",
    "C: Precision":   "#993C1D",
    "D: Short Game":  "#854F0B",
    "E: Veterans":    "#534AB7",
}
CLUSTER_LABELS      = list(CLUSTER_COLORS.keys())
CLUSTER_COLORS_LIST = list(CLUSTER_COLORS.values())

# CHANGED: swapped to strokes gained features available in your CSV
FEATURES = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]
FEATURE_LABELS = ["Off Tee", "Approach", "Around Green", "Putting", "SG Total"]

CHOSEN_K = 5
ELBOW_K  = list(range(1, 9))

def load_data(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = (
        raw.columns.str.strip()
                   .str.lower()
                   .str.replace(r"\s+", "_", regex=True)
    )
    rename = {
        "avg._driving_distance":  "avg_driving_distance",
        "avg_driving_distance":   "avg_driving_distance",
        "driving_accuracy_%":     "driving_accuracy_pct",
        "driving_accuracy_pct":   "driving_accuracy_pct",
        "greens_in_regulation_%": "gir_pct",
        "gir_%":                  "gir_pct",
        "scrambling_%":           "scrambling_pct",
        "scrambling_pct":         "scrambling_pct",
        "avg._putts_per_round":   "avg_putts_per_round",
        "avg_putts_per_round":    "avg_putts_per_round",
        "sg:_total":              "sg_total",
        "sg_total":               "sg_total",
        "player_name":            "player_name",
        "player":                 "player_name",  # CHANGED: maps your CSV's 'player' column
    }
    raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    available = [f for f in FEATURES if f in raw.columns]
    missing   = [f for f in FEATURES if f not in raw.columns]
    if missing:
        print(f"features not found {missing}")

    if "player_name" in raw.columns and "season" in raw.columns:
        counts   = raw.groupby("player_name")["season"].nunique()
        eligible = counts[counts >= 3].index
        raw      = raw[raw["player_name"].isin(eligible)]
        df       = raw.groupby("player_name")[available].mean().reset_index()
    else:
        df = raw[available].dropna().reset_index(drop=True)

    df = df.dropna(subset=available)
    print(f"Players after filtering (>= 3 seasons): {len(df)}")
    return df

def compute_elbow(X: np.ndarray) -> list:
    inertias = []
    for k in ELBOW_K:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias


def run_kmeans(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    km = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=20)
    df = df.copy()
    df["cluster_id"] = km.fit_predict(X)

    sg_col  = "sg_total" if "sg_total" in df.columns else df.columns[-2]
    sg_rank = (
        df.groupby("cluster_id")[sg_col]
          .mean()
          .sort_values(ascending=False)
    )
    id_to_label = {cid: CLUSTER_LABELS[i] for i, cid in enumerate(sg_rank.index)}
    df["cluster"] = df["cluster_id"].map(id_to_label)
    return df


def build_cluster_stats(df: pd.DataFrame):
    """Return (sizes, sg_means, normalised profiles [0-1]) from real data."""
    available = [f for f in FEATURES if f in df.columns]
    sizes, sg_means, profiles = [], [], []

    for label in CLUSTER_LABELS:
        grp = df[df["cluster"] == label]
        sizes.append(len(grp))
        sg_means.append(grp["sg_total"].mean() if "sg_total" in grp.columns else 0.0)

        row = []
        for feat in available:
            col_min = df[feat].min()
            col_max = df[feat].max()
            rng     = col_max - col_min if col_max != col_min else 1
            val     = (grp[feat].mean() - col_min) / rng
            if feat == "avg_putts_per_round":
                val = 1 - val
            row.append(round(val, 3))
        profiles.append(row)

    n_feat   = len(FEATURES)
    profiles = [p + [0.5] * (n_feat - len(p)) for p in profiles]
    return sizes, sg_means, np.array(profiles)


def top_players_per_cluster(df: pd.DataFrame, n: int = 6) -> dict:
    """Top-n players by SG: Total for each cluster label."""
    result = {}
    for label in CLUSTER_LABELS:
        grp = df[df["cluster"] == label]
        if "player_name" in grp.columns and "sg_total" in grp.columns:
            result[label] = grp.nlargest(n, "sg_total")["player_name"].tolist()
        elif "player_name" in grp.columns:
            result[label] = grp["player_name"].head(n).tolist()
        else:
            result[label] = [f"Player {i+1}" for i in range(n)]
    return result

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=9)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#222222")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#555555")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#555555")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)


def plot_elbow(ax, inertias: list):
    ax.plot(ELBOW_K, inertias, marker="o", color="#378ADD",
            linewidth=2, markersize=6, markerfacecolor="white",
            markeredgecolor="#378ADD", markeredgewidth=2, zorder=3)
    ax.fill_between(ELBOW_K, inertias, alpha=0.08, color="#378ADD")
    ax.axvline(CHOSEN_K, color="#E24B4A", linewidth=1.5,
               linestyle="--", label=f"Chosen k = {CHOSEN_K}", zorder=2)
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax, title="Elbow method -- choosing k",
             xlabel="Number of clusters (k)", ylabel="Inertia")
    ax.set_xticks(ELBOW_K)

def plot_cluster_sizes(ax, sizes: list):
    bars = ax.bar(CLUSTER_LABELS, sizes, color=CLUSTER_COLORS_LIST,
                  width=0.55, edgecolor="none", zorder=3)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9, color="#444444")
    style_ax(ax, title="Cluster size distribution", ylabel="Number of players")
    ax.set_ylim(0, max(sizes) * 1.18)
    ax.set_xticks(range(len(CLUSTER_LABELS)))
    ax.set_xticklabels(CLUSTER_LABELS, rotation=20, ha="right")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.grid(False)

def _draw_radar(fig, pos, idx: int, profiles: np.ndarray):
    N      = len(FEATURE_LABELS)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    values = list(profiles[idx]) + [profiles[idx][0]]
    color  = CLUSTER_COLORS_LIST[idx]

    ax = fig.add_subplot(pos, polar=True)
    ax.plot(angles, values, color=color, linewidth=1.5)
    ax.fill(angles, values, color=color, alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURE_LABELS, size=7, color="#555555")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], size=0)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.spines["polar"].set_color("#cccccc")
    ax.set_title(CLUSTER_LABELS[idx], size=9, fontweight="bold",
                 color=color, pad=10)

def plot_radar_grid(fig, gs_row, profiles: np.ndarray):
    for i in range(CHOSEN_K):
        _draw_radar(fig, gs_row[i], i, profiles)

def plot_sg_bar(ax, sg_means: list):
    bars = ax.bar(CLUSTER_LABELS, sg_means, color=CLUSTER_COLORS_LIST,
                  width=0.55, edgecolor="none", zorder=3)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, zorder=2)
    for bar, val in zip(bars, sg_means):
        offset = 0.04 if val >= 0 else -0.10
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, color="#444444")
    style_ax(ax, title="Mean SG: Total by cluster",
             ylabel="SG: Total (strokes vs. field avg)")
    ax.set_xticks(range(len(CLUSTER_LABELS)))
    ax.set_xticklabels(CLUSTER_LABELS, rotation=20, ha="right")
    lo, hi = min(sg_means), max(sg_means)
    ax.set_ylim(lo - 0.35, hi + 0.35)


def plot_membership_table(ax, sizes, sg_means, example_players):
    ax.axis("off")
    col_w = 1.0 / len(CLUSTER_LABELS)

    for ci, label in enumerate(CLUSTER_LABELS):
        x0    = ci * col_w + 0.01
        color = CLUSTER_COLORS_LIST[ci]

        ax.add_patch(FancyBboxPatch(
            (x0, 0.87), col_w - 0.02, 0.10,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor="none",
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(x0 + (col_w - 0.02) / 2, 0.921, label,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", transform=ax.transAxes)

        players = example_players.get(label, [])
        for pi, player in enumerate(players):
            y = 0.80 - pi * 0.115
            ax.text(x0 + 0.01, y, player, fontsize=7.5, va="center",
                    color="#222222", transform=ax.transAxes)
            if pi < len(players) - 1:
                ax.plot([x0 + 0.005, x0 + col_w - 0.015], [y - 0.06, y - 0.06],
                        color="#eeeeee", linewidth=0.5,
                        transform=ax.transAxes, clip_on=False)

        ax.add_patch(FancyBboxPatch(
            (x0, 0.01), col_w - 0.02, 0.11,
            boxstyle="round,pad=0.01",
            facecolor=color + "22", edgecolor=color + "55", linewidth=0.5,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(x0 + (col_w - 0.02) / 2, 0.085,
                f"SG mean: {sg_means[ci]:+.2f}",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(x0 + (col_w - 0.02) / 2, 0.030, f"{sizes[ci]} players",
                ha="center", va="center", fontsize=7,
                color="#666666", transform=ax.transAxes)

    ax.set_title("Cluster membership -- top players by SG: Total",
                 fontsize=11, fontweight="bold", pad=14, color="#222222")


def main():
    df_raw = load_data(CSV_PATH)

    available = [f for f in FEATURES if f in df_raw.columns]
    scaler    = StandardScaler()
    X         = scaler.fit_transform(df_raw[available].fillna(df_raw[available].mean()))
    df        = run_kmeans(df_raw, X)

    inertias                  = compute_elbow(X)
    sizes, sg_means, profiles = build_cluster_stats(df)
    example_players           = top_players_per_cluster(df, n=6)

    fig = plt.figure(figsize=(18, 22), facecolor="white")
    fig.subplots_adjust(hspace=0.55, wspace=0.35,
                        top=0.96, bottom=0.03, left=0.06, right=0.97)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[1, 1.1, 1, 1.3],
                           hspace=0.55, wspace=0.35)

    ax_elbow = fig.add_subplot(gs[0, 0])
    ax_sizes = fig.add_subplot(gs[0, 1])
    plot_elbow(ax_elbow, inertias)
    plot_cluster_sizes(ax_sizes, sizes)

    gs_radar = gridspec.GridSpecFromSubplotSpec(
        1, CHOSEN_K, subplot_spec=gs[1, :], wspace=0.45
    )
    plot_radar_grid(fig, gs_radar, profiles)

    legend_y = 0.535
    for i, (label, color) in enumerate(zip(CLUSTER_LABELS, CLUSTER_COLORS_LIST)):
        x = 0.07 + i * 0.185
        fig.text(x, legend_y, "●", color=color, fontsize=10, ha="center")
        fig.text(x + 0.012, legend_y, label, fontsize=7.5,
                 color="#444444", va="center")

    ax_sg    = fig.add_subplot(gs[2, :])
    ax_table = fig.add_subplot(gs[3, :])
    plot_sg_bar(ax_sg, sg_means)
    plot_membership_table(ax_table, sizes, sg_means, example_players)

    fig.text(0.5, 0.98, "PGA Tour Player Clustering Analysis  -  2015-2022",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#111111")
    fig.text(0.5, 0.965,
             f"k-means (k={CHOSEN_K}) on {len(available)} strokes gained features  "
             "-  similarity metric: cosine distance",
             ha="center", va="top", fontsize=9, color="#777777")

    out = "pga_cluster_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()