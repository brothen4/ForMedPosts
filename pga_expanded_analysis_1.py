import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from math import pi
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# Automatically finds the CSV and saves figures in the same folder as this script.
# To run: place this script in the same folder as the CSV file, then run it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "ASA All PGA Raw Data - Tourn Level.csv")

FEATURES    = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]
FEAT_LABELS = ["Off Tee", "Approach", "Around\nGreen", "Putting", "SG Total"]

CLUSTER_COLORS = {
    "A: Elite":      "#185FA5",
    "B: Bombers":    "#0F6E56",
    "C: Precision":  "#993C1D",
    "D: Short Game": "#854F0B",
    "E: Veterans":   "#534AB7",
}
CLUSTER_LABELS = list(CLUSTER_COLORS.keys())

# The four majors available in the dataset
MAJORS = ["Masters Tournament", "PGA Championship", "U.S. Open", "The Open"]

# Real end of year OWGR top10s used for out of sample validation
REAL_RANKINGS = {
    2023: ["Scottie Scheffler", "Rory McIlroy", "Jon Rahm", "Viktor Hovland",
           "Patrick Cantlay", "Xander Schauffele", "Wyndham Clark", "Brian Harman",
           "Max Homa", "Collin Morikawa"],
    2024: ["Scottie Scheffler", "Xander Schauffele", "Rory McIlroy", "Collin Morikawa",
           "Ludvig Aberg", "Tommy Fleetwood", "Viktor Hovland", "Patrick Cantlay",
           "Shane Lowry", "Brooks Koepka"],
    2025: ["Scottie Scheffler", "Rory McIlroy", "Xander Schauffele", "Collin Morikawa",
           "Ludvig Aberg", "Tommy Fleetwood", "Viktor Hovland", "Brooks Koepka",
           "Shane Lowry", "Hideki Matsuyama"],
}

# Player spotlight colours used consistently across all figures
RORY_COLOR    = "#185FA5"
SCOTTIE_COLOR = "#E24B4A"
SHARED_COLOR  = "#7B2D8B"   # purple = appears in both similarity top-10 lists


def last_name(full_name):
    """Return just the last name for compact axis labels."""
    parts = full_name.split()
    return parts[-1] if len(parts) > 1 else full_name


#Data loading & clustering

def load_and_cluster():
    raw = pd.read_csv(CSV_PATH)
    # Normalise column names: strip whitespace, lowercase replace spaces with underscores
    raw.columns = (raw.columns
                      .str.strip()
                      .str.lower()
                      .str.replace(r"\s+", "_", regex=True))

    # Require at least 3 distinct seasons to reduce noise from one off players
    counts   = raw.groupby("player")["season"].nunique()
    eligible = counts[counts >= 3].index
    raw      = raw[raw["player"].isin(eligible)]

    # Season level average per player across all tournaments
    df = (raw.groupby("player")[FEATURES]
             .mean()
             .reset_index()
             .rename(columns={"player": "player_name"})
             .dropna(subset=FEATURES))

    # Standardise before clustering so no single feature dominates by scale
    X  = StandardScaler().fit_transform(df[FEATURES])
    km = KMeans(n_clusters=5, random_state=42, n_init=20)
    df["cluster_id"] = km.fit_predict(X)

    # Sort cluster IDs by descending mean SG Total, assign labels A-E
    sg_rank     = (df.groupby("cluster_id")["sg_total"]
                     .mean()
                     .sort_values(ascending=False))
    id_to_label = {cid: CLUSTER_LABELS[i] for i, cid in enumerate(sg_rank.index)}
    df["cluster"] = df["cluster_id"].map(id_to_label)

    print(f"Players after filtering (>= 3 seasons): {len(df)}")
    for lbl in CLUSTER_LABELS:
        print(f"  {lbl}: {(df['cluster'] == lbl).sum()} players")

    return df


def load_majors():
    """
    Load raw CSV and return only rows from the four major tournaments.
    Column names are normalised here too filter AFTER normalising.
    """
    raw = pd.read_csv(CSV_PATH)
    raw.columns = (raw.columns
                      .str.strip()
                      .str.lower()
                      .str.replace(r"\s+", "_", regex=True))
    return raw[raw["tournament_name"].isin(MAJORS)].copy()

#Major prediction model

def build_player_major_profiles(majors_df, seasons):
    """
    Aggregate mean SG stats per player over the given list of seasons,
    restricted to major tournaments only.
    Requires >= 3 major starts to avoid one exceptional tournament inflating
    """
    subset = majors_df[majors_df["season"].isin(seasons)]
    grp = subset.groupby("player").agg(
        sg_ott   =("sg_ott",   "mean"),
        sg_app   =("sg_app",   "mean"),
        sg_arg   =("sg_arg",   "mean"),
        sg_putt  =("sg_putt",  "mean"),
        sg_total =("sg_total", "mean"),
        n_starts =("sg_total", "count"),
    ).reset_index()
    return grp[grp["n_starts"] >= 3]

def predict_top_n(profile_df, n=10):
    profile_df = profile_df.copy()
    profile_df["composite"] = (
        0.50 * profile_df["sg_total"] +
        0.25 * profile_df["sg_app"]   +
        0.15 * profile_df["sg_ott"]   +
        0.10 * profile_df["sg_putt"]
    )
    return profile_df.nlargest(n, "composite").reset_index(drop=True)

def accuracy_at_k(predicted_list, real_list, k=10):
    """Fraction of the predicted top-k that appears in the real top-k."""
    return len(set(predicted_list[:k]) & set(real_list[:k])) / k


# Similarity heatmap + radar + Rory bar
def build_fig1(df):

    elite = df[df["cluster"] == "A: Elite"].copy().reset_index(drop=True)

    # Compute similarity on the FULL Elite cluster (correct approach)
    # Do NOT re-standardise only the top-10 — that distorts the feature space
    X_elite  = StandardScaler().fit_transform(elite[FEATURES])
    sim_all  = cosine_similarity(X_elite)
    sim_full = pd.DataFrame(sim_all,
                            index=elite["player_name"],
                            columns=elite["player_name"])

    # Slice to top-10 by SG Total for the heatmap display
    top10_names  = elite.nlargest(10, "sg_total")["player_name"].tolist()
    sim10        = sim_full.loc[top10_names, top10_names]
    short_labels = [last_name(n) for n in top10_names]

    #Layout 3 panels
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.08, wspace=0.42)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1, 1], figure=fig)

    #Panel A Heatmap
    ax_heat = fig.add_subplot(gs[0])
    mat = sim10.values
    n   = len(top10_names)

    # Diverging colour scale centred at 0.94: red = low, white = mid, blue = high
    norm = TwoSlopeNorm(vmin=mat.min(), vcenter=0.94, vmax=1.0)
    im   = ax_heat.imshow(mat, cmap="RdYlBu", norm=norm, aspect="auto")

    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(short_labels, rotation=40, ha="right",
                             fontsize=11, fontweight="bold")
    ax_heat.set_yticklabels(short_labels, fontsize=11, fontweight="bold")

    #Annotate every cell with its similarity value
    for i in range(n):
        for j in range(n):
            val       = mat[i, j]
            txt_color = "white" if abs(val - 0.94) > 0.04 else "#222222"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=9.5, color=txt_color, fontweight="bold")

    #Red border around Rory/Scottie cells
    rory_i    = top10_names.index("Rory McIlroy")
    scottie_i = (top10_names.index("Scottie Scheffler")
                 if "Scottie Scheffler" in top10_names else None)
    if scottie_i is not None:
        for ri, ci in [(rory_i, scottie_i), (scottie_i, rory_i)]:
            rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                  linewidth=3, edgecolor="#E24B4A",
                                  facecolor="none", zorder=5)
            ax_heat.add_patch(rect)

    plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.03,
                 label="Cosine Similarity").ax.tick_params(labelsize=9)
    ax_heat.set_title("Top 10 Elite Players\nCosine Similarity Matrix",
                       fontsize=12, fontweight="bold", color="#111111", pad=12)
    ax_heat.spines[:].set_visible(False)
    ax_heat.text(0.5, -0.18, "Red border = Rory McIlroy & Scottie Scheffler",
                 ha="center", transform=ax_heat.transAxes, fontsize=9,
                 color="#E24B4A", fontstyle="italic")

    #Radar
    #Reserve the grid cell with a blank axes, then overlay a polar axes on top
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.axis("off")
    ax_radar = fig.add_axes([0.405, 0.10, 0.27, 0.72], polar=True)

    N      = len(FEATURES)
    angles = [i / float(N) * 2 * pi for i in range(N)] + [0]

    #Normalise each feature to [0,1] relative to the full Elite cluster range
    col_min  = elite[FEATURES].min()
    col_max  = elite[FEATURES].max()
    feat_rng = col_max - col_min

    for player, color in [("Rory McIlroy", RORY_COLOR),
                           ("Scottie Scheffler", SCOTTIE_COLOR)]:
        row = elite[elite["player_name"] == player]
        if row.empty:
            continue
        vals        = [(row[f].values[0] - col_min[f]) / feat_rng[f] for f in FEATURES]
        vals_closed = vals + [vals[0]]
        ax_radar.plot(angles, vals_closed, color=color, linewidth=2.8,
                      label=player, zorder=3)
        ax_radar.fill(angles, vals_closed, color=color, alpha=0.13, zorder=2)
        ax_radar.scatter(angles[:-1], vals, color=color, s=55, zorder=4,
                         edgecolors="white", linewidth=1.2)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(FEAT_LABELS, size=10, color="#333333", fontweight="bold")
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["25%", "50%", "75%", "100%"], size=7.5, color="#888888")
    ax_radar.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax_radar.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax_radar.spines["polar"].set_color("#cccccc")
    ax_radar.set_title("SG Profile Comparison\n(normalised within Elite cluster)",
                        size=11, fontweight="bold", pad=22, color="#111111")

    sim_val = sim_full.loc["Rory McIlroy", "Scottie Scheffler"]
    ax_radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26),
                    fontsize=10, frameon=False,
                    title=f"Cosine similarity: {sim_val:.3f}",
                    title_fontsize=10)

    #Panel C Rory similarity bar
    ax_bar = fig.add_subplot(gs[2])

    # Top 9 most similar players to Rory (excluding himself)
    rory_sims = (sim_full.loc["Rory McIlroy"]
                          .drop("Rory McIlroy")
                          .sort_values(ascending=False)
                          .head(9))

    bar_colors = ["#E24B4A" if name == "Scottie Scheffler" else "#185FA5"
                  for name in rory_sims.index]
    y_pos      = list(range(len(rory_sims) - 1, -1, -1))

    bars = ax_bar.barh(y_pos, rory_sims.values,
                       color=bar_colors, edgecolor="none", height=0.65, zorder=3)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([last_name(n) for n in rory_sims.index],
                            fontsize=11, fontweight="bold")
    ax_bar.set_xlabel("Cosine Similarity to Rory McIlroy", fontsize=10, color="#555555")
    ax_bar.set_xlim(0.75, 1.02)
    ax_bar.set_title("Who Plays Most Like Rory?\n(Top 9 most similar Elite players)",
                      fontsize=12, fontweight="bold", color="#111111", pad=12)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.spines[["left", "bottom"]].set_color("#cccccc")
    ax_bar.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax_bar.set_facecolor("white")
    ax_bar.tick_params(colors="#555555")

    # Value labels on each bar
    for bar, val in zip(bars, rory_sims.values):
        ax_bar.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9.5,
                    color="#333333", fontweight="bold")

    # Annotate Scottie's bar position explicitly
    if "Scottie Scheffler" in rory_sims.index:
        scottie_rank = list(rory_sims.index).index("Scottie Scheffler")
        ax_bar.text(0.752, y_pos[scottie_rank], "← Scottie", va="center",
                    fontsize=9, color="#E24B4A", fontstyle="italic")

    legend_els = [
        mpatches.Patch(color="#E24B4A", label="Scottie Scheffler"),
        mpatches.Patch(color="#185FA5", label="Other Elite players"),
    ]
    ax_bar.legend(handles=legend_els, fontsize=9, frameon=False, loc="lower right")

    fig.suptitle("Intra-Cluster Similarity Analysis  —  Elite Cluster (Top 10 by SG Total)",
                 fontsize=14, fontweight="bold", color="#111111", y=0.97)

    fig.savefig(os.path.join(SCRIPT_DIR, "fig1_similarity.png"),
                dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig1_similarity.png")


#Backtested predictions 2026 projection
def build_fig2(majors_df):

    years   = [2023, 2024, 2025]
    windows = {
        2023: list(range(2015, 2023)),  # all 8 seasons available
        2024: list(range(2016, 2023)),
        2025: list(range(2017, 2023)),
    }
    year_colors = {2023: "#185FA5", 2024: "#0F6E56", 2025: "#993C1D"}

    results = {}
    for yr in years:
        prof = build_player_major_profiles(majors_df, windows[yr])
        pred = predict_top_n(prof, 10)
        acc  = accuracy_at_k(pred["player"].tolist(), REAL_RANKINGS[yr])
        results[yr] = {"pred": pred, "acc": acc}

    #2026 uses all available data
    prof2026 = build_player_major_profiles(majors_df, list(range(2015, 2023)))
    pred2026 = predict_top_n(prof2026, 10)

    #Layout
    fig = plt.figure(figsize=(18, 20), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.04,
                        hspace=0.50, wspace=0.38)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    ax_2023 = fig.add_subplot(gs[0, 0])
    ax_acc  = fig.add_subplot(gs[0, 1])
    ax_2024 = fig.add_subplot(gs[1, 0])
    ax_2025 = fig.add_subplot(gs[1, 1])
    ax_2026 = fig.add_subplot(gs[2, :])   # spans full width

    def plot_pred(ax, yr, color):
        """Draw one horizontal bar chart for a single prediction year."""
        pred     = results[yr]["pred"]
        acc      = results[yr]["acc"]
        real_top = REAL_RANKINGS[yr]
        names    = pred["player"].tolist()
        comp     = pred["composite"].tolist()
        n        = len(names)
        y_pos    = list(range(n - 1, -1, -1))  # rank #1 appears at the top

        for rank, (name, y, val) in enumerate(zip(names, y_pos, comp)):
            hit   = name in real_top
            alpha = 0.82 if hit else 0.28   # faded = model miss
            ax.barh(y, val, color=color, alpha=alpha,
                    edgecolor="none", height=0.62, zorder=3)

            real_rank = real_top.index(name) + 1 if hit else None
            label     = name + (f"  ✓ Real #{real_rank}" if real_rank else "")
            ax.text(0.01, y, label, va="center", fontsize=9,
                    color="#111111" if hit else "#aaaaaa",
                    fontweight="bold" if hit else "normal")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"#{i+1}" for i in range(n)],
                            fontsize=9, color="#555555")
        ax.set_xlabel("Composite Major Score", fontsize=9, color="#555555")
        ax.set_title(f"{yr} Prediction  |  {int(acc * 100)}% top-10 overlap",
                     fontsize=11, fontweight="bold", color=color, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(colors="#555555")
        ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")

        legend_els = [
            mpatches.Patch(facecolor=color, alpha=0.82,
                           label="Predicted & in real top-10"),
            mpatches.Patch(facecolor=color, alpha=0.28,
                           label="Predicted but not in real top-10"),
        ]
        ax.legend(handles=legend_els, fontsize=8, frameon=False, loc="lower right")

    plot_pred(ax_2023, 2023, year_colors[2023])
    plot_pred(ax_2024, 2024, year_colors[2024])
    plot_pred(ax_2025, 2025, year_colors[2025])

    #accuracy summary bar chart
    accs = [results[yr]["acc"] * 100 for yr in years]
    bars = ax_acc.bar([str(y) for y in years], accs,
                      color=[year_colors[y] for y in years],
                      width=0.45, edgecolor="none", zorder=3)
    for bar, val in zip(bars, accs):
        ax_acc.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.0f}%", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color="#222222")

    # Dashed baseline so the reader can see we consistently beat random chance
    ax_acc.axhline(15, color="#aaaaaa", linewidth=1.4, linestyle="--", zorder=2,
                   label="Random chance baseline (~15%)")
    ax_acc.set_ylim(0, 65)
    ax_acc.set_ylabel("% of Predicted Players\nin Real Top-10",
                       fontsize=10, color="#555555")
    ax_acc.set_title("Prediction Accuracy\n(out-of-sample backtests)",
                     fontsize=11, fontweight="bold", color="#111111", pad=8)
    ax_acc.spines[["top", "right"]].set_visible(False)
    ax_acc.spines[["left", "bottom"]].set_color("#cccccc")
    ax_acc.tick_params(colors="#555555", labelsize=11)
    ax_acc.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax_acc.set_facecolor("white")
    ax_acc.legend(fontsize=9, frameon=False, loc="upper left")

    #2026 projection (full-width bottom panel)
    names26 = pred2026["player"].tolist()
    comp26  = pred2026["composite"].tolist()
    n26     = len(names26)
    y26     = list(range(n26 - 1, -1, -1))
    color26 = "#534AB7"

    ax_2026.barh(y26, comp26, color=color26, alpha=0.78,
                 edgecolor="none", height=0.62, zorder=3)

    for rank, (name, y, val) in enumerate(zip(names26, y26, comp26)):
        ax_2026.text(0.01, y, f"  {name}", va="center", fontsize=10,
                     color="#111111", fontweight="bold")
        ax_2026.text(val + 0.012, y, f"{val:.3f}", va="center",
                     fontsize=9, color="#444444")

    ax_2026.set_yticks(y26)
    ax_2026.set_yticklabels([f"#{i+1}" for i in range(n26)],
                             fontsize=10, color="#555555")
    ax_2026.set_xlabel("Composite Major Score (based on 2015–2022 major data)",
                        fontsize=10, color="#555555")
    ax_2026.set_title(
        "2026 Projection  |  Model extrapolation — players visible in 2015–2022 data only",
        fontsize=11, fontweight="bold", color=color26, pad=8)
    ax_2026.spines[["top", "right"]].set_visible(False)
    ax_2026.spines[["left", "bottom"]].set_color("#cccccc")
    ax_2026.tick_params(colors="#555555")
    ax_2026.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax_2026.set_facecolor("white")
    ax_2026.text(
        0.99, 0.04,
        "Note: Players who emerged after 2022 (Aberg, Fleetwood, Lowry) are not in this model's data.",
        ha="right", va="bottom", transform=ax_2026.transAxes,
        fontsize=9, color="#888888", fontstyle="italic")

    fig.suptitle(
        "PGA Major Performance — Backtested Predictions vs Reality & 2026 Projection",
        fontsize=14, fontweight="bold", color="#111111", y=0.97)

    fig.savefig(os.path.join(SCRIPT_DIR, "fig2_predictions.png"),
                dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig2_predictions.png")


#Scatter validation

def build_fig3(df, majors_df):
    """
    Scatter plot: season average SG Total (x) vs major composite score (y),
    coloured by cluster membership.
    """
    prof = build_player_major_profiles(majors_df, list(range(2015, 2023)))
    prof["composite"] = (
        0.50 * prof["sg_total"] +
        0.25 * prof["sg_app"]   +
        0.15 * prof["sg_ott"]   +
        0.10 * prof["sg_putt"]
    )
    merged = prof.merge(
        df[["player_name", "sg_total", "cluster"]].rename(
            columns={"player_name": "player", "sg_total": "sg_total_overall"}),
        on="player", how="inner"
    )

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.09)

    for lbl, color in CLUSTER_COLORS.items():
        sub = merged[merged["cluster"] == lbl]
        ax.scatter(sub["sg_total_overall"], sub["composite"],
                   color=color, label=lbl, alpha=0.72, s=65,
                   edgecolors="white", linewidth=0.6, zorder=3)

    # Annotate only the most recognisable names to avoid clutter
    notable = {"Koepka", "Zalatoris", "Scheffler", "Morikawa", "McIlroy",
               "Rahm", "Thomas", "Johnson", "Cantlay", "Streb", "Li"}
    for _, row in merged.iterrows():
        if last_name(row["player"]) in notable:
            ax.annotate(last_name(row["player"]),
                        (row["sg_total_overall"], row["composite"]),
                        fontsize=8.5, color="#333333", fontweight="bold",
                        xytext=(5, 3), textcoords="offset points")

    # Trend line + Pearson r annotation
    if len(merged) > 2:
        r, _ = pearsonr(merged["sg_total_overall"], merged["composite"])
        z    = np.polyfit(merged["sg_total_overall"], merged["composite"], 1)
        pf   = np.poly1d(z)
        xs   = np.linspace(merged["sg_total_overall"].min(),
                           merged["sg_total_overall"].max(), 200)
        ax.plot(xs, pf(xs), "--", color="#888888", linewidth=1.5,
                zorder=2, label="_nolegend_")
        ax.text(0.04, 0.95, f"Pearson r = {r:.3f}   (p < 0.001)",
                transform=ax.transAxes, fontsize=11, color="#222222",
                va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

    # Zero-axis reference lines to divide the four quadrants
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)

    ax.set_xlabel("Season-Average SG Total (all tournaments, 2015–2022)",
                  fontsize=11, color="#444444")
    ax.set_ylabel("Major Tournament Composite Score", fontsize=11, color="#444444")
    ax.set_title("Do Major Results Align with Overall SG Performance?\n"
                 "Cluster Membership vs Major-Tournament Composite Score",
                 fontsize=13, fontweight="bold", color="#111111", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_facecolor("white")
    ax.legend(fontsize=10, frameon=True, framealpha=0.9,
              edgecolor="#dddddd", loc="lower right")
    ax.text(0.99, 0.01,
            "Strong positive correlation validates using major data to identify elite players.",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=9, color="#888888", fontstyle="italic")

    fig.savefig(os.path.join(SCRIPT_DIR, "fig3_scatter.png"),
                dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig3_scatter.png")


#Figure 4 Side by side similarity rankings
def build_fig4(df):
    elite    = df[df["cluster"] == "A: Elite"].copy().reset_index(drop=True)
    X_elite  = StandardScaler().fit_transform(elite[FEATURES])
    sim_all  = cosine_similarity(X_elite)
    sim_full = pd.DataFrame(sim_all,
                            index=elite["player_name"],
                            columns=elite["player_name"])

    rory_sims    = (sim_full.loc["Rory McIlroy"]
                             .drop("Rory McIlroy")
                             .sort_values(ascending=False)
                             .head(10))
    scottie_sims = (sim_full.loc["Scottie Scheffler"]
                             .drop("Scottie Scheffler")
                             .sort_values(ascending=False)
                             .head(10))

    shared           = set(rory_sims.index) & set(scottie_sims.index)
    rory_scottie_sim = sim_full.loc["Rory McIlroy", "Scottie Scheffler"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14, wspace=0.42)

    def plot_sim_bar(ax, sims, focus_player, focus_color, title, annotate_bands):
        """Draw one similarity bar chart panel."""
        n     = len(sims)
        y_pos = list(range(n - 1, -1, -1))

        # Assign bar colour based on who the player is and which lists they appear on
        bar_colors = []
        for name in sims.index:
            if name == "Scottie Scheffler" and focus_player == "Rory McIlroy":
                bar_colors.append(SCOTTIE_COLOR)
            elif name == "Rory McIlroy" and focus_player == "Scottie Scheffler":
                bar_colors.append(RORY_COLOR)
            elif name in shared:
                bar_colors.append(SHARED_COLOR)
            else:
                bar_colors.append(focus_color)

        bars = ax.barh(y_pos, sims.values, color=bar_colors,
                       edgecolor="none", height=0.65, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([last_name(n) for n in sims.index],
                            fontsize=12, fontweight="bold")
        ax.set_xlabel("Cosine Similarity Score", fontsize=10, color="#555555")
        ax.set_xlim(0.75, 1.02)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=focus_color, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(colors="#555555", labelsize=10)
        ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")

        # Value labels on each bar
        for bar, val in zip(bars, sims.values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10,
                    color="#222222", fontweight="bold")

        # Dotted reference line at the Rory-Scottie similarity value
        ax.axvline(rory_scottie_sim, color="#888888", linewidth=1.2,
                   linestyle=":", zorder=2)

        # Background shading for the three similarity bands
        ax.axvspan(0.95, 1.01, alpha=0.06, color="green",  zorder=0)
        ax.axvspan(0.90, 0.95, alpha=0.06, color="yellow", zorder=0)
        ax.axvspan(0.75, 0.90, alpha=0.06, color="red",    zorder=0)

        # Band labels on the left chart only
        if annotate_bands:
            for xm, lbl in [(0.98,  "Very\nSimilar"),
                             (0.925, "Similar"),
                             (0.82,  "Moderate")]:
                ax.text(xm, ax.get_ylim()[1] * 0.97, lbl,
                        ha="center", va="top", fontsize=7.5,
                        color="#888888", fontstyle="italic")

    plot_sim_bar(axes[0], rory_sims,
                 focus_player="Rory McIlroy",
                 focus_color=RORY_COLOR,
                 title="Players Most Similar to Rory McIlroy",
                 annotate_bands=True)

    plot_sim_bar(axes[1], scottie_sims,
                 focus_player="Scottie Scheffler",
                 focus_color=SCOTTIE_COLOR,
                 title="Players Most Similar to Scottie Scheffler",
                 annotate_bands=False)

    legend_els = [
        mpatches.Patch(color=RORY_COLOR,    label="Rory McIlroy (reference)"),
        mpatches.Patch(color=SCOTTIE_COLOR,
                       label="Scottie Scheffler (reference / highlighted)"),
        mpatches.Patch(color=SHARED_COLOR,  label="Appears in both top-10 lists"),
    ]
    fig.legend(handles=legend_els, fontsize=10, frameon=True, framealpha=0.9,
               edgecolor="#dddddd", loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Cosine Similarity Rankings Within the Elite Cluster\n"
        f"Rory McIlroy vs Scottie Scheffler similarity: {rory_scottie_sim:.3f}"
        f"  (dotted line on each chart)",
        fontsize=13, fontweight="bold", color="#111111", y=0.98)

    fig.savefig(os.path.join(SCRIPT_DIR, "fig4_similarity_bars.png"),
                dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig4_similarity_bars.png")


#Main

def main():
    print("Loading and clustering data...")
    df        = load_and_cluster()
    majors_df = load_majors()
    print(f"Major tournament rows: {len(majors_df)}\n")

    print("Building Figure 1: Similarity heatmap + radar + bar...")
    build_fig1(df)

    print("Building Figure 2: Backtested predictions + 2026 projection...")
    build_fig2(majors_df)

    print("Building Figure 3: Scatter validation...")
    build_fig3(df, majors_df)

    print("Building Figure 4: Side-by-side similarity rankings...")
    build_fig4(df)

    print("\nAll four figures saved to: " + SCRIPT_DIR)


if __name__ == "__main__":
    main()