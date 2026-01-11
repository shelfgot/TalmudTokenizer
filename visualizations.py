"""visualization framework for Talmud tokenizer experiments
uses the results.json file from the larger framework

1. Constellation map
2. Normalized parallel-coordinates plot
3. Radar (spider) charts per algorithm
4. Lorenz curves of token frequency distributions
5. Pareto frontier plots (didn't work so well but whatever)
6. Metrics heatmap w/ glyph encodings
7. Token coverage percentage plots
8. Comparative segmentation strips (for a few sample sentences; it shows how each tokenizer segments them. not fully implemented.)
"""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import ExperimentConfig
from main import TalmudTokenizationExperiment
from RESEARCH_FINAL.talmud_tokenizers.tokenizer_base import BaseTokenizer, TokenizerWrapper
from talmud_tokenizers.canonical import BPETokenizer, WordPieceTokenizer
from talmud_tokenizers.advanced import (
    UnigramTokenizer, TokenMonsterTokenizer,
    SRETokenizer, TokenMonsterSREHybridTokenizer
)


ALGO_COLORS = {
    "bpe": "#1f77b4",
    "wordpiece": "#2ca02c",
    "sre": "#d62728",
    "tokenmonster": "#9467bd",
    "tokenmonster_sre": "#928a57",
    "unigram": "#8c564b",
}


@dataclass
class FlatResult:
    name: str
    algorithm: str
    renyi_entropy: float
    nsl: float
    fertility: float
    zipfian_alignment: float
    vocab_usage_percentage: float
    gini_coefficient: float
    total_tokens: int
    unique_tokens_used: int
    train_time: float


def load_results(results_path: Path) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """converts results.json into a DataFrame
    """
    with open(results_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows: List[FlatResult] = []
    for name, res in raw.items():
        if res.get("status") != "success":
            continue
        cfg = res["config"]
        intr = res["intrinsic"]
        dist = intr["distribution_stats"]
        rows.append(
            FlatResult(
                name=name,
                algorithm=cfg.get("algorithm", "unknown"),
                renyi_entropy=float(intr.get("renyi_entropy", 0.0)),
                nsl=float(intr.get("nsl", 0.0)),
                fertility=float(intr.get("fertility", 0.0)),
                zipfian_alignment=float(intr.get("zipfian_alignment", 0.0)),
                vocab_usage_percentage=float(dist.get("vocab_usage_percentage", 0.0)),
                gini_coefficient=float(dist.get("gini_coefficient", 0.0)),
                total_tokens=int(dist.get("total_tokens", 0)),
                unique_tokens_used=int(dist.get("unique_tokens_used", 0)),
                train_time=float(res.get("train_time", 0.0)),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df, raw


# 1. Constellation map


def plot_constellation(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "renyi_entropy",
        "nsl",
        "fertility",
        "zipfian_alignment",
        "vocab_usage_percentage",
        "gini_coefficient",
    ]
    X = df[metrics].values
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(df) - 1))
    embedding = tsne.fit_transform(X_scaled)

    df_plot = df.copy()
    df_plot["x"] = embedding[:, 0]
    df_plot["y"] = embedding[:, 1]

    plt.figure(figsize=(10, 8))
    for algo, group in df_plot.groupby("algorithm"):
        color = ALGO_COLORS.get(algo, None)
        for dropout_flag, sub in group.groupby("use_bpe_dropout"):
            label = f"{algo} ({'dropout' if dropout_flag else 'no-dropout'})"
            plt.scatter(
                sub["x"],
                sub["y"],
                c=color,
                marker=None,
                s=80,
                alpha=0.8,
                edgecolor="k",
                linewidths=0.5,
                label=label,
            )

    plt.title("Tokenizer Constellation Map (t-SNE embedding)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "constellation_map.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved constellation map to {out_path}")


# 2. Parallel-coordinates plot


def plot_parallel_coordinates(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Normalized parallel-coordinates plot; inversions mean higher is always better; top percentile graphics
    """
    metrics = [
        "renyi_entropy",
        "nsl",            
        "fertility",      
        "zipfian_alignment", 
        "vocab_usage_percentage",  
        "gini_coefficient",  
    ]

    df_norm = df.copy()
    #invert metrics measuring loss such that higher is better
    invert = {"nsl", "fertility", "gini_coefficient"}
    for m in metrics:
        vals = df_norm[m].values.astype(float)
        if m in invert:
            vals = -vals
        scaler = MinMaxScaler()
        vals_scaled = scaler.fit_transform(vals.reshape(-1, 1)).ravel()
        df_norm[m] = vals_scaled

    band_threshold = 0.75

    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))

    plt.fill_between(
        x,
        band_threshold,
        1.0,
        color="lightgray",
        alpha=0.3,
        label="Top band (≥ 75%)",
    )

    for _, row in df_norm.iterrows():
        y = [row[m] for m in metrics]
        algo = row["algorithm"]
        color = ALGO_COLORS.get(algo, "#333333")
        plt.plot(x, y, color=color, alpha=0.5, linewidth=1)

    plt.xticks(x, [m.replace("_", " ").title() for m in metrics], rotation=30)
    plt.ylabel("Normalized score (higher is better)")
    plt.title("Normalized Parallel-Coordinates Plot of Metrics")

    handles = []
    labels = []
    for algo, color in ALGO_COLORS.items():
        handles.append(plt.Line2D([0], [0], color=color, lw=2))
        labels.append(algo)
    handles.append(
        plt.Rectangle((0, 0), 1, 1, color="lightgray", alpha=0.3)
    )
    labels.append("Top band (≥ 75%)")

    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 1.05)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parallel_coordinates.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved parallel-coordinates plot to {out_path}")


# 3. Radar charts for each algorithm


def _normalize_metrics_for_radar(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    df_norm = df.copy()
    invert = {"nsl", "fertility", "gini_coefficient"}
    for m in metrics:
        vals = df_norm[m].values.astype(float)
        if m in invert:
            vals = -vals
        scaler = MinMaxScaler()
        vals_scaled = scaler.fit_transform(vals.reshape(-1, 1)).ravel()
        df_norm[m] = vals_scaled
    return df_norm


def plot_radar_per_algorithm(df: pd.DataFrame, out_dir: Path) -> None:
    """plot the mean normalized metric values across each algo
    configurations
    """
    metrics = [
        "renyi_entropy",
        "nsl",
        "fertility",
        "zipfian_alignment",
        "vocab_usage_percentage",
        "gini_coefficient",
    ]
    df_norm = _normalize_metrics_for_radar(df, metrics)

    # Radar setup
    labels = [m.replace("_", " ").title() for m in metrics]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close

    out_dir.mkdir(parents=True, exist_ok=True)

    for algo, group in df_norm.groupby("algorithm"):
        if group.empty:
            continue
        means = [group[m].mean() for m in metrics]
        values = means + means[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color=ALGO_COLORS.get(algo, "#1f77b4"), linewidth=2)
        ax.fill(angles, values, color=ALGO_COLORS.get(algo, "#1f77b4"), alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_title(f"Radar Chart - {algo}")

        fig.tight_layout()
        out_path = out_dir / f"radar_{algo}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved radar chart for {algo} to {out_path}")

# 4 & 7. Lorenz curves and token-economy coverage (calculation is related!!)

ALGO_TO_CLASS = {
    "bpe": BPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
    "tokenmonster": TokenMonsterTokenizer,
    "sre": SRETokenizer,
    "tokenmonster_sre": TokenMonsterSREHybridTokenizer
}


def _load_base_tokenizer(tokenizer_path: Path, algorithm: str) -> BaseTokenizer:
    cls = ALGO_TO_CLASS.get(algorithm)
    if cls is None:
        raise ValueError(f"I don't recognize this tokenizer algorithm: {algorithm}")
    return cls.load(tokenizer_path)


def _compute_token_frequencies(
    tokenizer: BaseTokenizer,
    texts: List[str],
    max_texts: int = 1000,
) -> np.ndarray:
    """encode sample of texts; return token frequency array
    """
    from collections import Counter

    counts = Counter()
    for t in texts[:max_texts]:
        token_ids = tokenizer.encode(t)
        counts.update(token_ids)
    freqs = np.array(list(counts.values()), dtype=float)
    return freqs


def _lorenz_from_freqs(freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """give back (x, y) for the Lorenz curve, given our freq array"""
    if freqs.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    sorted_vals = np.sort(freqs)
    cum_counts = np.cumsum(sorted_vals)
    total = cum_counts[-1]
    x = np.linspace(0, 1, len(sorted_vals))
    y = cum_counts / total
    return x, y


def plot_lorenz_and_coverage(
    df: pd.DataFrame,
    raw_results: Dict[str, Dict],
    results_dir: Path,
    out_dir: Path,
    max_texts: int = 800,
) -> None:
    
    exp_cfg = ExperimentConfig(
        corpus_source="sefaria_full",
        tractates=None,
        max_words_per_segment=40,
        min_words_per_segment=3,
        vocab_size=15000,
        output_dir=str(results_dir),
    )
    experiment = TalmudTokenizationExperiment(exp_cfg)
    print("Loading corpus for Lorenz/coverage plots...")
    experiment.load_corpus()
    test_texts = experiment.test_texts

    tok_dir = results_dir / "tokenizers"
    out_dir.mkdir(parents=True, exist_ok=True)

    #one line per algorithm (average across configs) on one curve
    plt.figure(figsize=(8, 6))

    for algo, group in df.groupby("algorithm"):
        lorenz_curves = []
        for _, row in group.iterrows():
            name = row["name"]
            cfg = raw_results[name]["config"]
            alg = cfg.get("algorithm", algo)
            tok_path = tok_dir / f"{name}.pkl"
            if not tok_path.exists():
                continue
            tokenizer = _load_base_tokenizer(tok_path, alg)
            freqs = _compute_token_frequencies(tokenizer, test_texts, max_texts=max_texts)
            if freqs.size == 0:
                continue
            x, y = _lorenz_from_freqs(freqs)
            lorenz_curves.append((x, y))

        if not lorenz_curves:
            continue
        #take avg of y over curves (same x length by padding/trim)
        min_len = min(len(y) for _, y in lorenz_curves)
        xs = lorenz_curves[0][0][:min_len]
        ys = np.vstack([y[:min_len] for _, y in lorenz_curves])
        y_mean = ys.mean(axis=0)
        plt.plot(xs, y_mean, label=algo, color=ALGO_COLORS.get(algo, None))

    plt.plot([0, 1], [0, 1], "k--", label="Equality line")
    plt.xlabel("Cumulative share of tokens")
    plt.ylabel("Cumulative share of frequency")
    plt.title("Lorenz Curves by Algorithm (averaged)")
    plt.legend()
    lorenz_path = out_dir / "lorenz_curves_by_algorithm.png"
    plt.tight_layout()
    plt.savefig(lorenz_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Lorenz curves to {lorenz_path}")

    #token-economy coverage, i.e. how many tokens for X% coverage
    coverage_thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    coverage_records = []

    for _, row in df.iterrows():
        name = row["name"]
        cfg = raw_results[name]["config"]
        alg = cfg.get("algorithm", row["algorithm"])
        tok_path = tok_dir / f"{name}.pkl"
        if not tok_path.exists():
            continue
        tokenizer = _load_base_tokenizer(tok_path, alg)
        freqs = _compute_token_frequencies(tokenizer, test_texts, max_texts=max_texts)
        if freqs.size == 0:
            continue
        freqs_sorted = np.sort(freqs)[::-1]
        cum = np.cumsum(freqs_sorted)
        total = cum[-1]
        for thr in coverage_thresholds:
            needed = int(np.searchsorted(cum, thr * total) + 1)
            coverage_records.append(
                {
                    "name": name,
                    "algorithm": alg,
                    "threshold": int(thr * 100),
                    "tokens_needed": needed,
                }
            )

    if coverage_records:
        cov_df = pd.DataFrame(coverage_records)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=cov_df,
            x="threshold",
            y="tokens_needed",
            hue="algorithm",
            palette=ALGO_COLORS,
        )
        plt.xlabel("Coverage threshold (%)")
        plt.ylabel("Tokens needed")
        plt.title("Token-Economy Coverage by Algorithm")
        plt.tight_layout()
        cov_path = out_dir / "token_coverage_by_algorithm.png"
        plt.savefig(cov_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved token coverage plot to {cov_path}")


# 5. Pareto frontier plots (NEEDS fixing)


def _compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str, maximize_x: bool, maximize_y: bool) -> pd.Series:

    arr = df[[x_col, y_col]].to_numpy()
    n = len(df)

    def better(i, j) -> bool:
        x_i, y_i = arr[i]
        x_j, y_j = arr[j]
        if maximize_x:
            cond_x = x_i >= x_j
        else:
            cond_x = x_i <= x_j
        if maximize_y:
            cond_y = y_i >= y_j
        else:
            cond_y = y_i <= y_j
        strictly = (x_i != x_j) or (y_i != y_j)
        return cond_x and cond_y and strictly

    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]:
            continue
        for j in range(n):
            if i == j or not is_front[j]:
                continue
            if better(j, i):
                is_front[i] = False
                break
    return pd.Series(is_front, index=df.index)


def plot_pareto_frontiers(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("renyi_entropy", "fertility", True, False, "renyi_vs_fertility"),
        ("nsl", "vocab_usage_percentage", False, True, "nsl_vs_vocab_usage"),
    ]

    for x_col, y_col, max_x, max_y, tag in pairs:
        df_local = df.copy()
        mask_front = _compute_pareto_front(df_local, x_col, y_col, max_x, max_y)

        plt.figure(figsize=(8, 6))
        #plot all points
        for algo, group in df_local.groupby("algorithm"):
            plt.scatter(
                group[x_col],
                group[y_col],
                c=ALGO_COLORS.get(algo, None),
                alpha=0.4,
                label=algo,
            )

        #make frontier
        front = df_local[mask_front]
        plt.scatter(
            front[x_col],
            front[y_col],
            c="black",
            s=80,
            marker="*",
            label="Pareto frontier",
        )

        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.title(f"Pareto Frontier: {x_col} vs {y_col}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out_path = out_dir / f"pareto_{tag}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Pareto plot {tag} to {out_path}")


# 6. metrics heatmap with glyphs


def plot_enhanced_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """heatmap with vocab_usage, fertility, gini."""
    metrics = ["renyi_entropy", "nsl", "fertility", "zipfian_alignment"]
    df_heat = df.set_index("name")
    mat = df_heat[metrics].to_numpy()

    mat_norm = (mat - mat.min(axis=0)) / (mat.max(axis=0) - mat.min(axis=0) + 1e-9)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(df_heat))))
    sns.heatmap(
        mat_norm,
        ax=ax,
        cmap="RdYlGn",
        cbar_kws={"label": "Normalized score"},
        yticklabels=df_heat.index,
        xticklabels=[m.replace("_", " ").title() for m in metrics],
    )

    #in overlay, the circle size = vocab_usage, color = fertility, edge color = gini
    usage = df_heat["vocab_usage_percentage"].to_numpy()
    fert = df_heat["fertility"].to_numpy()
    gini = df_heat["gini_coefficient"].to_numpy()

    usage_n = (usage - usage.min()) / (usage.max() - usage.min() + 1e-9)
    fert_n = (fert - fert.min()) / (fert.max() - fert.min() + 1e-9)
    gini_n = (gini - gini.min()) / (gini.max() - gini.min() + 1e-9)

    num_rows = len(df_heat)
    for i in range(num_rows):
        x = len(metrics) + 0.5
        y = i + 0.5
        size = 100 + 400 * usage_n[i]
        color = plt.cm.Blues(fert_n[i])
        edge = plt.cm.Greys(gini_n[i])
        ax.scatter(
            x,
            y,
            s=size,
            c=[color],
            edgecolors=[edge],
            linewidths=1,
        )

    ax.set_xlim(0, len(metrics) + 1.5)
    ax.set_title("Metrics Heatmap with Vocabulary Glyphs (size=usage, color=fertility, edge=gini)")

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="High vocab usage (bigger)",
               markerfacecolor="lightblue", markersize=10, markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", label="High fertility (bluer)",
               markerfacecolor=plt.cm.Blues(0.9), markersize=10, markeredgecolor="k"),
        Line2D([0], [0], marker="o", color="w", label="High gini (darker edge)",
               markerfacecolor="none", markeredgecolor="black", markersize=10, linewidth=2),
    ]
    ax.legend(
        handles=legend_elems,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


# 8. Segmentation strips (TODO fix this; displays way too small)


def _load_any_tokenizer(name: str, algorithm: str, tok_dir: Path) -> BaseTokenizer:
    tok_path = tok_dir / f"{name}.pkl"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer file not found: {tok_path}")
    return _load_base_tokenizer(tok_path, algorithm)


def plot_segmentation_strips(
    df: pd.DataFrame,
    raw_results: Dict[str, Dict],
    results_dir: Path,
    out_dir: Path,
    num_examples: int = 3,
) -> None:
    """segmentation strips for a few sample sentences.
    take `num_examples` sentences from test set and have one row per tokenizer and one column per example
    """
    exp_cfg = ExperimentConfig(
        corpus_source="sefaria_full",
        tractates=None,
        max_words_per_segment=40,
        min_words_per_segment=3,
        vocab_size=15000,
        output_dir=str(results_dir),
    )
    experiment = TalmudTokenizationExperiment(exp_cfg)
    print("Loading corpus for segmentation strips...")
    experiment.load_corpus()
    test_texts = experiment.test_texts
    if not test_texts:
        print("no test texts available. skipping segmentation strips")
        return

    examples = test_texts[:num_examples]
    tok_dir = results_dir / "tokenizers"
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen_rows = []
    seen_algos = set()
    for _, row in df.sort_values("algorithm").iterrows():
        algo = row["algorithm"]
        if algo in seen_algos:
            continue
        if row["use_bpe_dropout"]:
            continue
        chosen_rows.append(row)
        seen_algos.add(algo)

    if not chosen_rows:
        #just take all rows
        chosen_rows = [r for _, r in df.iterrows()]

    n_tok = len(chosen_rows)
    n_ex = len(examples)

    fig, axes = plt.subplots(
        n_tok,
        n_ex,
        figsize=(4 * n_ex, 1.6 * n_tok),
        squeeze=False,
    )

    for i, row in enumerate(chosen_rows):
        name = row["name"]
        cfg = raw_results[name]["config"]
        algo = cfg.get("algorithm", row["algorithm"])
        tokenizer = _load_any_tokenizer(name, algo, tok_dir)

        for j, text in enumerate(examples):
            ax = axes[i][j]
            ax.set_axis_off()

            token_ids = tokenizer.encode(text)
            tokens = [tokenizer.inverse_vocab.get(tid, "[UNK]") for tid in token_ids]

            x = 0.0
            for tok in tokens[:50]:  
                width = max(0.4, min(2.0, 0.3 * len(tok)))
                rect = plt.Rectangle(
                    (x, 0),
                    width,
                    1,
                    facecolor="lightblue",
                    edgecolor="k",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.text(
                    x + width / 2,
                    0.5,
                    tok,
                    ha="center",
                    va="center",
                    fontsize=6,
                )
                x += width

            ax.set_xlim(0, x)
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel(algo, rotation=0, labelpad=30, va="center")
            if i == 0:
                ax.set_title(f"Example {j + 1}", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "segmentation_strips.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved segmentation strips to {out_path}")


# CLI implementation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="make advanced visualizations from results.json",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="directory containing results.json and tokenizers/",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="make all of the visualizations now",
    )
    parser.add_argument("--constellation", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--radar", action="store_true")
    parser.add_argument("--lorenz", action="store_true")
    parser.add_argument("--pareto", action="store_true")
    parser.add_argument("--enhanced-heatmap", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--segmentation", action="store_true")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found at {results_path}")

    df, raw = load_results(results_path)
    if df.empty:
        print("No results found in results.json")
        return 0

    viz_dir = results_dir / "advanced_visualizations"

    run_all = args.generate_all

    if run_all or args.constellation:
        plot_constellation(df, viz_dir)

    if run_all or args.parallel:
        plot_parallel_coordinates(df, viz_dir)

    if run_all or args.radar:
        plot_radar_per_algorithm(df, viz_dir)

    if run_all or args.lorenz or run_all or args.coverage:
        plot_lorenz_and_coverage(df, raw, results_dir, viz_dir)

    if run_all or args.pareto:
        plot_pareto_frontiers(df, viz_dir)

    if run_all or args.enhanced_heatmap:
        plot_enhanced_heatmap(df, viz_dir)

    if run_all or args.segmentation:
        plot_segmentation_strips(df, raw, results_dir, viz_dir)

    print("Advanced visualizations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
