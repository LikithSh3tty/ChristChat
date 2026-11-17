"""Charts comparing the three approaches, drawn from measured result files only.

Run after experiments.evaluation:  python -m experiments.comparison
Writes PNGs to experiments/results/figures/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common.config import RESULTS_DIR  # noqa: E402

FIG = RESULTS_DIR / "figures"
FIG.mkdir(exist_ok=True)

# Categorical slots 1-3 of the validated reference palette (fixed order, never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8})


def label(row):
    return "LLM only" if row["system"] == "llm_closed_book" else f"RAG {row['prompt']}"


def grouped_bars(df, metric, title, fname, pct=True, ylabel=None):
    df = df.assign(variant=df.apply(label, axis=1))
    variants = list(dict.fromkeys(df["variant"]))
    models = list(dict.fromkeys(df["model"]))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        sub = df[df.variant == v].set_index("model").reindex(models)
        xs = [j + (i - (len(variants) - 1) / 2) * width for j in range(len(models))]
        bars = ax.bar(xs, sub[metric], width - 0.03, color=SERIES[i], label=v, zorder=2)
        for b, val in zip(bars, sub[metric]):
            if pd.notna(val):
                ax.annotate(f"{val:.0%}" if pct else f"{val:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(models)), models)
    if pct:
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel(ylabel or "")
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, ncol=len(variants), loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def retrieval_sweep():
    p = RESULTS_DIR / "model2_retrieval_sweep.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).groupby(["chunk_words", "k"])["hit"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for i, (cw, g) in enumerate(d.groupby("chunk_words")):
        ax.plot(g["k"], g["hit"], color=SERIES[i], lw=2, marker="o", ms=6, zorder=2)
        ax.annotate(f"{cw}-word chunks", (g["k"].iloc[-1], g["hit"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xticks(sorted(d["k"].unique()))
    ax.set_xlim(0.7, d["k"].max() + 1.8)
    ax.set_ylim(0.5, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Passages retrieved (k)")
    ax.set_title("Gold page found in top-k passages (23 answerable questions)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "retrieval_hit_rate.png", dpi=150)
    plt.close(fig)


def peft_chart():
    """Base vs LoRA on the same RAG inputs: four behaviour metrics side by side."""
    p = RESULTS_DIR / "model4_peft.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    metrics = {"correct": "Accuracy", "hallucinated": "Hallucination", "refused": "Refusal", "cites_page": "Cites a page"}
    agg = d.groupby("variant")[list(metrics)].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4))
    width = 0.38
    for i, (variant, name) in enumerate([("base", "Qwen2.5 0.5B base"), ("lora", "Qwen2.5 0.5B + LoRA")]):
        xs = [j + (i - 0.5) * width for j in range(len(metrics))]
        bars = ax.bar(xs, agg.loc[variant], width - 0.03, color=SERIES[i], label=name, zorder=2)
        for b, val in zip(bars, agg.loc[variant]):
            ax.annotate(f"{val:.0%}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                        fontsize=9, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(metrics)), list(metrics.values()))
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("PEFT: same RAG inputs, with and without the LoRA adapter", loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(FIG / "peft_base_vs_lora.png", dpi=150)
    plt.close(fig)


def main():
    s = pd.read_csv(RESULTS_DIR / "comparison_summary.csv")
    s = s[s["system"].isin(["llm_closed_book", "rag"])]
    peft_chart()
    grouped_bars(s, "accuracy", "Answer accuracy on 28 handbook questions", "accuracy.png")
    grouped_bars(s, "hallucination_rate", "Hallucination rate (confident wrong answers)", "hallucination.png")
    grouped_bars(s, "latency_p50_s", "Median response time per question (CPU)", "latency.png", pct=False, ylabel="seconds")
    retrieval_sweep()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
