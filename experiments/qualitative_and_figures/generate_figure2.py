"""
generate_figure2.py — Génération de la Figure 2 du papier WI-IAT 2026
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Produit :
  - results/figure2_ablation.pdf  → pour Overleaf (vecteur, haute qualité)
  - results/figure2_ablation.png  → pour aperçu rapide

Figure 2 : Grouped bar chart montrant ASR, RD, CI et AR
par configuration de défense.
Montre visuellement qu'aucune couche seule n'est optimale
et que All defenses offre le meilleur compromis.

Usage :
  pip install matplotlib numpy
  python generate_figure2.py
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "pdf.fonttype":    42,   # fonts embarquées dans le PDF
    "ps.fonttype":     42,
})

# ── Données issues des expériences ───────────────────────────────────────────

CONFIGS = [
    "No defense",
    "Filter only",
    "Isolate only",
    "Verify only",
    "All defenses",
]

# Résultats ablation_v2 (dataset custom, 10 questions, 6 attaques)
ASR = [0.233, 0.017, 0.217, 0.000, 0.000]   # Attack Success Rate
RD  = [0.191, 0.002, 0.256, 0.304, 0.097]   # Robustness Degradation
CI  = [0.917, 0.975, 0.921, 1.000, 1.000]   # Context Integrity
AR  = [0.711, 0.833, 0.664, 0.553, 0.725]   # Answer Relevance


def make_figure2():
    """
    Génère le grouped bar chart pour la Figure 2 du papier.

    Layout : 2x2 subplots, un par métrique.
    Chaque subplot montre les 5 configurations en barres.
    La configuration "All defenses" est mise en évidence.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))  # double colonne IEEE
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    x = np.arange(len(CONFIGS))
    width = 0.55

    # Palette : gris pour les configs simples, couleur pour All defenses
    colors = ["#B0B8C1", "#B0B8C1", "#B0B8C1", "#B0B8C1", "#2C7BB6"]
    edge_colors = ["#7A8894"] * 4 + ["#1A5276"]

    metrics = [
        (axes[0, 0], ASR, "Attack Success Rate (ASR)",
         "ASR", True,  "↓ lower is better"),
        (axes[0, 1], RD,  "Robustness Degradation (RD)",
         "RD",  True,  "↓ lower is better"),
        (axes[1, 0], CI,  "Context Integrity (CI)",
         "CI",  False, "↑ higher is better"),
        (axes[1, 1], AR,  "Answer Relevance (AR)",
         "AR",  False, "↑ higher is better"),
    ]

    short_labels = ["No\ndef.", "Filter\nonly", "Isolate\nonly",
                    "Verify\nonly", "All\ndef."]

    for ax, values, title, ylabel, lower_better, hint in metrics:
        bars = ax.bar(
            x, values, width,
            color=colors,
            edgecolor=edge_colors,
            linewidth=0.6,
            zorder=3,
        )

        # Ligne de référence baseline (No defense)
        ax.axhline(
            values[0], color="#E74C3C", linewidth=0.8,
            linestyle="--", alpha=0.6, zorder=2,
            label="No defense baseline"
        )

        # Valeurs au-dessus des barres
        for bar, val in zip(bars, values):
            ypos = bar.get_height() + 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:.2f}" if val > 0 else "0",
                ha="center", va="bottom",
                fontsize=7.5, color="#2C2C2C",
            )

        # Mettre en évidence "All defenses"
        bars[-1].set_linewidth(1.5)

        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylim(0, max(values) * 1.25 + 0.05)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Hint direction
        ax.text(
            0.98, 0.96, hint,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7, color="#666666",
            style="italic",
        )

    # Légende commune en bas
    patch_base = mpatches.Patch(
        facecolor="#B0B8C1", edgecolor="#7A8894", linewidth=0.6,
        label="Single-layer / No defense"
    )
    patch_all = mpatches.Patch(
        facecolor="#2C7BB6", edgecolor="#1A5276", linewidth=1.5,
        label="All defenses (proposed)"
    )
    line_ref = plt.Line2D(
        [0], [0], color="#E74C3C", linewidth=0.8,
        linestyle="--", alpha=0.8,
        label="No-defense baseline"
    )

    fig.legend(
        handles=[patch_base, patch_all, line_ref],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Fig. 2. Ablation study: ASR, RD, CI and AR across defense configurations\n"
        "(dataset: 10 queries × 6 attack strategies; generator: GPT-3.5-turbo)",
        fontsize=9, y=1.01,
    )

    return fig


def make_figure2b():
    """
    Figure alternative : line plot montrant l'évolution des métriques
    de No defense → All defenses. Plus compact pour une colonne IEEE.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(CONFIGS))

    metrics_line = [
        (ASR, "ASR", "#E74C3C", "o", "--"),
        (RD,  "RD",  "#E67E22", "s", "-."),
        (CI,  "CI",  "#27AE60", "^", "-"),
        (AR,  "AR",  "#2980B9", "D", ":"),
    ]

    for values, label, color, marker, ls in metrics_line:
        ax.plot(
            x, values,
            color=color, marker=marker, linestyle=ls,
            linewidth=1.4, markersize=5, label=label, zorder=3,
        )

    ax.axvline(4, color="#2C7BB6", linewidth=1.0, alpha=0.4, zorder=2)
    ax.text(4.05, 0.02, "All def.", fontsize=7.5, color="#2C7BB6", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["No\ndef.", "Filter", "Isolate", "Verify", "All\ndef."],
        fontsize=8
    )
    ax.set_ylabel("Metric value", fontsize=9)
    ax.set_ylim(-0.02, 1.15)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right", fontsize=7.5, ncol=2,
        frameon=True, framealpha=0.9,
    )
    ax.set_title(
        "Fig. 2. Defense ablation study\n(ASR↓, RD↓, CI↑, AR↑)",
        fontsize=9, fontweight="bold",
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)

    print("Génération Figure 2a (grouped bar chart)...")
    fig_a = make_figure2()
    fig_a.savefig("results/figure2a_ablation.pdf",
                  bbox_inches="tight", dpi=300)
    fig_a.savefig("results/figure2a_ablation.png",
                  bbox_inches="tight", dpi=300)
    print("  → results/figure2a_ablation.pdf")
    print("  → results/figure2a_ablation.png")

    print("Génération Figure 2b (line plot compact)...")
    fig_b = make_figure2b()
    fig_b.savefig("results/figure2b_ablation.pdf",
                  bbox_inches="tight", dpi=300)
    fig_b.savefig("results/figure2b_ablation.png",
                  bbox_inches="tight", dpi=300)
    print("  → results/figure2b_ablation.pdf")
    print("  → results/figure2b_ablation.png")

    print("\nFigures générées.")
    print("Recommandation : utiliser figure2b (line plot) pour une")
    print("colonne IEEE, figure2a pour une double colonne.")