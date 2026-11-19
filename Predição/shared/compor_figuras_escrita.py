#!/usr/bin/env python3
"""Compõe painéis auxiliares 1×3 (QB, RB, WR) das figuras da EDA e das curvas de Kaplan–Meier.

A tese não usa esses painéis: usa os PDFs individuais, levados para `Escrita Final/figuras/` pelo passo 47.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPOSITES = {
    # EDA
    ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/depth_comparado.png": [
        ROOT / "Análise Exploratória de Dados/figuras/depth_QB.png",
        ROOT / "Análise Exploratória de Dados/figuras/depth_RB.png",
        ROOT / "Análise Exploratória de Dados/figuras/depth_WR.png",
    ],
    ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/dist_carreira_comparado.png": [
        ROOT / "Análise Exploratória de Dados/figuras/dist_carreira_QB.png",
        ROOT / "Análise Exploratória de Dados/figuras/dist_carreira_RB.png",
        ROOT / "Análise Exploratória de Dados/figuras/dist_carreira_WR.png",
    ],
    ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/draft_comparado.png": [
        ROOT / "Análise Exploratória de Dados/figuras/draft_QB.png",
        ROOT / "Análise Exploratória de Dados/figuras/draft_RB.png",
        ROOT / "Análise Exploratória de Dados/figuras/draft_WR.png",
    ],
    ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/games_missed_comparado.png": [
        ROOT / "Análise Exploratória de Dados/figuras/games_missed_QB.png",
        ROOT / "Análise Exploratória de Dados/figuras/games_missed_RB.png",
        ROOT / "Análise Exploratória de Dados/figuras/games_missed_WR.png",
    ],
    ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/idade_comparado.png": [
        ROOT / "Análise Exploratória de Dados/figuras/idade_QB.png",
        ROOT / "Análise Exploratória de Dados/figuras/idade_RB.png",
        ROOT / "Análise Exploratória de Dados/figuras/idade_WR.png",
    ],
    ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/offense_pct_comparado.png": [
        ROOT / "Análise Exploratória de Dados/figuras/offense_pct_QB.png",
        ROOT / "Análise Exploratória de Dados/figuras/offense_pct_RB.png",
        ROOT / "Análise Exploratória de Dados/figuras/offense_pct_WR.png",
    ],
    # KM Curves
    ROOT / "Inferência/Modelagem de COX/Curvas/figuras/figuras para escrita/km_depth_comparado.png": [
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_depth_QB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_depth_RB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_depth_WR.png",
    ],
    ROOT / "Inferência/Modelagem de COX/Curvas/figuras/figuras para escrita/km_draft_comparado.png": [
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_draft_QB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_draft_RB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_draft_WR.png",
    ],
    ROOT / "Inferência/Modelagem de COX/Curvas/figuras/figuras para escrita/km_global_comparado.png": [
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_global_QB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_global_RB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_global_WR.png",
    ],
    ROOT / "Inferência/Modelagem de COX/Curvas/figuras/figuras para escrita/km_idade_comparado.png": [
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_idade_QB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_idade_RB.png",
        ROOT / "Inferência/Modelagem de COX/Curvas/figuras/km_idade_WR.png",
    ],
    # COX
    ROOT / "Inferência/Modelagem de COX/figuras/figuras para escrita/forest_plot_comparado.png": [
        ROOT / "Inferência/Modelagem de COX/figuras/forest_plot_QB.png",
        ROOT / "Inferência/Modelagem de COX/figuras/forest_plot_RB.png",
        ROOT / "Inferência/Modelagem de COX/figuras/forest_plot_WR.png",
    ],
    ROOT / "Inferência/Modelagem de COX/figuras/figuras para escrita/schoenfeld_comparado.png": [
        ROOT / "Inferência/Modelagem de COX/figuras/schoenfeld_QB.png",
        ROOT / "Inferência/Modelagem de COX/figuras/schoenfeld_RB.png",
        ROOT / "Inferência/Modelagem de COX/figuras/schoenfeld_WR.png",
    ],
    # Preparação
    ROOT / "Inferência/Preparação Modelagem/figuras/figuras para escrita/taxa_evento_comparado.png": [
        ROOT / "Inferência/Preparação Modelagem/figuras/taxa_evento_QB.png",
        ROOT / "Inferência/Preparação Modelagem/figuras/taxa_evento_RB.png",
        ROOT / "Inferência/Preparação Modelagem/figuras/taxa_evento_WR.png",
    ],
}

POS_LABELS = {"QB": "Quarterbacks", "RB": "Running Backs", "WR": "Wide Receivers"}

TEMPORAL_SOURCES = [
    ROOT / "Análise Exploratória de Dados/figuras/temporal_QB.png",
    ROOT / "Análise Exploratória de Dados/figuras/temporal_RB.png",
    ROOT / "Análise Exploratória de Dados/figuras/temporal_WR.png",
]
TEMPORAL_OUTPUT = ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/temporal_comparado.png"

CORR_SOURCES = [
    ROOT / "Análise Exploratória de Dados/figuras/corr_matrix_QB.png",
    ROOT / "Análise Exploratória de Dados/figuras/corr_matrix_RB.png",
    ROOT / "Análise Exploratória de Dados/figuras/corr_matrix_WR.png",
]
CORR_OUTPUT = ROOT / "Análise Exploratória de Dados/figuras/figuras para escrita/corr_matrix_comparado.png"


def _split_horizontal(img):
    """Divide figura 1×2 (dois painéis lado a lado) em metades esquerda/direita."""
    h, w = img.shape[0], img.shape[1]
    mid = w // 2
    return img[:, :mid], img[:, mid:]


def compose_vertical_3(sources, output, figsize=(14, 28)):
    """Empilha três figuras em coluna (largura total por painel)."""
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"Pulando {output.name}: fontes ausentes: {[s.name for s in missing]}")
        return False

    imgs = [mpimg.imread(s) for s in sources]
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    for ax, img, pos in zip(axes, imgs, ["QB", "RB", "WR"]):
        ax.imshow(img)
        ax.set_aspect("auto")
        ax.set_title(POS_LABELS[pos], fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gerado: {output.relative_to(ROOT)}")
    return True


def compose_temporal_2x3(sources, output):
    """Monta grade 2×3: linha 1 = jogadores/temporada; linha 2 = experiência média."""
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"Pulando {output.name}: fontes ausentes: {[s.name for s in missing]}")
        return False

    ativos, exp = [], []
    for path, pos in zip(sources, ["QB", "RB", "WR"]):
        img = mpimg.imread(path)
        left, right = _split_horizontal(img)
        ativos.append((left, f"{POS_LABELS[pos]} — ativos"))
        exp.append((right, f"{POS_LABELS[pos]} — exp."))

    row1 = ativos
    row2 = exp

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    for ax, (img, title) in zip(axes[0], row1):
        ax.imshow(img)
        ax.set_aspect("auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    for ax, (img, title) in zip(axes[1], row2):
        ax.imshow(img)
        ax.set_aspect("auto")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gerado: {output.relative_to(ROOT)}")
    return True


def main():
    generated = 0
    for output, sources in COMPOSITES.items():
        missing = [s for s in sources if not s.exists()]
        if missing:
            print(f"Pulando {output.name}: fontes ausentes: {[s.name for s in missing]}")
            continue

        imgs = [mpimg.imread(s) for s in sources]

        # Find common height
        heights = [img.shape[0] for img in imgs]
        widths = [img.shape[1] for img in imgs]
        max_h = max(heights)

        fig, axes = plt.subplots(1, 3, figsize=(22, 8.0))
        for ax, img, pos in zip(axes, imgs, ["QB", "RB", "WR"]):
            ax.imshow(img)
            ax.set_aspect('auto')
            ax.set_title(POS_LABELS[pos], fontsize=14, fontweight="bold")
            ax.axis("off")

        plt.tight_layout(pad=1.2)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        generated += 1
        print(f"Gerado: {output.relative_to(ROOT)}")

    if compose_temporal_2x3(TEMPORAL_SOURCES, TEMPORAL_OUTPUT):
        generated += 1
    if compose_vertical_3(CORR_SOURCES, CORR_OUTPUT):
        generated += 1

    total = len(COMPOSITES) + 2
    print(f"\n{generated}/{total} figuras compostas geradas.")

if __name__ == "__main__":
    main()
