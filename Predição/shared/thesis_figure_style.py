#!/usr/bin/env python3
"""Calibração de figuras para inclusão em LaTeX sem perda de legibilidade."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# UFSCar main.tex: A4, lmargin=3cm, rmargin=2cm → textwidth = 160mm
TEXTWIDTH_MM = 160.0
TEXTWIDTH_IN = TEXTWIDTH_MM / 25.4  # ≈ 6.299 in
TEXTWIDTH_PT = TEXTWIDTH_IN * 72.27  # ≈ 455.2 pt
BODY_FONT_PT = 12
LATEX_WIDTH_FRAC = 0.98  # typical \includegraphics width in c3


def apply_thesis_rcparams(font_scale: float = 1.0) -> None:
    base = BODY_FONT_PT * font_scale
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base + 1,
            "legend.fontsize": max(base - 1, 9),
            "xtick.labelsize": max(base - 1, 9),
            "ytick.labelsize": max(base - 1, 9),
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figsize_full(width_frac: float = LATEX_WIDTH_FRAC, aspect: float = 0.62) -> tuple[float, float]:
    """Return (width, height) in inches matching final LaTeX width."""
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)


def figsize_dual_horizontal(
    width_frac: float = LATEX_WIDTH_FRAC,
    aspect: float = 0.40,
) -> tuple[float, float]:
    """Return (width, height) for 1×2 dual-panel figures at \\textwidth."""
    return figsize_full(width_frac=width_frac, aspect=aspect)


def finalize_dual_horizontal(
    fig,
    *,
    bottom: float = 0.22,
    left: float = 0.12,
    wspace: float = 0.45,
    left_labelpad: float = 20,
    right_labelpad: float = 12,
) -> None:
    """Symmetric side-by-side panels: room for Y-labels and gap between panels."""
    fig.subplots_adjust(left=left, right=0.99, top=0.94, bottom=bottom, wspace=wspace)
    axes = fig.get_axes()
    if len(axes) >= 1:
        axes[0].yaxis.labelpad = left_labelpad
    if len(axes) >= 2:
        axes[1].yaxis.labelpad = right_labelpad


def format_p_value(p: float, *, threshold: float = 0.0001, decimals: int = 4) -> str:
    """Formata p-valor para anotações em figuras (evita 0,0000 quando p é muito pequeno)."""
    if p is None or (isinstance(p, float) and (p != p)):  # NaN
        return "NA"
    if p < threshold:
        return f"< {threshold:.{decimals}f}"
    return f"= {p:.{decimals}f}"


# ---------------------------------------------------------------- virgula decimal
def num_br(valor: float, casas: int | None = None) -> str:
    """Numero com virgula decimal. `casas=None` usa a forma curta (%g): 0,2; 1; 2,5."""
    if valor is None or (isinstance(valor, float) and valor != valor):
        return "NA"
    texto = f"{valor:g}" if casas is None else f"{valor:.{casas}f}"
    if texto in ("-0", "-0,0", "-0.0"):
        texto = texto[1:]
    return texto.replace(".", ",").replace("-", "\u2212")


def virgula_fmt(casas: int | str | None = None, axis=None):
    """FuncFormatter que escreve os ticks com virgula decimal (e sinal de menos tipografico).

    `casas="auto"` (com `axis`) usa o menor numero de casas que distingue todos os ticks do eixo.
    """
    from matplotlib.ticker import FuncFormatter

    def _fmt(v, _pos):
        n = casas
        if n == "auto":
            n = 0
            locs = [float(x) for x in axis.get_majorticklocs()] if axis is not None else [float(v)]
            while n < 4 and any(abs(x - round(x, n)) > 1e-9 for x in locs):
                n += 1
        return num_br(float(v), n)

    return FuncFormatter(_fmt)


def eixos_virgula(ax, x: bool = True, y: bool = True, casas_x: int | str | None = "auto", casas_y: int | str | None = "auto") -> None:
    """Aplica virgula decimal aos ticks numericos dos eixos escolhidos."""
    if x:
        ax.xaxis.set_major_formatter(virgula_fmt(casas_x, ax.xaxis))
    if y:
        ax.yaxis.set_major_formatter(virgula_fmt(casas_y, ax.yaxis))


def effective_font_pt(source_font_pt: float, figsize_width_in: float, latex_width_frac: float = LATEX_WIDTH_FRAC) -> float:
    latex_w = TEXTWIDTH_IN * latex_width_frac
    return source_font_pt * (latex_w / figsize_width_in)


def plot_cox_forest_thesis(
    smry,
    posicao: str,
    path: str | Path,
    *,
    cores: dict[str, str] | None = None,
) -> Path:
    """Forest plot Cox TDC: colunas separadas para rótulos, barras e estatísticas."""
    import numpy as np

    cores = cores or {"risco": "#D85A30", "protetor": "#1D9E75"}
    smry = smry.copy().sort_values("exp(coef)")
    n = len(smry)
    # Largura total do texto e altura fixa: tres paineis empilhados (um por posicao) cabem
    # numa pagina com \thesisfigforestfull; a escala no LaTeX fica perto de 1 e os rotulos
    # mantem o tamanho nominal.
    fh = 2.85
    fw = TEXTWIDTH_IN * LATEX_WIDTH_FRAC

    max_lbl = max(len(str(s)) for s in smry.index)
    lbl_r = min(2.0, max(1.25, max_lbl / 17.0))
    plot_r = 2.75
    txt_r = 1.0

    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[lbl_r, plot_r, txt_r],
        wspace=0.05,
        left=0.05,
        right=0.98,
        top=0.97,
        bottom=0.17,
    )
    ax_lbl = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax_txt = fig.add_subplot(gs[2])

    y_pos = list(range(n))
    log_hr = np.log(smry["exp(coef)"].values)
    err_lo = log_hr - np.log(smry["exp(coef) lower 95%"].values)
    err_hi = np.log(smry["exp(coef) upper 95%"].values) - log_hr
    colors = [cores["risco"] if v > 0 else cores["protetor"] for v in log_hr]

    ax.barh(
        y_pos,
        log_hr,
        xerr=[err_lo, err_hi],
        height=0.58,
        color=colors,
        alpha=0.85,
        capsize=4,
        error_kw={"linewidth": 1.2, "ecolor": "#444444"},
    )
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    x_hi = float(np.nanmax(log_hr + err_hi))
    x_lo = float(np.nanmin(log_hr - err_lo))
    span = max(x_hi - x_lo, 0.25)
    pad = max(0.10, 0.12 * span)
    ax.set_xlim(min(-0.15, x_lo - pad), max(0.15, x_hi + pad))

    ylim = (-0.5, n - 0.5)
    ax_lbl.set_ylim(ylim)
    ax.set_ylim(ylim)
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_xticks([])
    ax_lbl.set_yticks([])
    for spine in ax_lbl.spines.values():
        spine.set_visible(False)
    for i, name in enumerate(smry.index):
        ax_lbl.text(
            0.98,
            i,
            thesis_var_label(str(name)),
            ha="right",
            va="center",
            fontsize=10,
            clip_on=False,
        )

    ax.set_yticks([])
    ax.tick_params(left=False)
    ax.set_xlabel(
        "log(HR) — por 1 DP  |  laranja = ↑ risco   verde = ↓ risco",
        fontsize=11,
    )
    eixos_virgula(ax, x=True, y=False)

    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(-0.5, n - 0.5)
    for i, (_, row) in enumerate(smry.iterrows()):
        hr = row["exp(coef)"]
        lo = row["exp(coef) lower 95%"]
        hi = row["exp(coef) upper 95%"]
        p = row["p"]
        pstr = "p < 0,001" if p < 0.001 else f"p = {num_br(p, 3)}"
        ax_txt.text(
            0.0,
            i,
            f"HR={num_br(hr, 2)}\n({num_br(lo, 2)}-{num_br(hi, 2)}), {pstr}",
            va="center",
            ha="left",
            fontsize=9,
            linespacing=1.05,
            clip_on=False,
        )

    path = Path(path)
    pdf_path = path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None, pad_inches=0.08)
    plt.close(fig)
    return pdf_path


def choose_aalen_grid(n_terms: int) -> tuple[int, int]:
    """Três painéis por fileira — equilíbrio entre largura (títulos) e altura."""
    import math

    ncols = 3
    return ncols, math.ceil(n_terms / ncols)


def aalen_height_frac(n_terms: int) -> float:
    """Altura PDF alinhada ao número de fileiras (3 colunas)."""
    import math

    nrows = math.ceil(n_terms / 3)
    return {3: 0.73, 4: 0.78, 5: 0.83}.get(nrows, 0.85)


def aalen_latex_height_cmd(n_terms: int) -> str:
    """Nome da macro LaTeX correspondente à altura gerada."""
    hf = aalen_height_frac(n_terms)
    if hf >= 0.83:
        return r"\thesisfigaalenwr"
    if hf >= 0.78:
        return r"\thesisfigaalenq"
    return r"\thesisfigaalen"


def _carrega_glossario() -> dict[str, str]:
    """Rótulos das covariáveis: fonte única, a mesma do glossário da tese.

    O CSV é lido também pelo script R do Schoenfeld; figura, tabela e texto usam
    o mesmo nome para a mesma variável.
    """
    import csv

    caminho = Path(__file__).resolve().parent / "glossario_variaveis.csv"
    with caminho.open(encoding="utf-8") as fh:
        return {linha["variavel"]: linha["rotulo"] for linha in csv.DictReader(fh)}


THESIS_VAR_LABELS: dict[str, str] = _carrega_glossario()

# Nome das famílias de modelo como aparecem no texto.
THESIS_MODEL_LABELS: dict[str, str] = {
    "km": "Kaplan–Meier",
    "cox": "Cox",
    "gbsa": "GBSA",
    "rsf": "RSF",
    "aalen": "Aalen",
}


def thesis_var_label(term: str) -> str:
    if term in THESIS_VAR_LABELS:
        return THESIS_VAR_LABELS[term]
    # Indicador de ausencia de uma taxa imputada: rotulo da taxa + "(ausente)".
    if term.endswith("_missing") and term[: -len("_missing")] in THESIS_VAR_LABELS:
        return THESIS_VAR_LABELS[term[: -len("_missing")]] + " (ausente)"
    raise KeyError(f"covariável sem rótulo no glossário: {term}")


def thesis_model_label(modelo: str) -> str:
    return THESIS_MODEL_LABELS.get(str(modelo).lower(), str(modelo))


def _aalen_panel_label(term: str) -> str:
    return thesis_var_label(term)


def plot_aalen_cum_coefs_thesis(
    cum_df,
    var_df,
    posicao: str,
    path: str | Path,
    *,
    height_frac: float = 0.68,
    ncols: int | None = None,
    cor: str = "#2E4057",
) -> Path:
    """Painel Aalen: faixa de título por coluna + gráfico abaixo (sem invadir vizinho)."""
    import math
    import textwrap

    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    textheight_in = (29.7 - 2.5 - 2.0) / 2.54
    fw = TEXTWIDTH_IN * LATEX_WIDTH_FRAC
    fh = textheight_in * height_frac

    time_col = cum_df.columns[0]
    terms = [c for c in cum_df.columns if c not in (time_col, "(Intercept)")]
    n_terms = len(terms)
    if ncols is None:
        ncols, nrows = choose_aalen_grid(n_terms)
    else:
        nrows = math.ceil(n_terms / ncols)

    fs_title = max(7.0, min(8.5, 30.0 / nrows))
    fs_label = max(7.5, min(9.5, 40.0 / nrows))
    fs_xtick = max(7.0, fs_label - 0.5)
    lw = max(1.3, min(1.9, 9.0 / nrows))
    title_band = 0.42
    wrap_w = max(22, int(42 / ncols))

    fig = plt.figure(figsize=(fw, fh))
    outer_gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        left=0.13,
        right=0.99,
        top=0.98,
        bottom=0.10,
        hspace=0.52,
        wspace=0.48,
    )
    t_max = float(np.nanmax(cum_df[time_col].values))
    plot_axes: list = []

    for idx, term in enumerate(terms):
        row, col = divmod(idx, ncols)
        is_bottom = row == nrows - 1
        is_left = col == 0
        inner = outer_gs[row, col].subgridspec(
            2,
            1,
            height_ratios=[title_band, 1.0],
            hspace=0.04,
        )

        title_ax = fig.add_subplot(inner[0])
        title_ax.set_axis_off()
        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        label = textwrap.fill(_aalen_panel_label(term), width=wrap_w, break_long_words=False)
        title_ax.text(
            0.0,
            0.85,
            label,
            ha="left",
            va="top",
            fontsize=fs_title,
            fontweight="bold",
            linespacing=1.05,
            clip_on=False,
            transform=title_ax.transAxes,
        )

        plot_ax = fig.add_subplot(inner[1])
        plot_axes.append(plot_ax)

        t = cum_df[time_col].values
        b = cum_df[term].values
        if term in var_df.columns:
            se = np.sqrt(np.maximum(var_df[term].values, 0.0))
            lower, upper = b - 1.96 * se, b + 1.96 * se
        else:
            lower = upper = b
        plot_ax.step(t, b, where="post", color=cor, linewidth=lw)
        plot_ax.fill_between(t, lower, upper, step="post", color=cor, alpha=0.22)
        plot_ax.axhline(0, color="#888888", linestyle="--", linewidth=0.9)
        plot_ax.set_xlim(0, t_max)
        plot_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        plot_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        plot_ax.tick_params(axis="x", labelbottom=True, bottom=True, labelsize=fs_xtick, pad=2)
        if is_bottom:
            plot_ax.set_xlabel("Tempo (anos)", fontsize=fs_label, labelpad=3)
        if is_left:
            plot_ax.set_ylabel(r"$\hat{B}(t)$", fontsize=fs_label, labelpad=2)
        plot_ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            labelsize=fs_label - 0.5,
            pad=2,
        )
        plot_ax.grid(alpha=0.26, linewidth=0.55)

    fig.canvas.draw()
    for plot_ax in plot_axes:
        for lbl in plot_ax.get_yticklabels():
            lbl.set_visible(True)
            lbl.set_clip_on(False)

    for idx in range(n_terms, nrows * ncols):
        row, col = divmod(idx, ncols)
        inner = outer_gs[row, col].subgridspec(2, 1, height_ratios=[title_band, 1.0])
        for j in range(2):
            ax = fig.add_subplot(inner[j])
            ax.set_visible(False)

    fig.set_size_inches(fw, fh, forward=True)

    path = Path(path)
    pdf_path = path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")
    png_path = pdf_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(pdf_path, format="pdf", pad_inches=0.0, dpi=100)
        fig.savefig(png_path, format="png", dpi=200, pad_inches=0.05)
    plt.close(fig)
    return pdf_path


def plot_comparacao_cox_aalen_thesis(
    compare_covs: list[str],
    cox_hr,
    aalen_null_p: dict[str, float],
    cum_df,
    var_df,
    path: str | Path,
    *,
    height_frac: float = 0.70,
    fig_width_in: float | None = None,
    fig_height_in: float | None = None,
    font_scale: float = 1.0,
    cor: str = "#2E4057",
    cor_prot: str = "#1D9E75",
    cor_risk: str = "#D85A30",
) -> Path:
    """Comparação Cox vs Aalen calibrada para \\textwidth da tese."""
    import textwrap

    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    textheight_in = (29.7 - 2.5 - 2.0) / 2.54
    fw = fig_width_in if fig_width_in is not None else TEXTWIDTH_IN * LATEX_WIDTH_FRAC
    fh = fig_height_in if fig_height_in is not None else textheight_in * height_frac
    n_covs = len(compare_covs)
    fs_title = 7.8 * font_scale
    fs_label = 8.8 * font_scale
    fs_tick = 8.5 * font_scale
    lw = 1.4

    time_col = cum_df.columns[0]
    fig = plt.figure(figsize=(fw, fh))
    outer_gs = GridSpec(
        n_covs,
        2,
        figure=fig,
        left=0.11,
        right=0.99,
        top=0.96,
        bottom=0.10,
        hspace=0.38,
        wspace=0.56,
    )
    inner_hspace = 0.14
    gap_in = 0.07
    wrap_w = max(14, int(30 / max(1.0, font_scale**0.45)))
    line_in = fs_title / 72.0 * 1.18
    stat_line_in = fs_tick / 72.0 * 1.12
    row_h_in = (fh * 0.86) / n_covs
    plot_axes: list = []
    cox_by_cov = cox_hr.set_index("covariate") if "covariate" in cox_hr.columns else cox_hr

    for idx, cov in enumerate(compare_covs):
        is_bottom = idx == n_covs - 1
        label = _aalen_panel_label(cov)
        p_aalen = aalen_null_p.get(cov, float("nan"))
        if cov in cox_by_cov.index:
            row = cox_by_cov.loc[cov]
            hr = float(row["HR"] if "HR" in row else row["exp(coef)"])
            pval = float(row["p_value"] if "p_value" in row else row["p"])
            cox_title = f"Cox HR = {hr:.2f}\np = {pval:.3g}"
        else:
            cox_title = "Não no Cox final"

        label_wrapped = textwrap.fill(label, width=wrap_w, break_long_words=False)
        n_label_lines = label_wrapped.count("\n") + 1
        cox_stat_lines = cox_title.count("\n") + 1
        title_h_in = n_label_lines * line_in + 0.05
        stat_h_in = max(cox_stat_lines, 1) * stat_line_in + 0.04
        plot_h_in = max(row_h_in - title_h_in - stat_h_in - gap_in, row_h_in * 0.52)
        title_band = title_h_in / plot_h_in
        stat_band = stat_h_in / plot_h_in

        panel_specs = (
            (label_wrapped, cox_title),
            ("", f"Aalen p = {p_aalen:.3g}"),
        )

        for col, (panel_title, panel_stat) in enumerate(panel_specs):
            inner = outer_gs[idx, col].subgridspec(
                3,
                1,
                height_ratios=[title_band, stat_band, 1.0],
                hspace=inner_hspace,
            )

            title_ax = fig.add_subplot(inner[0])
            title_ax.set_axis_off()
            title_ax.set_xlim(0, 1)
            title_ax.set_ylim(0, 1)
            if panel_title:
                title_ax.text(
                    0.0,
                    1.0,
                    panel_title,
                    transform=title_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=fs_title,
                    fontweight="bold",
                    linespacing=1.15,
                    clip_on=True,
                )

            stat_ax = fig.add_subplot(inner[1])
            stat_ax.set_axis_off()
            stat_ax.set_xlim(0, 1)
            stat_ax.set_ylim(0, 1)
            if panel_stat:
                stat_ax.text(
                    0.0,
                    1.0,
                    panel_stat,
                    transform=stat_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=fs_tick,
                    fontweight="normal",
                    linespacing=1.12,
                )

            plot_ax = fig.add_subplot(inner[2])
            plot_axes.append(plot_ax)

            if col == 0:
                if cov in cox_by_cov.index:
                    row = cox_by_cov.loc[cov]
                    hr = float(row["HR"] if "HR" in row else row["exp(coef)"])
                    if "exp(coef) lower 95%" in row:
                        hr_low = float(row["exp(coef) lower 95%"])
                        hr_high = float(row["exp(coef) upper 95%"])
                    else:
                        hr_low = float(np.exp(row["CI_lower"]))
                        hr_high = float(np.exp(row["CI_upper"]))
                    color = cor_risk if hr > 1 else cor_prot
                    plot_ax.errorbar(
                        hr,
                        0,
                        xerr=[[hr - hr_low], [hr_high - hr]],
                        fmt="o",
                        color=color,
                        markersize=5.5,
                        capsize=3,
                        capthick=1.1,
                        elinewidth=1.1,
                        zorder=3,
                    )
                    lo = max(0.35, hr_low - 0.10)
                    hi = min(2.0, hr_high + 0.10)
                    lo = min(lo, 0.92)
                    hi = max(hi, 1.08)
                    plot_ax.set_xlim(lo, hi)
                    plot_ax.set_ylim(-0.45, 0.45)
                    plot_ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
                    plot_ax.set_yticks([])
                    plot_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                    plot_ax.tick_params(axis="x", labelbottom=True, labelsize=fs_tick, pad=2)
                    plot_ax.spines["left"].set_visible(False)
                    plot_ax.spines["top"].set_visible(False)
                    plot_ax.spines["right"].set_visible(False)
                    if is_bottom:
                        plot_ax.set_xlabel("Hazard Ratio", fontsize=fs_label, labelpad=4)
                else:
                    plot_ax.text(
                        0.5,
                        0.5,
                        "Não no Cox final",
                        ha="center",
                        va="center",
                        transform=plot_ax.transAxes,
                        fontsize=fs_label,
                    )
                    plot_ax.axis("off")
            elif cov in cum_df.columns:
                t = cum_df[time_col].values
                b = cum_df[cov].values
                if cov in var_df.columns:
                    se = np.sqrt(np.maximum(var_df[cov].values, 0.0))
                    lower, upper = b - 1.96 * se, b + 1.96 * se
                else:
                    lower = upper = b
                plot_ax.step(t, b, where="post", color=cor, linewidth=lw)
                plot_ax.fill_between(t, lower, upper, step="post", color=cor, alpha=0.22)
                y_pad = 0.08 * max(1e-6, float(np.nanmax(np.abs(np.r_[b, lower, upper]))))
                y_lo = float(np.nanmin(lower)) - y_pad
                y_hi = float(np.nanmax(upper)) + y_pad
                y_lo = min(y_lo, -y_pad)
                y_hi = max(y_hi, y_pad)
                plot_ax.set_xlim(0, float(np.nanmax(t)))
                plot_ax.set_ylim(y_lo, y_hi)
                plot_ax.axhline(0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
                plot_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
                plot_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                plot_ax.tick_params(axis="both", labelsize=fs_tick, pad=2)
                if is_bottom:
                    plot_ax.set_xlabel("Tempo (anos)", fontsize=fs_label, labelpad=4)
                plot_ax.set_ylabel(r"$\hat{B}(t)$", fontsize=fs_label, labelpad=4)
                plot_ax.spines["top"].set_visible(False)
                plot_ax.spines["right"].set_visible(False)
                plot_ax.grid(alpha=0.26, linewidth=0.55)
            else:
                plot_ax.text(
                    0.5,
                    0.5,
                    "Não no Aalen",
                    ha="center",
                    va="center",
                    transform=plot_ax.transAxes,
                    fontsize=fs_label,
                )
                plot_ax.axis("off")

    fig.canvas.draw()
    for ax in plot_axes:
        if not ax.lines and not ax.collections:
            continue
        for lbl in ax.get_yticklabels():
            lbl.set_visible(True)
            lbl.set_clip_on(False)

    path = Path(path)
    pdf_path = path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")
    png_path = pdf_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(pdf_path, format="pdf", pad_inches=0.0, dpi=100)
        fig.savefig(png_path, format="png", dpi=200, pad_inches=0.05)
    plt.close(fig)
    return pdf_path


def plot_campo_posicoes_nfl(path: str | Path) -> Path:
    """Diagrama do campo NFL (Fig. 1.1): largura = textwidth LaTeX, legenda em faixa inferior."""
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Bbox

    apply_thesis_rcparams(1.0)

    # Canvas calibrado para \includegraphics[width=\textwidth]:
    # fontes em BODY_FONT_PT ≈ 12 pt no PDF impresso.
    fw = TEXTWIDTH_IN
    field_h = fw / 3.05
    legend_h = 0.62
    fig_h = field_h + legend_h

    fs_player = BODY_FONT_PT - 3.0
    fs_legend = BODY_FONT_PT - 0.5
    fs_ez = BODY_FONT_PT - 1.0
    fs_yard = BODY_FONT_PT - 2.0
    marker_pt = 17.0
    gold = "#f2c94c"
    field = "#3f8f4a"
    ez_left = "#141414"
    ez_right = "#b85a3c"
    defense_face = "#c96a3a"

    fig = plt.figure(figsize=(fw, fig_h), facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[field_h, legend_h], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    leg_ax = fig.add_subplot(gs[1])

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 10, 1, facecolor=ez_left, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((10, 0), 100, 1, facecolor=field, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((110, 0), 10, 1, facecolor=ez_right, edgecolor="none", zorder=0))

    for x in range(10, 111, 5):
        lw = 1.4 if x % 10 == 0 else 0.7
        alpha = 0.95 if x % 10 == 0 else 0.55
        ax.plot([x, x], [0.03, 0.97], color="white", linewidth=lw, alpha=alpha, zorder=1)
    ax.plot([60, 60], [0.03, 0.97], color="white", linewidth=2.0, alpha=0.95, zorder=1)

    for yard, x in ((10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (40, 70), (30, 80), (20, 90), (10, 100)):
        # Numeros de jardas abaixo do campo: os jogadores da base os cobriam dentro do gramado.
        ax.text(x, -0.09, str(yard), ha="center", va="top", color="#333333", fontsize=fs_yard, zorder=2, clip_on=False)

    ax.text(5, 0.5, "ZONA DE\nPONTUAÇÃO", ha="center", va="center", color="white", fontsize=fs_ez, fontweight="bold", rotation=90, zorder=2)
    ax.text(115, 0.5, "ZONA DE\nPONTUAÇÃO", ha="center", va="center", color="white", fontsize=fs_ez, fontweight="bold", rotation=90, zorder=2)

    # Formacao em I, ataque para a direita. Sete jogadores na linha de scrimmage (x = 55): o WR
    # de cima (split end), a linha ofensiva e o TE. O outro WR (flanker) fica um pouco atras da linha
    # e por fora do TE; QB sob o centro, FB e RB em fila atras dele. Defesa 4-3 espelhada.
    # As distancias sao esquematicas: cada marcador ocupa cerca de cinco jardas.
    offense = [
        ("WR", 55.0, 0.94, True),
        ("LT", 55.0, 0.76, False),
        ("LG", 55.0, 0.63, False),
        ("C", 55.0, 0.50, False),
        ("RG", 55.0, 0.37, False),
        ("RT", 55.0, 0.24, False),
        ("TE", 55.0, 0.11, False),
        ("QB", 48.0, 0.50, True),
        ("FB", 41.0, 0.50, False),
        ("RB", 34.0, 0.50, True),
        ("WR", 49.5, 0.06, True),
    ]
    defense = [
        ("DE", 62.0, 0.24),
        ("DT", 62.0, 0.42),
        ("DT", 62.0, 0.58),
        ("DE", 62.0, 0.76),
        ("LB", 70.0, 0.30),
        ("LB", 70.0, 0.50),
        ("LB", 70.0, 0.70),
        ("CB", 62.0, 0.94),
        ("CB", 60.5, 0.06),
        ("SS", 79.0, 0.36),
        ("FS", 84.0, 0.64),
    ]

    def _draw_player(x: float, y: float, label: str, *, studied: bool = False, defense: bool = False) -> None:
        face = defense_face if defense else "#1a1a1a"
        edge = gold if studied else "white"
        ew = 2.6 if studied else 1.1
        ax.scatter(
            [x],
            [y],
            s=marker_pt**2,
            c=face,
            edgecolors=edge,
            linewidths=ew,
            zorder=4,
            clip_on=False,
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=fs_player,
            fontweight="bold",
            zorder=5,
            clip_on=False,
        )

    for label, x, y, studied in offense:
        _draw_player(x, y, label, studied=studied)
    for label, x, y in defense:
        _draw_player(x, y, label, defense=True)

    _pos = leg_ax.get_position()
    leg_ax.set_position([0.02, _pos.y0, 0.96, _pos.height])
    leg_ax.set_xlim(0, 3.3)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")
    # Legenda fora do campo: uma linha horizontal sobre fundo branco, abaixo dos numeros de jardas.
    legend_items = (
        (0.0, True, False, "QB / RB / WR (posições estudadas)"),
        (1.38, False, False, "Demais posições ofensivas"),
        (2.52, False, True, "Posições defensivas"),
    )
    for x0, studied, defense, text in legend_items:
        if defense:
            face, edge, ew = defense_face, "white", 1.0
        elif studied:
            face, edge, ew = "#1a1a1a", gold, 2.0
        else:
            face, edge, ew = "#1a1a1a", "white", 1.0
        leg_ax.scatter([x0 + 0.03], [0.5], s=12**2, c=face, edgecolors=edge, linewidths=ew, zorder=2, clip_on=False)
        leg_ax.text(x0 + 0.09, 0.5, text, ha="left", va="center", color="black", fontsize=fs_legend - 3.0, zorder=2, clip_on=False)

    path = Path(path)
    png_path = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    full_bbox = Bbox.from_bounds(0, 0, fw, fig_h)
    save_kw = dict(facecolor="white", bbox_inches=full_bbox, pad_inches=0)
    fig.savefig(pdf_path, format="pdf", **save_kw)
    fig.savefig(png_path, format="png", dpi=300, **save_kw)
    plt.close(fig)
    return png_path


def save_thesis_figure(fig, path: str | Path, *, also_png: bool = False, pad_inches: float = 0.12) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    if also_png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    return pdf_path
