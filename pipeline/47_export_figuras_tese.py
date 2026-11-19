#!/usr/bin/env python3
"""Leva as figuras do pipeline para `Escrita Final/figuras/` e **verifica a escrita**.

Duas responsabilidades:

1. copiar para `Escrita Final/figuras/` os PDFs produzidos pelos passos do
   pipeline (EDA, Kaplan-Meier, Cox, Aalen, Cox-Aalen, figuras do Cap. 5);
2. ler os `.tex` COMPILADOS pelo `main.tex` da tese, extrair toda figura
   referenciada e falhar se alguma nao existir na pasta de figuras.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "run_all.py").exists() and (candidate / "DataBase").exists():
            return candidate
    raise RuntimeError("Project root not found")


ROOT = find_project_root()
LIVRO = ROOT / "Escrita Final" / "texto"
DEST = ROOT / "Escrita Final" / "figuras"


def build_copy_map() -> list[tuple[Path, str]]:
    """Origem no pipeline -> nome no destino (mesmo nome do arquivo)."""
    mapping: list[tuple[Path, str]] = []

    def add(dir_rel: str, pattern: str) -> None:
        src_dir = ROOT / dir_rel
        if not src_dir.exists():
            return
        for p in sorted(src_dir.glob(pattern)):
            mapping.append((p, p.name))

    eda = "Análise Exploratória de Dados/figuras"
    km = "Inferência/Modelagem de COX/Curvas/figuras"
    cox = "Inferência/Modelagem de COX/figuras"
    aalen = "Inferência/Modelagem de Aalen/figuras"

    for pos in ("QB", "RB", "WR"):
        add(eda, f"corr_matrix_{pos}.pdf")
        add(eda, f"dist_carreira_{pos}.pdf")
        add(eda, f"draft_{pos}.pdf")
        add(eda, f"games_missed_{pos}.pdf")
        add(eda, f"depth_{pos}.pdf")
        add(km, f"km_global_{pos}.pdf")
        add(km, f"km_draft_{pos}.pdf")
        add(km, f"km_idade_{pos}.pdf")
        add(km, f"km_depth_{pos}.pdf")
        add(cox, f"schoenfeld_{pos}.pdf")
        add(cox, f"forest_plot_{pos}.pdf")
        add(aalen, f"aalen_cum_coefs_{pos}.pdf")

    add(eda, "career_lag_evo_QB.pdf")
    add(aalen, "comparacao_cox_aalen_QB.pdf")
    # Casos da leitura aplicada (passo 62): figura do Capitulo 5.
    add("documentos/estudo_leitura_aplicada/resultados", "tgb_casos_leitura.pdf")
    # Lista 2024 com maiores e menores previsoes por posicao (passo 62).
    add("documentos/estudo_leitura_aplicada/resultados", "tgb_lista_2024.pdf")
    # Lista 2024 por quintil de risco previsto (passo 62).
    add("documentos/estudo_leitura_aplicada/resultados", "tgb_lista_quintis.pdf")
    # Landmarking por quintil de risco (passo 65).
    add("documentos/estudo_leitura_aplicada/resultados", "tgb_landmark_quintis.pdf")
    # Termos do Aalen que violaram proporcionalidade no Cox (passo 62): 2x2 com B(t) e IC 95%.
    add("documentos/estudo_leitura_aplicada/resultados", "aalen_termos_violacao.pdf")
    return mapping


def compiled_tex() -> list[Path]:
    """`main.tex` da tese mais os arquivos que ele inclui."""
    main = LIVRO / "main.tex"
    if not main.exists():
        raise FileNotFoundError(f"Escrita viva ausente: {main}")
    arquivos = [main]
    for nome in re.findall(r"\\include\{([^}]+)\}", main.read_text(encoding="utf-8")):
        alvo = LIVRO / f"{nome}.tex"
        if alvo.exists():
            arquivos.append(alvo)
    return arquivos


def figuras_citadas(arquivos: list[Path]) -> dict[str, set[str]]:
    """Nome do arquivo .pdf -> capitulos que o citam."""
    citadas: dict[str, set[str]] = {}
    for caminho in arquivos:
        texto = caminho.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"([A-Za-z0-9_\-]+\.pdf)", texto):
            citadas.setdefault(match.group(1), set()).add(caminho.name)
    return citadas


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    copiados, ausentes = 0, []
    for src, nome in build_copy_map():
        if not src.exists():
            ausentes.append(str(src.relative_to(ROOT)))
            continue
        shutil.copy2(src, DEST / nome)
        copiados += 1
    print(f"{copiados} figuras do pipeline copiadas para {DEST.relative_to(ROOT)}")
    if ausentes:
        print(f"{len(ausentes)} origem(ns) ausente(s) — rode os passos anteriores:")
        for a in ausentes[:15]:
            print(f"  - {a}")

    # Compositos e PNGs duplicados nao entram na pasta de figuras da tese.
    for old in DEST.glob("*_comparado.*"):
        old.unlink(missing_ok=True)
    for png in DEST.glob("*.png"):
        if png.name == "campo_posicoes_nfl.png":
            continue
        if png.with_suffix(".pdf").exists():
            png.unlink(missing_ok=True)

    # Portao: toda figura citada pela escrita compilada precisa existir.
    citadas = figuras_citadas(compiled_tex())
    faltando = {
        nome: sorted(caps) for nome, caps in citadas.items() if not (DEST / nome).exists()
    }
    print(f"\nFiguras citadas pelos capitulos compilados: {len(citadas)}")
    if faltando:
        print(f"FALTANDO na pasta de figuras: {len(faltando)}")
        for nome, caps in sorted(faltando.items()):
            print(f"  - {nome} (citada em {', '.join(caps)})")
        print("\nA escrita cita figura que o pipeline nao produz.")
        return 1
    print("Todas as figuras citadas existem na pasta da escrita.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
