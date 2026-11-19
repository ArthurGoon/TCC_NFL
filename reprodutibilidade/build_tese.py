#!/usr/bin/env python3
"""Compila a tese em uma pasta isolada, sem tocar no PDF versionado."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path("Escrita Final") / "texto"
FIGURES_DIR = Path("Escrita Final") / "figuras"


class BuildError(RuntimeError):
    pass


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def prepare_build(repo: Path, build_dir: Path | None) -> tuple[Path, Path]:
    repo = repo.resolve(strict=True)
    source = repo / SOURCE_DIR
    figures = repo / FIGURES_DIR
    if not source.is_dir() or not figures.is_dir() or not _under(source, repo) or not _under(figures, repo):
        raise BuildError("Fontes ou diretorio de figuras da tese ausentes")
    if build_dir is None:
        destination = Path(tempfile.mkdtemp(prefix="tcc_nfl_tese_"))
    else:
        destination = build_dir.expanduser().resolve(strict=False)
        if _under(destination, repo):
            raise BuildError("--build-dir deve ficar fora do checkout para manter o build isolado")
        destination.mkdir(parents=True, exist_ok=False)
    if _under(destination, repo):
        raise BuildError("O diretorio de build nao pode ficar dentro do checkout")
    for item in source.iterdir():
        if item.name == "figuras":
            if not item.is_dir() or item.resolve() != figures.resolve():
                raise BuildError("A pasta figuras da tese diverge das figuras oficiais")
            continue
        if item.is_symlink() and not _under(item, source):
            raise BuildError(f"Fonte aponta para fora do diretorio da tese: {item.name}")
        if item.is_file() and item.suffix.lower() in {".tex", ".bib", ".bst", ".cls", ".sty", ".cfg", ".def"}:
            shutil.copy2(item, destination / item.name)
    if not (destination / "main.tex").is_file():
        raise BuildError("main.tex nao foi copiado para o build")
    for item in figures.rglob("*"):
        if item.is_symlink() and not _under(item, figures):
            raise BuildError(f"Figura aponta para fora da pasta de figuras: {item}")
    shutil.copytree(figures, destination / "figuras", symlinks=False)
    if not any(item.is_file() for item in (destination / "figuras").rglob("*")):
        raise BuildError("Diretorio de figuras copiado esta vazio")
    return repo, destination


def page_count(pdf_path: Path) -> int:
    try:
        import fitz

        with fitz.open(pdf_path) as document:
            return document.page_count
    except ImportError:
        pdfinfo = shutil.which("pdfinfo")
        if not pdfinfo:
            raise BuildError("PyMuPDF ou pdfinfo necessario para contar paginas")
        result = subprocess.run([pdfinfo, str(pdf_path)], check=True, capture_output=True, text=True, timeout=30)
        match = re.search(r"(?m)^Pages:\s+(\d+)", result.stdout)
        if not match:
            raise BuildError("pdfinfo nao retornou contagem de paginas")
        return int(match.group(1))


def run_build(repo: Path, build_dir: Path | None) -> dict[str, Any]:
    repo, work = prepare_build(repo, build_dir)
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "build_dir": str(work),
        "commands": [],
        "status": "RUNNING",
    }
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        report.update(status="FAIL", error="pdflatex e bibtex precisam estar no PATH")
        (work / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise BuildError(report["error"])

    commands = [
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"],
        [bibtex, "main"],
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"],
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"],
    ]
    try:
        for command in commands:
            if "bibtex" in Path(command[0]).name.lower():
                label = "bibtex"
            else:
                latex_pass = sum(
                    Path(item["command"][0]).name == Path(pdflatex).name
                    for item in report["commands"]
                ) + 1
                label = f"pdflatex_pass{latex_pass}"
            log_path = work / f"{label}.log.txt"
            try:
                with log_path.open("w", encoding="utf-8", errors="replace") as log_stream:
                    completed = subprocess.run(
                        command,
                        cwd=work,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=900,
                    )
            except subprocess.TimeoutExpired as exc:
                report["commands"].append({"command": command, "status": "TIMEOUT", "log": str(log_path)})
                raise BuildError(f"Timeout no comando: {command[0]}") from exc
            report["commands"].append(
                {"command": command, "status": completed.returncode, "log": str(log_path)}
            )
            if completed.returncode != 0:
                raise BuildError(f"Comando terminou com codigo {completed.returncode}: {command[0]}")
            if label.startswith("pdflatex_pass") and latex_pass >= 3:
                current_log = (work / "main.log").read_text(encoding="utf-8", errors="replace")
                if "Rerun to get" in current_log or "Label(s) may have changed" in current_log:
                    if latex_pass >= 6:
                        raise BuildError("Referencias nao estabilizaram apos seis passagens LaTeX")
                    commands.append(command.copy())

        pdf_path = work / "main.pdf"
        log_path = work / "main.log"
        if not pdf_path.is_file() or not log_path.is_file():
            raise BuildError("Compilacao terminou sem main.pdf ou main.log")
        latex_log = log_path.read_text(encoding="utf-8", errors="replace")
        errors = re.findall(r"(?m)^! .+$|Emergency stop|Fatal error occurred", latex_log, flags=re.IGNORECASE)
        undefined = re.findall(r"(?im)^.*(?:Citation|Reference).*undefined.*$", latex_log)
        overfull = re.findall(r"(?im)^.*Overfull \\[hv]box.*$", latex_log)
        pages = page_count(pdf_path)
        report.update(
            status="PASS" if not errors and not undefined else "FAIL",
            pdf=str(pdf_path),
            pages=pages,
            latex_errors=errors,
            undefined_references=undefined,
            cross_references_stable=True,
            overfull_count=len(overfull),
            overfull=overfull,
        )
        if errors or undefined:
            raise BuildError(f"LaTeX reportou {len(errors)} erro(s) e {len(undefined)} referencia(s) indefinida(s)")
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        if isinstance(exc, BuildError):
            raise
        raise BuildError(str(exc)) from exc
    finally:
        (work / "BUILD_REPORT.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_ROOT, help="raiz do checkout")
    parser.add_argument("--build-dir", type=Path, help="diretorio novo, fora do checkout, para os arquivos compilados")
    args = parser.parse_args()
    try:
        report = run_build(args.repo, args.build_dir)
    except (BuildError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"BUILD FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"BUILD {report['status']}: {report['pdf']} | paginas={report['pages']} "
        f"| erros={len(report['latex_errors'])} | indefinidas={len(report['undefined_references'])} "
        f"| overfull={report['overfull_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
