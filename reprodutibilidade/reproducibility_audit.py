#!/usr/bin/env python3
"""Audit and clean reproducibility artifacts for the TCC pipeline."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import os
from pathlib import Path
from typing import Any
from evidencia_execucao import (
    EvidenceError,
    artifact_snapshot,
    collect_identity,
    collect_documentation_snapshot,
    sha256_file,
    validate_writing_audit_report,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reprodutibilidade" / "REPRODUCIBILITY_MANIFEST.json"
RUN_ALL_PATH = ROOT / "run_all.py"

ARTIFACT_SUFFIXES = {".csv", ".json", ".pkl", ".png", ".rds", ".pdf"}
# .tex entra na lista: os capitulos da tese e os relatorios de estudo sao
# documentos versionados; sem isso, um .tex novo escapava do contrato.
SOURCE_SUFFIXES = {".py", ".r", ".R", ".txt", ".md", ".tex"}
IGNORED_DIRS = {
    ".git",
    ".venv",
    ".venv_rsf",
    "__pycache__",
    "tmp",
}
FORBIDDEN_SOURCE_PATTERNS = (
    "/home/" + "arthur/TCC",
    "C:\\Users\\arthur",
)
FORBIDDEN_RUN_ALL_PATTERNS = (
    "ju" + "pyter",
    "nb" + "convert",
    "ipy" + "kernel",
    "--allow-" + "errors",
)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifesto ausente: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def git_ignored(relatives: list[str]) -> set[str]:
    """Subconjunto de caminhos que o .gitignore exclui (best-effort).

    O manifesto descreve a entrega. O que o proprio repositorio marca como
    rascunho de processo interno (scratch do R, snapshots de auditoria, scripts
    locais) nao entra no contrato de limpeza/recriacao. Se o git nao estiver
    disponivel, nada e ignorado e a auditoria segue puramente estatica.
    """
    if not relatives:
        return set()
    if not (ROOT / ".git").exists():
        return set()
    try:
        proc = subprocess.run(
            # -z: entrada e saida separadas por NUL, sem escape de acentos.
            # Sem isso o git devolve "Predi\303\247\303\243o/..." e a comparacao falha.
            ["git", "-C", str(ROOT), "check-ignore", "-z", "--stdin"],
            input="\0".join(relatives),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise RuntimeError(f"Falha ao consultar exclusoes Git: {exc}") from exc
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore terminou com codigo {proc.returncode}: {proc.stderr}")
    return {line for line in proc.stdout.split("\0") if line}


def iter_repo_files() -> list[Path]:
    files: list[Path] = []
    def onerror(error: OSError) -> None:
        raise RuntimeError(f"Falha na varredura de fontes: {error}") from error

    for base, dirs, names in os.walk(ROOT, onerror=onerror):
        dirs[:] = sorted(name for name in dirs if name not in IGNORED_DIRS)
        for name in sorted(names):
            path = Path(base) / name
            try:
                mode = path.stat().st_mode
            except OSError as exc:
                raise RuntimeError(f"Falha lendo metadados de fonte: {path}: {exc}") from exc
            if not stat.S_ISREG(mode):
                continue
            files.append(path)
    ignored = git_ignored([rel(path) for path in files])
    return [path for path in files if rel(path) not in ignored]


def load_run_all_steps() -> list[dict[str, str]]:
    tree = ast.parse(RUN_ALL_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "STEPS":
            return [{"name": name, "path": path} for name, path in ast.literal_eval(node.value)]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "STEPS":
                    return [{"name": name, "path": path} for name, path in ast.literal_eval(node.value)]
    raise RuntimeError("Nao encontrei STEPS em run_all.py")


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def audit_static(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    entries = manifest.get("files", [])
    by_path = {entry["path"]: entry for entry in entries}
    steps = load_run_all_steps()
    step_names = {step["name"] for step in steps}

    if len(steps) != manifest.get("pipeline", {}).get("step_count"):
        failures.append(
            f"run_all.py tem {len(steps)} passos, manifesto declara "
            f"{manifest.get('pipeline', {}).get('step_count')}"
        )
    declared_steps = manifest.get("pipeline", {}).get("steps")
    if declared_steps is not None and declared_steps != steps:
        failures.append("Manifesto e run_all.py divergem em nomes/ordem/caminhos dos passos")
    for required_source in (
        "run_all.py",
        "reprodutibilidade/evidencia_execucao.py",
        "reprodutibilidade/reproducibility_audit.py",
        "reprodutibilidade/build_tese.py",
    ):
        if by_path.get(required_source, {}).get("category") != "source":
            failures.append(f"Fonte operacional ausente do manifesto: {required_source}")

    for step in steps:
        if not (ROOT / step["path"]).exists():
            failures.append(f"Passo ausente no run_all.py: {step['name']} -> {step['path']}")

    for entry in entries:
        category = entry.get("category")
        path = ROOT / entry["path"]
        if category in {"source", "input", "manual_doc"} and not path.exists():
            failures.append(f"Arquivo {category} ausente: {entry['path']}")
        if category == "generated" and entry.get("producer") not in step_names:
            failures.append(
                f"Artefato gerado sem produtor valido: {entry['path']} -> {entry.get('producer')}"
            )

    for path in iter_repo_files():
        relative = rel(path)
        if path.suffix.lower() in ARTIFACT_SUFFIXES and relative not in by_path:
            failures.append(f"Artefato nao classificado no manifesto: {relative}")
        if path.suffix in SOURCE_SUFFIXES and relative not in by_path:
            failures.append(f"Fonte/documento nao classificado no manifesto: {relative}")

    for entry in entries:
        if entry.get("category") not in {"source", "manual_doc"}:
            continue
        path = ROOT / entry["path"]
        if not path.exists() or path.suffix.lower() not in {".py", ".r"}:
            continue
        text = source_text(path)
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            if pattern in text:
                failures.append(f"Caminho absoluto proibido em fonte: {entry['path']} ({pattern})")

    run_all_text = RUN_ALL_PATH.read_text(encoding="utf-8", errors="ignore")
    for pattern in FORBIDDEN_RUN_ALL_PATTERNS:
        if pattern in run_all_text:
            failures.append(f"run_all.py ainda contem dependencia proibida: {pattern}")

    return failures


def audit_post_run(
    manifest: dict[str, Any],
    allow_skipped: bool = False,
    strict: bool = False,
    release_audit: bool = False,
) -> list[str]:
    failures = audit_static(manifest)
    log_path = ROOT / manifest.get("pipeline", {}).get("execution_log", "EXECUTION_LOG.json")
    expected_steps = load_run_all_steps()
    log: list[dict[str, Any]] = []
    log_bytes = b""
    if not log_path.is_file():
        failures.append(f"Log de execucao ausente: {rel(log_path)}")
    else:
        try:
            log_bytes = log_path.read_bytes()
            parsed = json.loads(log_bytes.decode("utf-8"))
            if not isinstance(parsed, list):
                raise ValueError("o log nao e uma lista")
            log = parsed
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"Log de execucao invalido/ilegivel: {exc}")

    if len(log) != len(expected_steps):
        failures.append(f"Log tem {len(log)} passos, esperado {len(expected_steps)}")
    pulados: set[str] = set()
    for index, (entry, expected) in enumerate(zip(log, expected_steps), 1):
        if not isinstance(entry, dict):
            failures.append(f"Entrada {index} do log nao e objeto")
            continue
        if (entry.get("step"), entry.get("name"), entry.get("path")) != (
            index,
            expected["name"],
            expected["path"],
        ):
            failures.append(f"Passo {index} com nome, ordem ou caminho divergente")
        if entry.get("status") == "SKIPPED":
            pulados.add(entry.get("name"))
        elif entry.get("status") != "OK":
            failures.append(f"Passo {index} tem status nao concluido: {entry.get('status')}")
    if pulados and not allow_skipped:
        failures.append("Log parcial contem passos SKIPPED sem --allow-skipped")

    sem_uso: set[str] = set()
    arquivo_sem_uso = ROOT / "reprodutibilidade" / "figuras_nao_citadas.csv"
    if arquivo_sem_uso.is_file():
        try:
            with arquivo_sem_uso.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if "figura_gerada_nao_citada" not in (reader.fieldnames or []):
                    failures.append("Lista de figuras sem coluna figura_gerada_nao_citada")
                else:
                    sem_uso = {
                        (linha.get("figura_gerada_nao_citada") or "").strip()
                        for linha in reader
                    }
        except (OSError, csv.Error) as exc:
            failures.append(f"Falha ao ler lista de figuras removiveis: {exc}")
    elif strict:
        failures.append("Lista de figuras sem uso ausente")
    filtered: set[str] = set()
    for item in sem_uso:
        if not item:
            continue
        relpath = Path(item)
        if relpath.is_absolute() or ".." in relpath.parts:
            failures.append(f"Lista de figuras inclui caminho proibido: {item}")
            continue
        if "figuras" in relpath.parts:
            filtered.add(relpath.as_posix())
    sem_uso = filtered

    for entry in manifest.get("files", []):
        if entry.get("category") != "generated" or not entry.get("required", True):
            continue
        if allow_skipped and entry.get("producer") in pulados:
            continue
        if entry["path"] in sem_uso:
            continue
        if not (ROOT / entry["path"]).exists():
            failures.append(f"Artefato esperado nao foi recriado: {entry['path']}")

    if strict:
        checkpoint_path = ROOT / "reprodutibilidade" / "EXECUTION_CHECKPOINT.json"
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Checkpoint ausente ou invalido: {exc}")
            return failures
        if not isinstance(checkpoint, dict):
            failures.append("Checkpoint nao e um objeto JSON")
            return failures
        if checkpoint.get("schema_version") != 1:
            failures.append("Schema do checkpoint nao suportado")
        if checkpoint.get("artifact_scan_error") or checkpoint.get("artifact_hashes") is None:
            failures.append("Checkpoint sem varredura de artefatos valida")
        if checkpoint.get("log_sha256") != hashlib.sha256(log_bytes).hexdigest():
            failures.append("Hash do log diverge do checkpoint")
        if checkpoint.get("expected_steps") != expected_steps:
            failures.append("Checkpoint diverge dos nomes/ordem/caminhos em run_all.py")
        if checkpoint.get("completed_step_count") != len(expected_steps):
            failures.append("Checkpoint nao registra todos os passos concluidos")
        if any(
            entry.get("run_id") != checkpoint.get("run_id")
            or entry.get("fingerprint") != checkpoint.get("fingerprint")
            for entry in log
            if isinstance(entry, dict)
        ):
            failures.append("Identidade run_id/fingerprint inconsistente entre log e checkpoint")
        try:
            identity = checkpoint.get("identity")
            if not isinstance(identity, dict):
                raise EvidenceError("identidade ausente")
            current_identity = collect_identity(
                ROOT,
                fast=bool(identity["config"]["fast"]),
                clean=bool(identity["config"]["clean"]),
                skip_post_audit=bool(identity["config"].get("skip_post_audit", False)),
            )
            if current_identity != identity:
                failures.append("Codigo/entradas/configuracao/ambiente computacional mudaram desde o inicio")
            if checkpoint.get("fingerprint") != identity.get("fingerprint"):
                failures.append("Fingerprint nao corresponde a identidade serializada")
            actual_hashes, _signatures = artifact_snapshot(ROOT, manifest)
            if actual_hashes != checkpoint.get("artifact_hashes"):
                failures.append("Artefatos atuais divergem dos hashes do checkpoint")
            producer_step = {step["name"]: step for step in expected_steps}
            entries_by_name = {item.get("name"): item for item in log if isinstance(item, dict)}
            for artifact in manifest.get("files", []):
                if artifact.get("category") != "generated" or not artifact.get("required", True):
                    continue
                producer = artifact.get("producer")
                if allow_skipped and producer in pulados:
                    continue
                relative = artifact["path"]
                if relative in sem_uso:
                    continue
                if relative not in actual_hashes:
                    failures.append(f"Artefato esperado ausente: {relative}")
                    continue
                if producer not in producer_step:
                    failures.append(f"Artefato sem passo produtor: {relative} -> {producer}")
                    continue
                record = entries_by_name.get(producer, {})
                if record.get("status") != "OK" or record.get("produced_artifacts", {}).get(relative) != actual_hashes[relative]:
                    failures.append(f"Sem prova de escrita nesta execucao: {relative} ({producer})")
            incomplete = bool(checkpoint.get("development_incomplete"))
            model_status = checkpoint.get("model_run_status")
            expected_model_status = "INCOMPLETE" if incomplete else "PASS"
            if model_status != expected_model_status:
                failures.append(f"MODEL_RUN_STATUS diverge do registro: {model_status}")
            if not isinstance(checkpoint.get("model_documentation_snapshot"), dict):
                failures.append("Snapshot documental do model run ausente")
            if not incomplete and pulados:
                failures.append("Model run completo contem passos SKIPPED")
            if incomplete and (not allow_skipped or not pulados):
                failures.append("Model run incompleto nao foi autorizado por --allow-skipped")
            if release_audit and incomplete:
                failures.append("Release documental recusado para model run incompleto")
            if release_audit and checkpoint.get("release_audit_status") != "PASS":
                failures.append("RELEASE_AUDIT_STATUS nao esta PASS")
            if release_audit:
                try:
                    current_docs = collect_documentation_snapshot(ROOT)
                    if current_docs != checkpoint.get("release_documentation_snapshot"):
                        failures.append("Documentacao mudou desde o snapshot de release")
                except EvidenceError as exc:
                    failures.append(f"Snapshot documental de release invalido: {exc}")
            writing_audit = checkpoint.get("writing_audit")
            if not isinstance(writing_audit, dict):
                raise EvidenceError("resultado da auditoria de escrita ausente")
            writing = writing_audit.get("status")
            if incomplete:
                if not allow_skipped or not pulados or writing != "NOT_RUN_DEVELOPMENT_INCOMPLETE":
                    failures.append("Estado --fast incompleto nao foi registrado de forma consistente")
            elif release_audit:
                if writing != "PASS":
                    failures.append("Auditoria de release sem aprovacao de escrita --strict")
                try:
                    report_evidence = validate_writing_audit_report(
                        ROOT / "reprodutibilidade" / "AUDITORIA_ESCRITA_VS_PIPELINE.md", None
                    )
                    for key in ("report_sha256", "checked_values", "covered_checks"):
                        if writing_audit.get(key) != report_evidence[key]:
                            failures.append(f"Evidencia de cobertura da escrita diverge: {key}")
                except EvidenceError as exc:
                    failures.append(f"Relatorio escrito sem cobertura/integridade valida: {exc}")
        except (EvidenceError, KeyError, TypeError, OSError) as exc:
            failures.append(f"Falha verificando identidade/artefatos do checkpoint: {exc}")

    return failures


def clean_generated(
    manifest: dict[str, Any], dry_run: bool, preserve_producers: frozenset[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """Apaga artefatos `generated`+`cleanable`.

    `preserve_producers` protege os artefatos de passos que NAO vao rodar nesta
    execucao (o `--fast` pula os estudos pesados). Sem isso, o atalho de iteracao
    apagaria tune, leitura e furos e nao os recriaria, deixando o repositorio sem
    os resultados que a tese cita.
    """
    removed: list[str] = []
    preserved: list[str] = []
    for entry in manifest.get("files", []):
        if entry.get("category") != "generated" or not entry.get("cleanable", False):
            continue
        if entry.get("producer") in preserve_producers:
            preserved.append(entry["path"])
            continue
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Caminho de limpeza fora do escopo: {entry['path']}")
        path = ROOT / relative
        try:
            path.resolve(strict=False).relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Caminho de limpeza fora do repositorio: {entry['path']}") from exc
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"Artefato para limpeza nao e arquivo: {entry['path']}")
        if not dry_run:
            path.unlink()
        removed.append(entry["path"])

    return removed, preserved


def print_result(title: str, failures: list[str]) -> int:
    if failures:
        print(f"{title}: FAIL ({len(failures)} problema(s))")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print(f"{title}: OK")
    return 0


def main() -> int:
    global ROOT, MANIFEST_PATH, RUN_ALL_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT, help="raiz do checkout a auditar")
    parser.add_argument("--static", action="store_true", help="roda auditoria estatica")
    parser.add_argument("--post-run", action="store_true", help="roda auditoria pos-execucao")
    parser.add_argument("--strict", action="store_true", help="exige identidade e evidencia completa")
    parser.add_argument(
        "--release-audit", action="store_true", help="exige gate e snapshot documental de release"
    )
    parser.add_argument("--clean-generated", action="store_true", help="apaga artefatos generated cleanable")
    parser.add_argument("--dry-run", action="store_true", help="mostra limpeza sem apagar")
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="aceita passos SKIPPED (--fast) e nao exige os artefatos deles",
    )
    parser.add_argument(
        "--preserve-producers",
        default="",
        help=(
            "nomes de passos separados por virgula cujos artefatos NAO devem ser "
            "apagados na limpeza (usado pelo --fast para nao destruir o que ele pula)"
        ),
    )
    args = parser.parse_args()

    ROOT = args.repo.resolve(strict=True)
    MANIFEST_PATH = ROOT / "reprodutibilidade" / "REPRODUCIBILITY_MANIFEST.json"
    RUN_ALL_PATH = ROOT / "run_all.py"
    if ROOT != Path(__file__).resolve().parents[1]:
        other_audit = ROOT / "reprodutibilidade" / "reproducibility_audit.py"
        other_evidence = ROOT / "reprodutibilidade" / "evidencia_execucao.py"
        if not other_audit.is_file() or not other_evidence.is_file():
            parser.error("--repo precisa conter os scripts operacionais completos")
        if sha256_file(other_audit) != sha256_file(Path(__file__)) or sha256_file(other_evidence) != sha256_file(Path(__file__).with_name("evidencia_execucao.py")):
            parser.error("--repo usa versao diferente dos scripts operacionais desta execucao")
    if (args.strict or args.release_audit) and not args.post_run:
        parser.error("--strict/--release-audit exigem --post-run")
    if args.release_audit and not args.strict:
        parser.error("--release-audit exige --strict")

    if not any((args.static, args.post_run, args.clean_generated)):
        args.static = True

    manifest = load_manifest()
    preservar = frozenset(
        nome.strip() for nome in args.preserve_producers.split(",") if nome.strip()
    )

    if args.clean_generated:
        removed, preserved = clean_generated(
            manifest, dry_run=args.dry_run, preserve_producers=preservar
        )
        action = "seriam removidos" if args.dry_run else "removidos"
        print(f"Artefatos gerados {action}: {len(removed)}")
        for path in removed[:25]:
            print(f" - {path}")
        if len(removed) > 25:
            print(f" - ... +{len(removed) - 25}")
        if preservar:
            print(f"Artefatos preservados (produtor pulado): {len(preserved)}")

    exit_code = 0
    if args.static:
        exit_code |= print_result("Auditoria estatica", audit_static(manifest))
    if args.post_run:
        exit_code |= print_result(
            "Auditoria pos-execucao",
            audit_post_run(
                manifest,
                allow_skipped=args.allow_skipped,
                strict=args.strict,
                release_audit=args.release_audit,
            ),
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
