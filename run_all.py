#!/usr/bin/env python3
"""Pipeline completo reprodutivel - 48 passos.

Executa, em ordem, os scripts Python/R de `pipeline/`, a partir da base em
`DataBase/`, grava todos os resultados e, ao final, confere os numeros da tese
(`Escrita Final/texto/`) contra os resultados gerados.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from reprodutibilidade.evidencia_execucao import (
    CheckoutLock,
    EvidenceError,
    LockHeldError,
    artifact_snapshot,
    atomic_write_json,
    collect_identity,
    collect_documentation_snapshot,
    run_process_group,
    sha256_file,
    validate_writing_audit_report,
    validate_resume,
)


ROOT = Path(__file__).resolve().parent
REPRO_DIR = ROOT / "reprodutibilidade"
AUDIT_SCRIPT = REPRO_DIR / "reproducibility_audit.py"
MANIFEST_PATH = REPRO_DIR / "REPRODUCIBILITY_MANIFEST.json"
EXECUTION_LOG_PATH = REPRO_DIR / "EXECUTION_LOG.json"
CHECKPOINT_PATH = REPRO_DIR / "EXECUTION_CHECKPOINT.json"
LOCK_PATH = REPRO_DIR / ".run_all.lock"
WRITING_AUDIT_SCRIPT = REPRO_DIR / "audit_escrita_vs_pipeline.py"
WRITING_AUDIT_REPORT = REPRO_DIR / "AUDITORIA_ESCRITA_VS_PIPELINE.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
random.seed(42)

STEPS: list[tuple[str, str]] = [
    ("qualidade_dados", "pipeline/49_qualidade_dados.py"),
    ("prep_cox_QB", "pipeline/04_prep_cox_qb.py"),
    ("prep_cox_RB", "pipeline/05_prep_cox_rb.py"),
    ("prep_cox_WR", "pipeline/06_prep_cox_wr.py"),
    ("eda_QB", "pipeline/01_eda_qb.py"),
    ("eda_RB", "pipeline/02_eda_rb.py"),
    ("eda_WR", "pipeline/03_eda_wr.py"),
    ("estatisticas_eda", "pipeline/50_estatisticas_eda.py"),
    ("series_figuras", "pipeline/52_series_figuras.py"),
    ("add_features", "pipeline/07_add_features.py"),
    # A ordem de execucao nao segue o numero do arquivo (os prep rodam antes
    # das EDA). O X preditivo congelado precisa existir antes dos passos preditivos.
    ("covs_preditivas", "pipeline/38_covs_preditivas.py"),
    ("curvas_KM_QB", "pipeline/08_curvas_km_qb.py"),
    ("curvas_KM_RB", "pipeline/09_curvas_km_rb.py"),
    ("curvas_KM_WR", "pipeline/10_curvas_km_wr.py"),
    ("cox_tdc_QB", "pipeline/11_cox_tdc_qb.py"),
    ("cox_tdc_RB", "pipeline/12_cox_tdc_rb.py"),
    ("cox_tdc_WR", "pipeline/13_cox_tdc_wr.py"),
    ("aalen_tdc_QB", "pipeline/14_aalen_tdc_qb.py"),
    ("aalen_tdc_RB", "pipeline/15_aalen_tdc_rb.py"),
    ("aalen_tdc_WR", "pipeline/16_aalen_tdc_wr.py"),
    ("sensibilidade_inferencial", "pipeline/51_sensibilidade_inferencial.py"),
    ("sensibilidade_selecao", "pipeline/64_sensibilidade_selecao.py"),
    ("aalen_discreto", "pipeline/66_aalen_discreto.py"),
    ("compor_figuras_escrita", "pipeline/35_compor_figuras_escrita.py"),
    # --- Inferencia hibrida, pacote preditivo oficial e figuras da tese ---
    ("cox_aalen_inferencial", "pipeline/36_cox_aalen_inferencial.py"),
    ("tgb_pacote_default", "pipeline/37_tgb_pacote_default.py"),
    # Busca aninhada dos HPs (cada fold usa so o seu treino). O pacote oficial le a
    # tabela `hps_por_fold.csv` que ela grava, por isso precisa vir antes dele.
    ("tune_preditivo", "pipeline/42_tune_preditivo.py"),
    ("tgb_pacote_oficial", "pipeline/39_tgb_pacote_oficial.py"),
    # Sensibilidade: o mesmo pacote oficial com os HPs padrao (sem busca). Saida separada
    # em resultados/sensibilidade_hp_default; nao alimenta a escrita.
    ("sensibilidade_hp_padrao", "pipeline/61_sensibilidade_hp_padrao.py"),
    ("tgb_auditoria", "pipeline/40_tgb_auditoria.py"),
    # IC e teste pareado do ranking preditivo. Le a matriz de sobrevivencia do
    # pacote oficial, entao precisa vir depois dele.
    ("ic_preditivo", "pipeline/53_ic_preditivo.py"),
    # Mapa onde/quando/como do erro e calibracao. Le a matriz de sobrevivencia
    # do pacote oficial e grava a figura citada pela escrita.
    ("diagnostico_erro", "pipeline/55_diagnostico_erro.py"),
    # Lambda do Cox inferencial, janela do X preditivo e saidas em 2023 como censura.
    ("sensibilidade_metodo", "pipeline/58_sensibilidade_metodo.py"),
    # Quanto o desfecho futuro no treino inflaria o C (rotulo com a carreira inteira x em tempo real).
    ("estudo_vazamento", "pipeline/59_estudo_vazamento.py"),
    ("aalen_preditivo", "pipeline/45_aalen_preditivo.py"),
    # Estudos pesados. `leitura_preditiva` precisa vir ANTES das figuras: ele
    # grava `perm_importance_media.csv` e `predicao_2024_modelos.csv`, que
    # `figuras_estudo` e `tgb_figuras` leem. A cadeia e:
    #   leitura_preditiva -> figuras_estudo -> tgb_figuras -> export_figuras_tese
    # Sondagem condicional das arestas da grade. Le `hps_por_fold.csv` do tune,
    # entao precisa vir depois dele.
    ("fronteira_tune", "pipeline/54_fronteira_tune.py"),
    # Sensibilidade do tune: validacao interna de 1 e 3 anos e grade do Cox ate 10.
    ("sensibilidade_tune", "pipeline/57_sensibilidade_tune.py"),
    ("furos_preditivos", "pipeline/44_furos_preditivos.py"),
    ("leitura_preditiva", "pipeline/43_leitura_preditiva.py"),
    # Leitura aplicada: so le os artefatos de leitura_preditiva e dos modelos finais
    # (casos, tipologia do erro, concordancia, inferencia x importancia, DP).
    ("leitura_aplicada", "pipeline/62_leitura_aplicada.py"),
    # Refaz o landmarking do modelo principal guardando a previsao por jogador.
    ("landmark_jogadores", "pipeline/65_landmark_jogadores.py"),
    # Testes de robustez: referencia so com experiencia, forma funcional, sementes etc.
    ("testes_defesa", "pipeline/67_testes_defesa.py"),
    # Variacoes recentes de trajetoria acrescentadas aos dois ramos.
    ("variacoes_recentes", "pipeline/68_variacoes_recentes.py"),
    ("figuras_estudo", "pipeline/46_figuras_estudo.py"),
    ("tgb_figuras", "pipeline/41_tgb_figuras.py"),
    ("campo_posicoes_nfl", "pipeline/48_campo_posicoes_nfl.py"),
    # Figura do walk-forward (C por origem, posicao e modelo). Le os resumos do pacote
    # oficial e o IC do passo ic_preditivo, e grava direto em Escrita Final/figuras.
    # Vem antes do export, que falha se a escrita citar figura ausente.
    ("figura_walkforward", "pipeline/63_figura_walkforward.py"),
    # Ultimo: leva as figuras para a pasta da escrita e FALHA se a escrita citar
    # alguma figura que o pipeline nao produziu.
    ("export_figuras_tese", "pipeline/47_export_figuras_tese.py"),
]

# Estudos pesados (horas de maquina). Pulados por `--fast`, que existe para
# iterar durante o desenvolvimento; a execucao padrao roda tudo.
HEAVY_STEPS: frozenset[str] = frozenset(
    {
        "sensibilidade_tune",
        "leitura_preditiva",
        "furos_preditivos",
        # Refaz o ajuste do Cox em varios limiares, cenarios e imputacoes: e
        # pesado em tempo, ainda que nao em memoria.
        "sensibilidade_inferencial",
        # Repete a selecao do Cox em 50 reamostras de jogadores por posicao.
        "sensibilidade_selecao",
    }
)

# Passos longos que nao podem ser pulados no `--fast` (o pacote oficial le os HPs da
# busca aninhada), mas precisam do limite de tempo dos pesados.
PASSOS_LONGOS: frozenset[str] = frozenset({"tune_preditivo"})

# Passos que dependem dos pesados e por isso sao pulados junto com eles. Sem
# isto, `--fast` quebraria a cadeia das figuras (e o portao final), porque
# `figuras_estudo` e `tgb_figuras` leem saidas de `leitura_preditiva`.
DEPENDEM_DE_PESADOS: frozenset[str] = frozenset(
    {"figuras_estudo", "tgb_figuras", "export_figuras_tese"}
)
PULADOS_EM_FAST: frozenset[str] = HEAVY_STEPS | DEPENDEM_DE_PESADOS

STEP_DEPENDENCIES: dict[str, list[str]] = {
    "cox_tdc_QB": ["Inferência/shared/run_schoenfeld_diagnostics.R"],
    "cox_tdc_RB": ["Inferência/shared/run_schoenfeld_diagnostics.R"],
    "cox_tdc_WR": ["Inferência/shared/run_schoenfeld_diagnostics.R"],
    "aalen_tdc_QB": [
        "Inferência/Modelagem de Aalen/shared/run_aalen_univariate_screening.R",
        "Inferência/Modelagem de Aalen/shared/run_aalen_tied_inference.R",
    ],
    "aalen_tdc_RB": [
        "Inferência/Modelagem de Aalen/shared/run_aalen_univariate_screening.R",
        "Inferência/Modelagem de Aalen/shared/run_aalen_tied_inference.R",
    ],
    "aalen_tdc_WR": [
        "Inferência/Modelagem de Aalen/shared/run_aalen_univariate_screening.R",
        "Inferência/Modelagem de Aalen/shared/run_aalen_tied_inference.R",
    ],
    "tgb_pacote_oficial": ["documentos/estudo_tune_preditivo/resultados/hps_por_fold.csv"],
}


def find_rscript() -> str | None:
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = [
        Path.home() / "AppData" / "Local" / "Programs" / "R",
        Path("C:/Program Files/R"),
        Path("C:/Program Files (x86)/R"),
    ]
    for base in candidates:
        if not base.exists():
            continue
        matches = sorted(base.glob("R-*/bin/Rscript.exe"), reverse=True)
        matches.extend(sorted(base.glob("R-*/bin/x64/Rscript.exe"), reverse=True))
        for match in matches:
            if match.exists():
                return str(match)
    return None


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    rscript = find_rscript()
    if rscript:
        pasta = str(Path(rscript).parent)
        caminhos = env.get("PATH", "").split(os.pathsep)
        if not caminhos or caminhos[0] != pasta:
            env["PATH"] = pasta + os.pathsep + env.get("PATH", "")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_command(command: list[str], timeout: int) -> tuple[bool, float, str]:
    log_dir = ROOT / "tmp" / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(command[-1]).stem if command else "command")
    log_path = log_dir / f"{time.time_ns()}_{label}.log"
    return run_process_group(command, ROOT, command_env(), timeout, log_path)


def run_audit(args: list[str]) -> tuple[bool, str]:
    success, _elapsed, error = run_command([sys.executable, str(AUDIT_SCRIPT), *args], timeout=300)
    return success, error


# Timeout por passo. Os estudos pesados (tune aninhado, permutacao, ablacao) levam horas.
STEP_TIMEOUT_DEFAULT = 1800
STEP_TIMEOUT_HEAVY = 6 * 60 * 60


def run_step(script_path: Path, timeout: int = STEP_TIMEOUT_DEFAULT) -> tuple[bool, float, str]:
    return run_command([sys.executable, str(script_path)], timeout=timeout)


# --------------------------------------------------------- figuras sem uso
# O passo `series_figuras` calcula, a partir do manifesto, quais figuras geradas
# nenhum `.tex` do repositorio cita. Essas figuras sao ruido: ficam no repositorio
# sem que ninguem as leia. A varredura abaixo as apaga DEPOIS de todos os passos
# (o `export_figuras_tese` ja copiou o que a escrita usa) e so dentro de diretorios
# `figuras/`, para que nenhuma remocao possa alcancar um artefato de resultado.
FIGURAS_SEM_USO = REPRO_DIR / "figuras_nao_citadas.csv"


def limpar_figuras_sem_uso() -> int:
    """Apaga as figuras geradas que nenhum documento cita. Devolve quantas saiu."""
    if not FIGURAS_SEM_USO.exists():
        raise EvidenceError(f"Relatorio de figuras ausente: {FIGURAS_SEM_USO.relative_to(ROOT)}")

    removidas = 0
    with FIGURAS_SEM_USO.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "figura_gerada_nao_citada" not in (reader.fieldnames or []):
            raise EvidenceError("Relatorio de figuras sem coluna figura_gerada_nao_citada")
        for linha in reader:
            caminho = (linha.get("figura_gerada_nao_citada") or "").strip()
            if not caminho:
                continue
            rel = Path(caminho)
            if rel.is_absolute() or ".." in rel.parts:
                raise EvidenceError(f"Caminho de figura fora do escopo seguro: {caminho}")
            alvo = ROOT / rel
            if "figuras" not in rel.parts:
                raise EvidenceError(f"Caminho listado nao e figura: {caminho}")
            try:
                alvo.resolve(strict=False).relative_to(ROOT.resolve())
            except ValueError as exc:
                raise EvidenceError(f"Figura fora do repositorio: {caminho}") from exc
            if alvo.exists():
                if not alvo.is_file():
                    raise EvidenceError(f"Caminho de figura nao e arquivo: {caminho}")
                alvo.unlink()
                removidas += 1
    return removidas


def load_manifest() -> dict:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Manifesto ausente ou invalido: {exc}") from exc
    if not isinstance(manifest.get("files"), list):
        raise EvidenceError("Manifesto sem files valido")
    return manifest


def expected_steps() -> list[dict[str, str]]:
    return [{"name": name, "path": path} for name, path in STEPS]


def ensure_generated_dirs(manifest: dict) -> None:
    for entry in manifest["files"]:
        if entry.get("category") != "generated":
            continue
        relative = Path(entry.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError(f"Caminho gerado fora do escopo: {entry.get('path')}")
        target = ROOT / relative
        try:
            target.parent.resolve(strict=False).relative_to(ROOT.resolve())
        except ValueError as exc:
            raise EvidenceError(f"Diretorio gerado fora do repositorio: {relative}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)


def state_checkpoint(
    log: list[dict],
    identity: dict,
    run_id: str,
    manifest: dict,
    completed: int,
    attempts: list[dict],
    writing_audit: dict,
    started_at: str,
    artifact_error: str = "",
    model_documentation_snapshot: dict | None = None,
    model_run_status: str = "IN_PROGRESS",
    release_audit_status: str = "NOT_RUN",
) -> None:
    log_bytes = atomic_write_json(EXECUTION_LOG_PATH, log)
    artifacts = None
    scan_error = artifact_error
    if not scan_error:
        try:
            artifacts, _signatures = artifact_snapshot(ROOT, manifest)
        except EvidenceError as exc:
            scan_error = str(exc)
    checkpoint = {
        "schema_version": 1,
        "run_id": run_id,
        "fingerprint": identity["fingerprint"],
        "identity": identity,
        "expected_steps": expected_steps(),
        "completed_step_count": completed,
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "artifact_hashes": artifacts,
        "artifact_scan_error": scan_error,
        "attempts": attempts,
        "writing_audit": writing_audit,
        "model_documentation_snapshot": model_documentation_snapshot,
        "model_run_status": model_run_status,
        "release_audit_status": release_audit_status,
        "development_incomplete": bool(identity["config"]["fast"]),
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(CHECKPOINT_PATH, checkpoint)


def release_audit_locked(manifest: dict) -> int:
    checkpoint: dict = {}
    try:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if checkpoint.get("model_run_status") != "PASS":
            raise EvidenceError("Model run nao esta PASS; release nao executa nem promove modelos")
        identity = checkpoint["identity"]
        current_identity = collect_identity(
            ROOT,
            fast=bool(identity["config"]["fast"]),
            clean=bool(identity["config"]["clean"]),
            skip_post_audit=bool(identity["config"].get("skip_post_audit", False)),
        )
        if current_identity != identity:
            raise EvidenceError("Codigo/entradas/ambiente computacional mudaram; release recusado")
        current_hashes, _signatures = artifact_snapshot(ROOT, manifest)
        if current_hashes != checkpoint.get("artifact_hashes"):
            raise EvidenceError("Artefatos do model run mudaram; release recusado")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, EvidenceError) as exc:
        if checkpoint.get("model_run_status") == "PASS":
            checkpoint["release_audit_status"] = "FAIL"
            checkpoint["release_audit_error"] = str(exc)
            atomic_write_json(CHECKPOINT_PATH, checkpoint)
        print(f"FAIL release preflight: {exc}", file=sys.stderr)
        return 1
    model_ok, model_error = run_audit(["--post-run", "--strict"])
    if not model_ok:
        checkpoint["release_audit_status"] = "FAIL"
        checkpoint["release_audit_error"] = model_error
        atomic_write_json(CHECKPOINT_PATH, checkpoint)
        print(f"MODEL_RUN_STATUS nao validado; release nao iniciado:\n{model_error}", file=sys.stderr)
        return 1

    try:
        docs_before = collect_documentation_snapshot(ROOT)
    except EvidenceError as exc:
        checkpoint["release_audit_status"] = "FAIL"
        checkpoint["release_audit_error"] = str(exc)
        atomic_write_json(CHECKPOINT_PATH, checkpoint)
        print(f"MODEL_RUN_STATUS=PASS; RELEASE_AUDIT_STATUS=FAIL: {exc}", file=sys.stderr)
        return 1
    checkpoint["release_audit_status"] = "RUNNING"
    checkpoint.pop("release_audit_error", None)
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    try:
        prior_stat = WRITING_AUDIT_REPORT.stat()
        prior_signature = (prior_stat.st_size, prior_stat.st_mtime_ns, prior_stat.st_ctime_ns)
    except FileNotFoundError:
        prior_signature = None
    audit_ok, _audit_elapsed, audit_error = run_command(
        [sys.executable, str(WRITING_AUDIT_SCRIPT), "--strict"], timeout=300
    )
    coverage = {}
    if audit_ok:
        try:
            coverage = validate_writing_audit_report(WRITING_AUDIT_REPORT, prior_signature)
        except EvidenceError as exc:
            audit_ok, audit_error = False, str(exc)
    try:
        docs_after = collect_documentation_snapshot(ROOT)
    except EvidenceError as exc:
        docs_after = {}
        audit_ok, audit_error = False, str(exc)
    if docs_after != docs_before:
        audit_ok, audit_error = False, "Fontes TeX/Bib mudaram durante a auditoria de release"
    checkpoint["writing_audit"] = {
        "status": "PASS" if audit_ok else "FAIL",
        "error": audit_error,
        "auditor_sha256": sha256_file(WRITING_AUDIT_SCRIPT),
        **coverage,
    }
    checkpoint["release_documentation_snapshot"] = docs_after
    checkpoint["release_audit_status"] = "PASS" if audit_ok else "FAIL"
    if not audit_ok:
        checkpoint["release_audit_error"] = audit_error
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    if not audit_ok:
        print(
            "MODEL_RUN_STATUS=PASS; RELEASE_AUDIT_STATUS=FAIL: " + audit_error
            + ". Concilie a prosa e reexecute --release-audit.",
            file=sys.stderr,
        )
        return 1

    audit_ok, audit_error = run_audit(["--post-run", "--strict", "--release-audit"])
    if not audit_ok:
        checkpoint["release_audit_status"] = "FAIL"
        checkpoint["release_audit_error"] = audit_error
        atomic_write_json(CHECKPOINT_PATH, checkpoint)
        print(f"MODEL_RUN_STATUS=PASS; RELEASE_AUDIT_STATUS=FAIL:\n{audit_error}", file=sys.stderr)
        return 1
    print("MODEL_RUN_STATUS=PASS; RELEASE_AUDIT_STATUS=PASS")
    return 0


def check_external_dependencies() -> list[str]:
    missing: list[str] = []
    if find_rscript() is None:
        missing.append(
            "Rscript nao encontrado no PATH; instale R 4.x e os pacotes de "
            "requirements/requirements_R.txt"
        )
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT, help="raiz do checkout a executar")
    parser.add_argument("--no-clean", action="store_true", help="nao limpa artefatos gerados antes da execucao")
    parser.add_argument(
        "--skip-post-audit",
        action="store_true",
        help="registrada na identidade da execucao; as verificacoes finais continuam obrigatorias",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="pula os estudos pesados (tune, leitura, furos); use so para iterar",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continua do primeiro passo pendente reaproveitando o que ja esta OK no "
            "EXECUTION_LOG.json; nao limpa artefatos (limpar apagaria o que se reaproveita)"
        ),
    )
    parser.add_argument(
        "--release-audit",
        action="store_true",
        help="revalida o model run concluido e atualiza apenas o gate documental",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        configure_root(args.repo)
        source_root = Path(__file__).resolve().parent
        if sha256_file(ROOT / "run_all.py") != sha256_file(Path(__file__)):
            raise EvidenceError("--repo contem run_all.py diferente do entrypoint executado")
        if sha256_file(ROOT / "reprodutibilidade" / "evidencia_execucao.py") != sha256_file(
            source_root / "reprodutibilidade" / "evidencia_execucao.py"
        ):
            raise EvidenceError("--repo contem evidencia_execucao.py diferente do modulo executado")
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    lock = CheckoutLock(LOCK_PATH)
    try:
        lock.acquire()
    except LockHeldError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt("SIGTERM recebido")

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        return run_locked(args)
    except KeyboardInterrupt:
        print("Interrupcao recebida; processos do passo encerrados e checkpoint preservado.", file=sys.stderr)
        return 130
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"FAIL: evidencia/preflight: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        lock.release()


def configure_root(root: Path) -> None:
    global ROOT, REPRO_DIR, AUDIT_SCRIPT, MANIFEST_PATH, EXECUTION_LOG_PATH
    global CHECKPOINT_PATH, LOCK_PATH, WRITING_AUDIT_SCRIPT, WRITING_AUDIT_REPORT, FIGURAS_SEM_USO
    ROOT = root.resolve(strict=True)
    REPRO_DIR = ROOT / "reprodutibilidade"
    AUDIT_SCRIPT = REPRO_DIR / "reproducibility_audit.py"
    MANIFEST_PATH = REPRO_DIR / "REPRODUCIBILITY_MANIFEST.json"
    EXECUTION_LOG_PATH = REPRO_DIR / "EXECUTION_LOG.json"
    CHECKPOINT_PATH = REPRO_DIR / "EXECUTION_CHECKPOINT.json"
    LOCK_PATH = REPRO_DIR / ".run_all.lock"
    WRITING_AUDIT_SCRIPT = REPRO_DIR / "audit_escrita_vs_pipeline.py"
    WRITING_AUDIT_REPORT = REPRO_DIR / "AUDITORIA_ESCRITA_VS_PIPELINE.md"
    FIGURAS_SEM_USO = REPRO_DIR / "figuras_nao_citadas.csv"


def run_locked(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if args.release_audit:
        if args.resume or args.fast or args.no_clean:
            raise EvidenceError("--release-audit nao combina com --resume, --fast ou --no-clean")
        return release_audit_locked(manifest)
    prior_checkpoint: dict | None = None
    prior_log: list[dict] = []
    completed = 0
    attempts: list[dict] = []
    clean_policy = not args.no_clean and not args.resume
    model_documentation_snapshot = collect_documentation_snapshot(ROOT)
    if args.resume:
        try:
            prior_checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            saved_config = prior_checkpoint["identity"]["config"]
            if prior_checkpoint.get("model_run_status") == "PASS":
                raise EvidenceError("Model run ja concluiu; use --release-audit, nao --resume")
            if bool(saved_config.get("fast")) != bool(args.fast):
                raise EvidenceError("--fast precisa coincidir com a execucao retomada")
            if bool(saved_config.get("skip_post_audit", False)) != bool(args.skip_post_audit):
                raise EvidenceError("--skip-post-audit precisa coincidir com a execucao retomada")
            clean_policy = bool(saved_config["clean"])
            if prior_checkpoint.get("model_documentation_snapshot") != model_documentation_snapshot:
                raise EvidenceError("Fontes TeX/Bib mudaram; retomada do model run recusada")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"Checkpoint de retomada ausente/invalido: {exc}") from exc

    identity = collect_identity(
        ROOT, fast=args.fast, clean=clean_policy, skip_post_audit=args.skip_post_audit
    )
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    writing_audit = {"status": "PENDING"}
    if args.resume:
        if prior_checkpoint is None:
            raise EvidenceError("Checkpoint de retomada ausente")
        try:
            log_bytes = EXECUTION_LOG_PATH.read_bytes()
            current_artifacts, _signatures = artifact_snapshot(ROOT, manifest)
        except OSError as exc:
            raise EvidenceError(f"Log/artefatos de retomada ausentes: {exc}") from exc
        prior_log, completed = validate_resume(
            prior_checkpoint, log_bytes, identity, expected_steps(), current_artifacts
        )
        run_id = prior_checkpoint["run_id"]
        started_at = prior_checkpoint.get("started_at", started_at)
        attempts = list(prior_checkpoint.get("attempts", []))
        if len(prior_log) > completed:
            attempts.append(prior_log[-1])
        prior_log = prior_log[:completed]

    if args.skip_post_audit:
        print("--skip-post-audit mantido por compatibilidade; gates finais obrigatorios continuam ativos.")
    if args.fast:
        print("--fast: modo de desenvolvimento; esta execucao nao recebera aprovacao final.")

    audit_ok, audit_error = run_audit(["--static"])
    if not audit_ok:
        raise EvidenceError(f"Auditoria estatica pre-execucao falhou:\n{audit_error}")
    missing_dependencies = check_external_dependencies()
    if missing_dependencies:
        raise EvidenceError("Dependencias externas ausentes: " + "; ".join(missing_dependencies))

    pendentes = [step for step in STEPS if not (args.fast and step[0] in PULADOS_EM_FAST)]
    pulados = {name for name, _path in STEPS} - {name for name, _path in pendentes}
    if clean_policy and not args.resume:
        clean_args = ["--clean-generated"]
        if pulados:
            clean_args += ["--preserve-producers", ",".join(sorted(pulados))]
        clean_ok, clean_error = run_audit(clean_args)
        if not clean_ok:
            raise EvidenceError(f"Limpeza de artefatos gerados falhou:\n{clean_error}")
    ensure_generated_dirs(manifest)

    log = list(prior_log)
    if not args.resume:
        state_checkpoint(
            log, identity, run_id, manifest, 0, attempts, writing_audit, started_at,
            model_documentation_snapshot=model_documentation_snapshot,
        )

    ok = sum(item.get("status") in {"OK", "SKIPPED"} for item in log)
    for index, (name, path) in enumerate(STEPS[completed:], completed + 1):
        dependencies = STEP_DEPENDENCIES.get(name, [])
        if name in pulados:
            entry = {
                "step": index,
                "name": name,
                "path": path,
                "dependencies": dependencies,
                "run_id": run_id,
                "fingerprint": identity["fingerprint"],
                "status": "SKIPPED",
                "produced_artifacts": {},
                "time": 0,
                "error": "Desenvolvimento incompleto (--fast)",
            }
            log.append(entry)
            completed = index
            state_checkpoint(
                log, identity, run_id, manifest, completed, attempts, writing_audit, started_at,
                model_documentation_snapshot=model_documentation_snapshot,
            )
            continue

        full = ROOT / path
        before_hashes, before_signatures = artifact_snapshot(ROOT, manifest)
        timeout = STEP_TIMEOUT_HEAVY if name in HEAVY_STEPS | PASSOS_LONGOS else STEP_TIMEOUT_DEFAULT
        started = time.monotonic()
        if not full.is_file():
            success, elapsed, error = False, 0.0, f"arquivo de passo ausente: {path}"
            after_hashes, after_signatures = before_hashes, before_signatures
        else:
            try:
                success, elapsed, error = run_step(full, timeout)
            except KeyboardInterrupt:
                success, elapsed = False, time.monotonic() - started
                error = "Interrupcao recebida; grupo de processos encerrado."
            try:
                after_hashes, after_signatures = artifact_snapshot(ROOT, manifest)
            except EvidenceError as exc:
                success, error = False, f"Varredura de evidencia falhou: {exc}"
                after_hashes, after_signatures = {}, {}

        produced = {
            artifact: digest
            for artifact, digest in after_hashes.items()
            if artifact not in before_hashes
            or before_signatures.get(artifact) != after_signatures.get(artifact)
        }
        status = "OK" if success else "FAIL"
        entry = {
            "step": index,
            "name": name,
            "path": path,
            "dependencies": dependencies,
            "run_id": run_id,
            "fingerprint": identity["fingerprint"],
            "status": status,
            "produced_artifacts": produced,
            "time": round(elapsed, 1),
            "timeout_s": timeout,
            "error": error,
        }
        log.append(entry)
        if success:
            completed = index
            ok += 1
        else:
            attempts.append(entry)
        state_checkpoint(
            log,
            identity,
            run_id,
            manifest,
            completed,
            attempts,
            writing_audit,
            started_at,
            artifact_error=error if "Varredura de evidencia falhou" in error else "",
            model_documentation_snapshot=model_documentation_snapshot,
        )
        if not success:
            print(f"FAIL passo {index}/{len(STEPS)} {name}: {error}", file=sys.stderr)
            return 1

    try:
        removed = limpar_figuras_sem_uso()
    except (OSError, EvidenceError, csv.Error) as exc:
        writing_audit = {"status": "FAIL", "error": f"limpeza de figuras sem uso: {exc}"}
        state_checkpoint(
            log, identity, run_id, manifest, completed, attempts, writing_audit, started_at,
            model_documentation_snapshot=model_documentation_snapshot,
        )
        print(writing_audit["error"], file=sys.stderr)
        return 1
    if removed:
        print(f"Figuras geradas sem uso removidas: {removed}")
    state_checkpoint(
        log, identity, run_id, manifest, completed, attempts, writing_audit, started_at,
        model_documentation_snapshot=model_documentation_snapshot,
    )

    try:
        current_identity = collect_identity(
            ROOT, fast=args.fast, clean=clean_policy, skip_post_audit=args.skip_post_audit
        )
        current_docs = collect_documentation_snapshot(ROOT)
    except EvidenceError as exc:
        source_error = str(exc)
    else:
        source_error = ""
        if current_identity != identity:
            source_error = "Fontes/entradas/ambiente computacional mudaram durante a execucao"
        elif current_docs != model_documentation_snapshot:
            source_error = "Fontes TeX/Bib mudaram durante o model run"
    if source_error:
        writing_audit = {"status": "FAIL", "error": source_error}
        state_checkpoint(
            log, identity, run_id, manifest, completed, attempts, writing_audit, started_at,
            model_documentation_snapshot=model_documentation_snapshot, model_run_status="FAIL",
        )
        print(source_error, file=sys.stderr)
        return 1

    if args.fast:
        writing_audit = {"status": "NOT_RUN_DEVELOPMENT_INCOMPLETE"}
        model_status = "INCOMPLETE"
    else:
        writing_audit = {"status": "PENDING_RELEASE"}
        model_status = "PASS"
    state_checkpoint(
        log, identity, run_id, manifest, completed, attempts, writing_audit, started_at,
        model_documentation_snapshot=model_documentation_snapshot,
        model_run_status=model_status,
    )
    if args.fast:
        audit_ok, audit_error = run_audit(["--post-run", "--strict", "--allow-skipped"])
        if not audit_ok:
            print(f"Auditoria do model run falhou:\n{audit_error}", file=sys.stderr)
            return 1
        print(f"MODEL_RUN_STATUS=INCOMPLETE ({ok}/{len(STEPS)}); RELEASE_AUDIT_STATUS=NOT_RUN")
        return 0
    return release_audit_locked(manifest)


if __name__ == "__main__":
    sys.exit(main())
