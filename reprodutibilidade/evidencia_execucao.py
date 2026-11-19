#!/usr/bin/env python3
"""Identidade, checkpoint e controle seguro de processos do pipeline."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


class EvidenceError(RuntimeError):
    pass


class LockHeldError(RuntimeError):
    pass


DOCUMENTATION_SUFFIXES = {".tex", ".bib", ".bst", ".cls", ".sty", ".cfg", ".def"}


def path_com_rscript(rscript: str) -> str:
    """PATH do run_all: a pasta do Rscript vem primeiro, sem repetir se ja estiver la."""
    pasta = str(Path(rscript).parent)
    atual = os.environ.get("PATH", "")
    return atual if atual.split(os.pathsep)[0] == pasta else pasta + os.pathsep + atual


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise EvidenceError(f"Arquivo mudou durante o hash: {path}")
    return digest.hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise EvidenceError(f"Caminho fora do repositorio no manifesto: {relative}")
    path = root / rel
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"Arquivo ausente ou fora do repositorio: {relative}") from exc
    if not path.is_file():
        raise EvidenceError(f"Arquivo-fonte/entrada nao regular: {relative}")
    return path


def declared_hashes(root: Path, entries: list[dict[str, Any]], categories: set[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for entry in entries:
        if entry.get("category") not in categories:
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise EvidenceError("Entrada de manifesto sem caminho valido")
        selected[relative] = sha256_file(_safe_file(root, relative))
    if not selected and categories & {"source", "input"}:
        raise EvidenceError(f"Manifesto sem arquivos nas categorias {sorted(categories)}")
    return dict(sorted(selected.items()))


def collect_documentation_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    try:
        manifest = json.loads(
            _safe_file(root, "reprodutibilidade/REPRODUCIBILITY_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Manifesto ilegivel para snapshot documental: {exc}") from exc
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise EvidenceError("Manifesto sem lista files valida para snapshot documental")
    documents = [
        entry
        for entry in entries
        if entry.get("category") == "manual_doc"
        and Path(entry.get("path", "")).suffix.lower() in DOCUMENTATION_SUFFIXES
    ]
    files = declared_hashes(root, documents, {"manual_doc"})
    if not files:
        raise EvidenceError("Manifesto sem fontes TeX/Bib da tese")
    return {"files": files, "sha256": canonical_sha256(files)}


def python_subprocess_snapshot() -> dict[str, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", "import sys; print(sys.version)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError(f"Subprocesso Python indisponivel: {exc}") from exc
    version = result.stdout.strip()
    if not version or version.splitlines()[0] != sys.version.splitlines()[0]:
        raise EvidenceError("Subprocesso Python nao confirmou o mesmo interpretador/versao")
    return {"executable": str(Path(sys.executable).resolve()), "version": version}


def _rscript() -> str:
    from shutil import which

    found = which("Rscript")
    if found:
        return str(Path(found).resolve())
    if os.name == "nt":
        candidates = [
            Path.home() / "AppData" / "Local" / "Programs" / "R",
            Path("C:/Program Files/R"),
            Path("C:/Program Files (x86)/R"),
        ]
        for base in candidates:
            if base.is_dir():
                matches = sorted(base.glob("R-*/bin/Rscript.exe"), reverse=True)
                matches.extend(sorted(base.glob("R-*/bin/x64/Rscript.exe"), reverse=True))
                if matches:
                    return str(matches[0].resolve())
    raise EvidenceError("Rscript ausente; fingerprint do ambiente nao pode ser calculado")


def environment_snapshot() -> dict[str, Any]:
    python_child = python_subprocess_snapshot()
    packages = sorted(
        [distribution.metadata.get("Name", "").lower(), distribution.version]
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    rscript = _rscript()
    r_expr = (
        'cat(R.version.string, "\\n"); '
        'p <- installed.packages()[, c("Package", "Version"), drop=FALSE]; '
        'p <- p[order(p[, 1]), , drop=FALSE]; '
        'write.table(p, sep="\\t", row.names=FALSE, col.names=FALSE, quote=FALSE)'
    )
    try:
        result = subprocess.run(
            [rscript, "--vanilla", "-e", r_expr],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError(f"Falha ao identificar R/pacotes instalados: {exc}") from exc
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("R version"):
        raise EvidenceError("Rscript nao informou uma versao R verificavel")
    relevant_environment = (
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OMP_DYNAMIC",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DEFAULT_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RCPP_PARALLEL_NUM_THREADS",
        "R_LIBS",
        "R_LIBS_USER",
        "TZ",
        "LANG",
        "LC_ALL",
    )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_subprocess": python_child,
        "platform": platform.platform(),
        "python_packages": packages,
        "rscript": rscript,
        "r_version": lines[0],
        "r_packages": lines[1:],
        "environment_variables": {
            **{
                key: os.environ.get(key, "<unset>")
                for key in relevant_environment
                if key not in ("PATH", "MPLCONFIGDIR")
            },
            "PATH": path_com_rscript(rscript),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0"),
            # run_all define esta pasta so nos subprocessos: o valor padrao nao identifica ambiente
            "MPLCONFIGDIR": "<repo>/tmp/matplotlib" if os.environ.get("MPLCONFIGDIR", "").replace(os.sep, "/").endswith("/tmp/matplotlib") or not os.environ.get("MPLCONFIGDIR") else os.environ["MPLCONFIGDIR"],
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
            "PYTHONUTF8": os.environ.get("PYTHONUTF8", "1"),
        },
    }


def build_identity(
    file_hashes: dict[str, Any], environment: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    code_input_hash = canonical_sha256({"files": file_hashes, "config": config})
    environment_hash = canonical_sha256(environment)
    fingerprint = canonical_sha256(
        {"code_input_sha256": code_input_hash, "environment_sha256": environment_hash}
    )
    return {
        "files": file_hashes,
        "environment": environment,
        "config": config,
        "code_input_sha256": code_input_hash,
        "environment_sha256": environment_hash,
        "fingerprint": fingerprint,
    }


def collect_identity(
    root: Path, fast: bool, clean: bool, skip_post_audit: bool = False
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = root / "reprodutibilidade" / "REPRODUCIBILITY_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Manifesto ilegivel: {exc}") from exc
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise EvidenceError("Manifesto sem lista files valida")
    sources = declared_hashes(root, entries, {"source"})
    inputs = declared_hashes(root, entries, {"input"})
    required = (
        "run_all.py",
        "reprodutibilidade/evidencia_execucao.py",
        "reprodutibilidade/reproducibility_audit.py",
        "reprodutibilidade/build_tese.py",
    )
    explicit = {name: sha256_file(_safe_file(root, name)) for name in required}
    config_files = {
        "manifest": sha256_file(_safe_file(root, "reprodutibilidade/REPRODUCIBILITY_MANIFEST.json")),
        "requirements_R": sha256_file(_safe_file(root, "requirements/requirements_R.txt")),
        "requirements_python": sha256_file(_safe_file(root, "requirements/requirements_avaliacao.txt")),
    }
    environment = environment_snapshot()
    config = {
        "fast": bool(fast),
        "clean": bool(clean),
        "skip_post_audit": bool(skip_post_audit),
        "pythonhashseed": "0",
        "random_seed": 42,
    }
    return build_identity(
        {
            "sources": sources,
            "inputs": inputs,
            "execution_sources": explicit,
            "configuration": config_files,
        },
        environment,
        config,
    )


def artifact_snapshot(
    root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, str], dict[str, tuple[int, int, int]]]:
    hashes: dict[str, str] = {}
    signatures: dict[str, tuple[int, int, int]] = {}
    for entry in manifest.get("files", []):
        if entry.get("category") != "generated":
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise EvidenceError("Artefato gerado sem caminho valido no manifesto")
        path = root / relative
        try:
            stat_before = path.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise EvidenceError(f"Falha acessando artefato {relative}: {exc}") from exc
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"Artefato fora do repositorio: {relative}") from exc
        if not stat.S_ISREG(stat_before.st_mode):
            raise EvidenceError(f"Artefato nao e arquivo regular: {relative}")
        try:
            hashes[relative] = sha256_file(path)
            after = path.stat()
            if (stat_before.st_size, stat_before.st_mtime_ns, stat_before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise EvidenceError(f"Artefato mudou durante a varredura: {relative}")
            signatures[relative] = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        except OSError as exc:
            raise EvidenceError(f"Falha ao varrer artefato {relative}: {exc}") from exc
    return dict(sorted(hashes.items())), signatures


def validate_resume(
    checkpoint: dict[str, Any],
    log_bytes: bytes,
    identity: dict[str, Any],
    expected_steps: list[dict[str, str]],
    actual_artifacts: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    if checkpoint.get("schema_version") != 1:
        raise EvidenceError("Checkpoint ausente ou schema desconhecido; retomada recusada")
    if checkpoint.get("fingerprint") != identity.get("fingerprint"):
        raise EvidenceError("Fingerprint de codigo/entrada/configuracao/ambiente mudou")
    if checkpoint.get("identity") != identity:
        raise EvidenceError("Identidade completa diverge do checkpoint")
    if checkpoint.get("log_sha256") != hashlib.sha256(log_bytes).hexdigest():
        raise EvidenceError("Log ausente, truncado ou alterado desde o checkpoint")
    if checkpoint.get("expected_steps") != expected_steps:
        raise EvidenceError("Nomes/ordem/caminhos dos passos divergem do checkpoint")
    if checkpoint.get("artifact_hashes") != actual_artifacts:
        raise EvidenceError("Artefato ausente, alterado ou corrompido desde o checkpoint")
    try:
        log = json.loads(log_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Log de retomada invalido: {exc}") from exc
    if not isinstance(log, list):
        raise EvidenceError("Log de retomada nao e uma lista")
    completed = checkpoint.get("completed_step_count")
    if not isinstance(completed, int) or completed < 0 or completed > len(expected_steps):
        raise EvidenceError("Contagem de passos concluidos invalida no checkpoint")
    if len(log) not in {completed, completed + 1}:
        raise EvidenceError("Log nao corresponde ao checkpoint passo a passo")
    for index, entry in enumerate(log):
        if not isinstance(entry, dict) or index >= len(expected_steps):
            raise EvidenceError("Log contem passo invalido ou duplicado")
        expected = expected_steps[index]
        if (entry.get("step"), entry.get("name"), entry.get("path")) != (
            index + 1,
            expected["name"],
            expected["path"],
        ):
            raise EvidenceError("Log com passo duplicado, fora de ordem ou caminho divergente")
        if entry.get("run_id") != checkpoint.get("run_id") or entry.get("fingerprint") != identity["fingerprint"]:
            raise EvidenceError("Identidade de execucao diverge entre log e checkpoint")
        allowed = {"OK", "SKIPPED"} if index < completed else {"FAIL"}
        if entry.get("status") not in allowed:
            raise EvidenceError("Status do log diverge da contagem/checkpoint")
    return log, completed


def atomic_write_json(path: Path, value: Any) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    return encoded


def validate_writing_audit_report(
    report_path: Path, previous_signature: tuple[int, int, int] | None
) -> dict[str, Any]:
    try:
        if not report_path.is_file():
            raise EvidenceError("Auditoria --strict nao produziu o relatorio esperado")
        stat = report_path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if previous_signature is not None and signature == previous_signature:
            raise EvidenceError("Relatorio da auditoria nao mudou nesta execucao")
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"Falha lendo relatorio da auditoria: {exc}") from exc
    total_match = re.search(r"Valores conferidos:\s*\*\*(\d+)\*\*", text)
    if not total_match:
        raise EvidenceError("Relatorio da auditoria sem contagem de valores verificavel")
    total = int(total_match.group(1))
    coverage = {
        name: int(count)
        for name, count in re.findall(r"(?m)^\| `([^`]+)` \| (\d+) \|\s*$", text)
    }
    if total <= 0 or not coverage:
        raise EvidenceError("Auditoria de escrita sem cobertura efetiva")
    if any(count <= 0 for count in coverage.values()):
        raise EvidenceError("Uma ou mais checagens da auditoria ficaram sem cobertura")
    if sum(coverage.values()) != total:
        raise EvidenceError("Contagem global diverge da soma da cobertura por checagem")
    try:
        report_hash = sha256_file(report_path)
    except OSError as exc:
        raise EvidenceError(f"Falha calculando hash do relatorio de auditoria: {exc}") from exc
    return {
        "report_sha256": report_hash,
        "checked_values": total,
        "covered_checks": coverage,
    }


class CheckoutLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def acquire(self) -> "CheckoutLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0, os.SEEK_END)
                if self.stream.tell() == 0:
                    self.stream.write(b"\0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise LockHeldError(f"Ja existe pipeline ativo neste checkout: {self.path}") from exc
        return self

    def release(self) -> None:
        if self.stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> "CheckoutLock":
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_process_group(
    command: list[str], cwd: Path, env: dict[str, str], timeout: int, log_path: Path
) -> tuple[bool, float, str]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_group(process)
                elapsed = time.monotonic() - started
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
                return False, elapsed, f"Timeout apos {timeout}s; grupo encerrado. Log: {log_path}\n{tail}"
            except BaseException:
                _terminate_group(process)
                raise
        elapsed = time.monotonic() - started
        if return_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
            return False, elapsed, f"Comando terminou com codigo {return_code}. Log: {log_path}\n{tail}"
        return True, elapsed, ""
    except OSError as exc:
        return False, time.monotonic() - started, str(exc)[:4000]
