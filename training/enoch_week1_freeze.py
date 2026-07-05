#!/usr/bin/env python3
"""Materialize and verify the immutable Week-1 W1.0 control bundle.

The production policies come from a clean ``git archive`` of the frozen
reference.  The authoritative Week-1 evaluator is built from an immutable
snapshot of the current evaluator/mechanics sources because the strict,
feature-isolated evaluator necessarily post-dates the production reference.
Every copied input and executable is SHA-256 bound into the control manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

try:
    from training import enoch_week1
except ImportError:  # Direct ``python training/enoch_week1_freeze.py`` execution.
    import enoch_week1  # type: ignore[no-redef]


DEFAULT_REFERENCE = "c813c8a"
CONTROL_PROBE_RELATIVE = Path("core/examples/enoch_control_probe.rs")
CONTROL_PROBE_BUNDLE_SOURCE = Path("source/enoch-control-probe.rs")
REFERENCE_PROBE_PATCH_RELATIVE = Path("training/enoch_control_probe_reference.patch")
REFERENCE_PROBE_PATCH_BUNDLE_SOURCE = Path(
    "source/enoch-control-probe-reference.patch"
)
REFERENCE_PROBE_PATCH_ARTIFACT_ID = "source/enoch-control-probe-reference-patch"
REFERENCE_PROBE_PREPATCH_PREREQUISITES = (
    Path("bin/enoch-0"),
    Path("bin/expert-0"),
    Path("bin/grandmaster-0"),
    Path("bin/reference-evaluator"),
    Path("bin/validate-expert-model"),
    Path("preflight/expert-model-validation.json"),
    Path("preflight/reference-model-contract-tests.json"),
)
CONTROL_PROBE_REGRESSION_SEED = 0x58B76B703FF0E148
CONTROL_PROBE_DETERMINISM_REPETITIONS = 8
CONTROL_PROBE_ARTIFACT_IDS = frozenset(
    {
        "binary/enoch-control-probe-current",
        "binary/enoch-control-probe-reference",
        "preflight/control-probe-determinism",
        "source/enoch-control-probe",
        REFERENCE_PROBE_PATCH_ARTIFACT_ID,
    }
)
REFERENCE_EXAMPLES = (
    "enoch_benchmark",
    "gm_benchmark",
    "model_control_eval",
    "paired_eval",
    "validate_expert_model",
)
REFERENCE_MODEL_FILES = (
    "expert_model.onnx",
    "expert_model.onnx.golden.json",
    "expert_model.onnx.manifest.json",
    "expert_model.promotion.json",
    "expert_model.training.manifest.json",
)
CURRENT_SOURCE_ROOTS = ("core/src", "mechanics/src")
CURRENT_SOURCE_FILES = (
    "Cargo.lock",
    "Cargo.toml",
    "core/Cargo.toml",
    CONTROL_PROBE_RELATIVE.as_posix(),
    "core/examples/enoch_eval.rs",
    "mechanics/Cargo.toml",
    REFERENCE_PROBE_PATCH_RELATIVE.as_posix(),
)
SAFE_ENV_NAMES = (
    "CARGO_HOME",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "RUSTUP_HOME",
    "TMPDIR",
)


class FreezeError(RuntimeError):
    """Raised when a control bundle cannot be proven complete."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise FreezeError(
            f"command failed ({completed.returncode}): {rendered}\n{completed.stdout}"
        )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copy_hashed(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    shutil.copymode(source, destination)
    return _sha256_file(destination)


def _safe_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base is None else base
    cleaned, _ = enoch_week1.sanitized_evaluator_environment(source)
    environment = {
        name: cleaned[name]
        for name in SAFE_ENV_NAMES
        if name in cleaned and cleaned[name]
    }
    if "PATH" not in environment:
        raise FreezeError("PATH is required to build the control bundle")
    environment.update(
        {
            "CARGO_INCREMENTAL": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "RUST_BACKTRACE": "1",
        }
    )
    return environment


def _canonical_source_paths(workspace: Path) -> list[Path]:
    paths: set[Path] = set()
    for relative in CURRENT_SOURCE_ROOTS:
        root = workspace / relative
        paths.update(path for path in root.rglob("*.rs") if path.is_file())
    for relative in CURRENT_SOURCE_FILES:
        path = workspace / relative
        if not path.is_file():
            raise FreezeError(f"required evaluator source is missing: {relative}")
        paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(workspace).as_posix())


def _snapshot_sources(workspace: Path, destination: Path) -> tuple[str, list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in _canonical_source_paths(workspace):
            relative = path.relative_to(workspace).as_posix()
            digest = _sha256_file(path)
            records.append({"path": relative, "sha256": digest})
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return enoch_week1.canonical_json_sha256(records), records


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        destination_root = destination.resolve()
        for member in members:
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise FreezeError(f"unsafe path in git archive: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise FreezeError(f"unsafe member type in git archive: {member.name!r}")
        if "filter" in inspect.signature(archive.extractall).parameters:
            archive.extractall(destination, members=members, filter="data")
        else:
            # Python 3.11 and older lack extraction filters. The fresh-root
            # containment and regular-file/directory checks above provide the
            # equivalent safety needed for a trusted local ``git archive``;
            # clamp mode bits before using the legacy extractor as well.
            for member in members:
                member.mode &= 0o755
            archive.extractall(destination, members=members)


def _inject_control_probe(workspace: Path, reference_source: Path, bundle: Path) -> str:
    """Bind one exact probe source and inject it into the clean reference tree."""

    source = workspace / CONTROL_PROBE_RELATIVE
    if not source.is_file():
        raise FreezeError(f"required control probe source is missing: {source}")
    frozen_source = bundle / CONTROL_PROBE_BUNDLE_SOURCE
    source_sha256 = _copy_hashed(source, frozen_source)
    injected_sha256 = _copy_hashed(
        frozen_source, reference_source / CONTROL_PROBE_RELATIVE
    )
    if injected_sha256 != source_sha256:
        raise FreezeError("injected reference control probe differs from frozen source")
    return source_sha256


def _reference_probe_patch_paths(patch: Path) -> set[str]:
    try:
        lines = patch.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FreezeError(f"could not read reference probe patch {patch}: {exc}") from exc
    paths: set[str] = set()
    for line in lines:
        if line.startswith("diff --git "):
            fields = line.split()
            if (
                len(fields) != 4
                or not fields[2].startswith("a/")
                or not fields[3].startswith("b/")
            ):
                raise FreezeError("reference probe patch has a malformed diff header")
            paths.update((fields[2][2:], fields[3][2:]))
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            paths.add(line[6:])
    if not paths:
        raise FreezeError("reference probe patch contains no repository paths")
    return paths


def _bind_reference_probe_patch(workspace: Path, bundle: Path) -> tuple[Path, str]:
    """Freeze the one compatibility patch allowed in the reference probe build."""

    missing = [
        path.as_posix()
        for path in REFERENCE_PROBE_PREPATCH_PREREQUISITES
        if not (bundle / path).is_file()
    ]
    if missing:
        raise FreezeError(
            "reference probe patch cannot be bound before pristine binaries and "
            f"model-contract tests are staged: {missing}"
        )
    source = workspace / REFERENCE_PROBE_PATCH_RELATIVE
    if not source.is_file():
        raise FreezeError(f"required reference probe patch is missing: {source}")
    expected_paths = {
        "core/src/bot/harness.rs",
        "core/src/bot/heuristics.rs",
        "core/src/bot/policy.rs",
    }
    if _reference_probe_patch_paths(source) != expected_paths:
        raise FreezeError(
            "reference probe patch must touch exactly the audited harness.rs, "
            "heuristics.rs, and policy.rs"
        )
    frozen = bundle / REFERENCE_PROBE_PATCH_BUNDLE_SOURCE
    digest = _copy_hashed(source, frozen)
    if _reference_probe_patch_paths(frozen) != expected_paths:
        raise FreezeError("frozen reference probe patch changed while being copied")
    return frozen, digest


def _apply_reference_probe_patch(
    reference_source: Path,
    patch: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Apply the bound determinism normalization to the probe-only source tree."""

    targets = {
        relative: reference_source / relative
        for relative in (
            "core/src/bot/harness.rs",
            "core/src/bot/heuristics.rs",
            "core/src/bot/policy.rs",
        )
    }
    missing = [relative for relative, path in targets.items() if not path.is_file()]
    if missing:
        raise FreezeError(f"reference source lacks patch targets: {missing}")
    before = {relative: _sha256_file(path) for relative, path in targets.items()}
    _run(
        ("git", "apply", "--check", "--whitespace=error-all", str(patch)),
        cwd=reference_source,
        environment=environment,
    )
    _run(
        ("git", "apply", "--whitespace=error-all", str(patch)),
        cwd=reference_source,
        environment=environment,
    )
    _run(
        ("git", "apply", "--check", "--reverse", str(patch)),
        cwd=reference_source,
        environment=environment,
    )
    after = {relative: _sha256_file(path) for relative, path in targets.items()}
    unchanged = [relative for relative in targets if after[relative] == before[relative]]
    if unchanged:
        raise FreezeError(f"reference probe patch did not change targets: {unchanged}")
    return {
        "patch_sha256": _sha256_file(patch),
        "target_files_after_sha256": after,
        "target_files_before_sha256": before,
    }


def _configuration_hash(value: Mapping[str, Any]) -> str:
    return enoch_week1.canonical_json_sha256(value)


def _hardware_summary() -> str:
    # Deliberately excludes hostname, serial number, and user identity.
    cpu_count = os.cpu_count() or 0
    machine = platform.machine() or "unknown"
    processor = platform.processor() or "unknown"
    memory = "unknown"
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and physical_pages > 0:
            memory = str(page_size * physical_pages)
    except (OSError, TypeError, ValueError):
        pass
    if sys.platform == "darwin":
        if memory == "unknown":
            try:
                memory = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                pass
    return f"machine={machine};processor={processor};logical_cpus={cpu_count};memory_bytes={memory}"


def _operating_system() -> str:
    return f"system={platform.system()};release={platform.release()};version={platform.version()};machine={platform.machine()}"


def _build_reference(
    reference_source: Path,
    target: Path,
    environment: Mapping[str, str],
) -> dict[str, str]:
    build_environment = dict(environment)
    build_environment["CARGO_TARGET_DIR"] = str(target)
    command = ["cargo", "build", "--locked", "--release", "-p", "shengji-core"]
    for example in REFERENCE_EXAMPLES:
        command.extend(("--example", example))
    _run(command, cwd=reference_source, environment=build_environment)
    return {
        example: str(target / "release" / "examples" / example)
        for example in REFERENCE_EXAMPLES
    }


def _build_current_evaluator(
    workspace: Path,
    target: Path,
    environment: Mapping[str, str],
    source_hash: str,
) -> tuple[Path, str]:
    build_environment = dict(environment)
    build_environment["CARGO_TARGET_DIR"] = str(target)
    build_environment["SHENGJI_SOURCE_SHA"] = source_hash
    _run(
        (
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "shengji-core",
            "--example",
            "enoch_eval",
        ),
        cwd=workspace,
        environment=build_environment,
    )
    strict_test_output = _run(
        (
            "cargo",
            "test",
            "--locked",
            "-p",
            "shengji-core",
            "bot::search::tests::strict_search_rejects_a_zero_sample_prior_fallback",
            "--",
            "--exact",
        ),
        cwd=workspace,
        environment=build_environment,
    )
    if (
        "test bot::search::tests::strict_search_rejects_a_zero_sample_prior_fallback ... ok"
        not in strict_test_output
    ):
        raise FreezeError("strict evaluator preflight did not execute its required test")
    return target / "release" / "examples" / "enoch_eval", strict_test_output


def _build_control_probe(
    source: Path,
    target: Path,
    environment: Mapping[str, str],
) -> Path:
    build_environment = dict(environment)
    build_environment["CARGO_TARGET_DIR"] = str(target)
    _run(
        (
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "shengji-core",
            "--example",
            "enoch_control_probe",
        ),
        cwd=source,
        environment=build_environment,
    )
    return target / "release" / "examples" / "enoch_control_probe"


def _control_probe_determinism_record(
    reference_probe: Path,
    current_probe: Path,
    environment: Mapping[str, str],
    *,
    repetitions: int = CONTROL_PROBE_DETERMINISM_REPETITIONS,
) -> dict[str, Any]:
    """Require fresh-process determinism within and across both probe builds."""

    if repetitions < 2:
        raise FreezeError("control probe determinism requires at least two repetitions")
    outputs: dict[str, list[str]] = {"reference": [], "current": []}
    for label, binary in (("reference", reference_probe), ("current", current_probe)):
        command = (
            str(binary),
            "--policy",
            "enoch-greedy",
            "--seed",
            str(CONTROL_PROBE_REGRESSION_SEED),
        )
        for _ in range(repetitions):
            outputs[label].append(
                _run(command, cwd=binary.parent, environment=environment)
            )
        if any(output != outputs[label][0] for output in outputs[label][1:]):
            raise FreezeError(f"{label} control probe is not process-deterministic")
    if outputs["reference"][0] != outputs["current"][0]:
        raise FreezeError("reference and current control probes diverge on regression seed")
    stdout = outputs["reference"][0].encode("utf-8")
    return {
        "command_succeeded": True,
        "policy": "enoch-greedy",
        "repetitions_per_binary": repetitions,
        "seed": CONTROL_PROBE_REGRESSION_SEED,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
    }


def freeze_bundle(
    workspace: Path,
    protocol_path: Path,
    output: Path,
    reference: str,
) -> Path:
    protocol = enoch_week1.load_json_object(protocol_path)
    enoch_week1.validate_protocol(protocol)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable control bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = _safe_environment()

    reference_commit = _run(
        ("git", "rev-parse", f"{reference}^{{commit}}"),
        cwd=workspace,
        environment=environment,
    )
    if not reference_commit or any(char not in "0123456789abcdef" for char in reference_commit):
        raise FreezeError("reference did not resolve to a canonical commit SHA")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        reference_archive = temporary / "source" / "production-reference.tar"
        reference_archive.parent.mkdir(parents=True)
        _run(
            (
                "git",
                "archive",
                "--format=tar",
                f"--output={reference_archive}",
                reference_commit,
            ),
            cwd=workspace,
            environment=environment,
        )
        reference_source_sha256 = _sha256_file(reference_archive)
        reference_source = temporary / "build" / "reference-source"
        _extract_archive(reference_archive, reference_source)
        control_probe_source_sha256 = _inject_control_probe(
            workspace, reference_source, temporary
        )

        # Build every production artifact from untouched c813, then immediately
        # copy it out of Cargo's target directory. No compatibility patch exists
        # in the reference tree yet, and no later build can replace these files.
        reference_target = temporary / "build" / "reference-target"
        reference_bins = _build_reference(
            reference_source, reference_target, environment
        )
        binary_destinations = {
            "enoch-0": "enoch_benchmark",
            "expert-0": "model_control_eval",
            "grandmaster-0": "gm_benchmark",
            "reference-evaluator": "paired_eval",
        }
        binary_hashes: dict[str, str] = {}
        staged_reference_artifacts: dict[str, str] = {}
        staged_reference_paths: dict[str, Path] = {}
        for policy, example in binary_destinations.items():
            artifact_id = f"binary/{policy}"
            destination = temporary / "bin" / policy
            digest = _copy_hashed(Path(reference_bins[example]), destination)
            binary_hashes[policy] = digest
            staged_reference_artifacts[artifact_id] = digest
            staged_reference_paths[artifact_id] = destination
        staged_validator = temporary / "bin" / "validate-expert-model"
        validator_binary_hash = _copy_hashed(
            Path(reference_bins["validate_expert_model"]), staged_validator
        )
        staged_reference_artifacts[
            "binary/validate-expert-model"
        ] = validator_binary_hash
        staged_reference_paths["binary/validate-expert-model"] = staged_validator

        # Production model-contract tests also execute before patching c813.
        reference_test_environment = dict(environment)
        reference_test_environment["CARGO_TARGET_DIR"] = str(reference_target)
        reference_test_output = _run(
            (
                "cargo",
                "test",
                "--locked",
                "-p",
                "shengji-core",
                "bot::expert::model_path_tests::",
                "--",
                "--test-threads=1",
            ),
            cwd=reference_source,
            environment=reference_test_environment,
        )
        for required_test in (
            "model_path_override_round_trips",
            "manifest_rejects_width_drift_and_untyped_v2_outputs",
            "embedded_model_has_no_value_output",
        ):
            marker = f"test bot::expert::model_path_tests::{required_test} ... ok"
            if marker not in reference_test_output:
                raise FreezeError(
                    f"production model-contract preflight did not execute {required_test}"
                )

        reference_model_dir = reference_source / "core" / "src" / "bot"
        validator_output = _run(
            (
                str(staged_validator),
                str(reference_model_dir / "expert_model.onnx"),
                str(reference_model_dir / "expert_model.onnx.manifest.json"),
                str(reference_model_dir / "expert_model.onnx.golden.json"),
            ),
            cwd=reference_source,
            environment=environment,
        )
        enoch_week1.atomic_write_json(
            temporary / "preflight" / "reference-model-contract-tests.json",
            {"command_succeeded": True, "output": reference_test_output},
        )
        enoch_week1.atomic_write_json(
            temporary / "preflight" / "expert-model-validation.json",
            {"command_succeeded": True, "output": validator_output},
        )

        model_hashes: dict[str, str] = {}
        for filename in REFERENCE_MODEL_FILES:
            source = reference_model_dir / filename
            if not source.is_file():
                raise FreezeError(f"production reference lacks model artifact {filename}")
            model_hashes[filename] = _copy_hashed(
                source, temporary / "models" / filename
            )

        # Only after all untouched production binaries are safely staged and all
        # production tests have passed may the probe-only compatibility patch be
        # bound and applied. Its target directory is separate as a second guard
        # against Cargo replacing an original production artifact.
        reference_probe_patch, reference_probe_patch_sha256 = (
            _bind_reference_probe_patch(workspace, temporary)
        )
        patch_application = _apply_reference_probe_patch(
            reference_source, reference_probe_patch, environment
        )
        reference_probe_bin = _build_control_probe(
            reference_source,
            temporary / "build" / "reference-probe-target",
            environment,
        )
        for artifact_id, expected in staged_reference_artifacts.items():
            if _sha256_file(staged_reference_paths[artifact_id]) != expected:
                raise FreezeError(
                    f"staged production binary changed after probe patch: {artifact_id}"
                )

        evaluator_archive = temporary / "source" / "week1-evaluator-source.tar.gz"
        evaluator_source_sha256, source_records = _snapshot_sources(
            workspace, evaluator_archive
        )
        source_records_by_path = {
            record["path"]: record["sha256"] for record in source_records
        }
        if (
            source_records_by_path.get(CONTROL_PROBE_RELATIVE.as_posix())
            != control_probe_source_sha256
        ):
            raise FreezeError("control probe source changed while it was being injected")
        if (
            source_records_by_path.get(REFERENCE_PROBE_PATCH_RELATIVE.as_posix())
            != reference_probe_patch_sha256
        ):
            raise FreezeError("reference probe patch changed while it was being bound")
        evaluator_file_list = temporary / "source" / "week1-evaluator-source-files.json"
        enoch_week1.atomic_write_json(evaluator_file_list, source_records)
        evaluator_source_identity_sha256 = _sha256_file(evaluator_file_list)

        evaluator_bin, strict_test_output = _build_current_evaluator(
            workspace,
            temporary / "build" / "evaluator-target",
            environment,
            evaluator_source_identity_sha256,
        )
        current_probe_bin = _build_control_probe(
            workspace,
            temporary / "build" / "evaluator-target",
            environment,
        )
        post_build_archive = temporary / "build" / "post-build-source-check.tar.gz"
        post_build_source_sha256, post_build_records = _snapshot_sources(
            workspace, post_build_archive
        )
        if (
            post_build_source_sha256 != evaluator_source_sha256
            or post_build_records != source_records
        ):
            raise FreezeError("evaluator source changed while the frozen binary was building")
        if (
            _sha256_file(reference_source / CONTROL_PROBE_RELATIVE)
            != control_probe_source_sha256
        ):
            raise FreezeError("injected reference control probe changed during the build")
        if _sha256_file(reference_probe_patch) != reference_probe_patch_sha256:
            raise FreezeError("frozen reference probe patch changed during the build")
        enoch_week1.atomic_write_json(
            temporary / "preflight" / "strict-evaluator-test.json",
            {"command_succeeded": True, "output": strict_test_output},
        )

        reference_probe_destination = (
            temporary / "bin" / "enoch-control-probe-reference"
        )
        current_probe_destination = temporary / "bin" / "enoch-control-probe-current"
        reference_probe_binary_hash = _copy_hashed(
            reference_probe_bin, reference_probe_destination
        )
        current_probe_binary_hash = _copy_hashed(
            current_probe_bin, current_probe_destination
        )
        determinism_record = _control_probe_determinism_record(
            reference_probe_destination, current_probe_destination, environment
        )
        determinism_record["reference_patch_application"] = patch_application
        enoch_week1.atomic_write_json(
            temporary / "preflight" / "control-probe-determinism.json",
            determinism_record,
        )

        frozen_probe_source_sha256 = _sha256_file(
            temporary / CONTROL_PROBE_BUNDLE_SOURCE
        )
        if frozen_probe_source_sha256 != control_probe_source_sha256:
            raise FreezeError("frozen control probe source changed during the build")
        artifact_hashes: dict[str, str] = {
            "preflight/expert-model-validation": _sha256_file(
                temporary / "preflight" / "expert-model-validation.json"
            ),
            "preflight/control-probe-determinism": _sha256_file(
                temporary / "preflight" / "control-probe-determinism.json"
            ),
            "preflight/reference-model-contract-tests": _sha256_file(
                temporary / "preflight" / "reference-model-contract-tests.json"
            ),
            "preflight/strict-evaluator-test": _sha256_file(
                temporary / "preflight" / "strict-evaluator-test.json"
            ),
            "protocol/week1-seed-protocol": _copy_hashed(
                protocol_path, temporary / "protocol" / "week1-seed-protocol.json"
            ),
            "source/enoch-control-probe": frozen_probe_source_sha256,
            REFERENCE_PROBE_PATCH_ARTIFACT_ID: reference_probe_patch_sha256,
            "source/production-reference": reference_source_sha256,
            "source/week1-evaluator": _sha256_file(evaluator_archive),
            "source/week1-evaluator-file-list": _sha256_file(
                temporary / "source" / "week1-evaluator-source-files.json"
            ),
            "binary/enoch-control-probe-current": current_probe_binary_hash,
            "binary/enoch-control-probe-reference": reference_probe_binary_hash,
        }
        artifact_hashes.update(staged_reference_artifacts)
        evaluator_binary_hash = _copy_hashed(
            evaluator_bin, temporary / "bin" / "enoch-week1-evaluator"
        )
        artifact_hashes["binary/week1-evaluator"] = evaluator_binary_hash

        for filename, digest in model_hashes.items():
            artifact_hashes[f"model/{filename}"] = digest
        embedded_model_hash = model_hashes["expert_model.onnx"]
        no_model_hash = enoch_week1.canonical_json_sha256(
            {"model": "none", "reason": "heuristic-policy-tier"}
        )

        search_knobs = {
            "adaptive_budget": False,
            "belief_model": None,
            "enoch-0": {
                "budget_ms": 2200,
                "max_candidates": 6,
                "max_worlds": 144,
                "policy": "EnochHeuristic",
                "rollout_policy": "EnochHeuristic",
                "rollout_tricks": 12,
            },
            "expert-0": {
                "budget_ms": 2200,
                "max_candidates": 6,
                "max_worlds": 144,
                "policy": "Net",
                "rollout_policy": "Heuristic",
                "rollout_tricks": 12,
            },
            "grandmaster-0": {
                "budget_multiplier": 3.0,
                "max_candidates": 8,
                "max_worlds": 400,
                "policy": "EnochHeuristic",
                "rollout_policy": "Heuristic",
                "rollout_tricks": "full-hand",
            },
            "persistent_belief": False,
            "puct": False,
            "risk_adjustment": False,
            "runtime_model_overrides": None,
        }
        policy_configs = {
            name: _configuration_hash(search_knobs[name])
            for name in ("enoch-0", "expert-0", "grandmaster-0")
        }
        policies = {
            "enoch-0": enoch_week1.build_frozen_policy_identity(
                source_sha256=reference_source_sha256,
                binary_sha256=binary_hashes["enoch-0"],
                model_sha256=no_model_hash,
                configuration_sha256=policy_configs["enoch-0"],
            ),
            "expert-0": enoch_week1.build_frozen_policy_identity(
                source_sha256=reference_source_sha256,
                binary_sha256=binary_hashes["expert-0"],
                model_sha256=embedded_model_hash,
                configuration_sha256=policy_configs["expert-0"],
            ),
            "grandmaster-0": enoch_week1.build_frozen_policy_identity(
                source_sha256=reference_source_sha256,
                binary_sha256=binary_hashes["grandmaster-0"],
                model_sha256=no_model_hash,
                configuration_sha256=policy_configs["grandmaster-0"],
            ),
        }
        evaluator_config = enoch_week1.WEEK1_EVALUATOR_CONTRACT
        evaluator_identity = enoch_week1.build_frozen_evaluator_identity(
            source_sha256=evaluator_source_identity_sha256,
            binary_sha256=evaluator_binary_hash,
            configuration_sha256=_configuration_hash(evaluator_config),
        )
        rustc_version = _run(("rustc", "-Vv"), cwd=workspace, environment=environment)
        cargo_version = _run(("cargo", "-V"), cwd=workspace, environment=environment)
        manifest = enoch_week1.build_w1_0_control_manifest(
            protocol,
            production_reference=reference_commit,
            policy_identities=policies,
            evaluator_identity=evaluator_identity,
            artifact_hashes=artifact_hashes,
            compiler=f"{rustc_version}\n{cargo_version}",
            operating_system=_operating_system(),
            hardware_summary=_hardware_summary(),
            search_knobs=search_knobs,
            effective_environment={
                name: environment[name]
                for name in sorted(environment)
                if name not in {"HOME", "PATH", "CARGO_HOME", "RUSTUP_HOME", "TMPDIR"}
            },
            replay_commands=(
                "bin/enoch-control-probe-reference --policy enoch-greedy --seed 0",
                "bin/enoch-control-probe-current --policy enoch-greedy --seed 0",
                "bin/enoch-week1-evaluator --help",
                "bin/validate-expert-model models/expert_model.onnx models/expert_model.onnx.manifest.json models/expert_model.onnx.golden.json",
            ),
            model_selection_contract={
                "fallback_disabled": True,
                "intended_model_loaded": True,
                "policy_selection_independent": True,
                "q_selection_independent": True,
                "value_selection_independent": True,
            },
            model_selection_evidence=enoch_week1.MODEL_SELECTION_EVIDENCE_IDS,
        )
        enoch_week1.atomic_write_json(temporary / "control-manifest.json", manifest)
        enoch_week1.atomic_write_json(
            temporary / "bundle-index.json",
            {
                "artifact_hashes": artifact_hashes,
                "control_manifest_fingerprint": manifest["control_manifest_fingerprint"],
                "production_reference": reference_commit,
                "protocol_fingerprint": manifest["protocol_fingerprint"],
            },
        )
        shutil.rmtree(temporary / "build")
        os.replace(temporary, output)
        return output
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _bundle_artifact_path(bundle: Path, artifact_id: str) -> Path:
    category, separator, name = artifact_id.partition("/")
    if not separator or not name:
        raise FreezeError(f"malformed artifact id: {artifact_id!r}")
    if category == "binary":
        filename = "enoch-week1-evaluator" if name == "week1-evaluator" else name
        return bundle / "bin" / filename
    if category == "model":
        return bundle / "models" / name
    if category == "preflight":
        return bundle / "preflight" / f"{name}.json"
    if category == "protocol" and name == "week1-seed-protocol":
        return bundle / "protocol" / "week1-seed-protocol.json"
    if category == "source":
        filenames = {
            "enoch-control-probe": "enoch-control-probe.rs",
            "enoch-control-probe-reference-patch": "enoch-control-probe-reference.patch",
            "production-reference": "production-reference.tar",
            "week1-evaluator": "week1-evaluator-source.tar.gz",
            "week1-evaluator-file-list": "week1-evaluator-source-files.json",
        }
        try:
            filename = filenames[name]
        except KeyError as exc:
            raise FreezeError(f"unknown source artifact: {name}") from exc
        return bundle / "source" / filename
    raise FreezeError(f"unknown artifact category: {category}")


def verify_bundle(protocol_path: Path, bundle: Path) -> str:
    protocol = enoch_week1.load_json_object(protocol_path)
    manifest = enoch_week1.load_json_object(bundle / "control-manifest.json")
    fingerprint = enoch_week1.validate_w1_0_control_manifest(protocol, manifest)
    artifact_hashes = manifest["artifact_hashes"]
    missing = CONTROL_PROBE_ARTIFACT_IDS - set(artifact_hashes)
    if missing:
        raise FreezeError(f"control probe artifacts are missing: {sorted(missing)}")
    for artifact_id, expected in artifact_hashes.items():
        path = _bundle_artifact_path(bundle, artifact_id)
        if not path.is_file() or _sha256_file(path) != expected:
            raise FreezeError(f"artifact hash mismatch: {artifact_id} ({path})")
    return fingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="build a new immutable W1.0 bundle")
    freeze.add_argument("--workspace", type=Path, default=Path.cwd())
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--reference", default=DEFAULT_REFERENCE)
    verify = subparsers.add_parser("verify", help="verify all frozen hashes")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            path = freeze_bundle(
                args.workspace.resolve(),
                args.protocol.resolve(),
                args.output.resolve(),
                args.reference,
            )
            print(path)
        else:
            print(verify_bundle(args.protocol.resolve(), args.bundle.resolve()))
    except (FreezeError, enoch_week1.ProtocolError, FileExistsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
