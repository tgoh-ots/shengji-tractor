#!/usr/bin/env python3
"""Week 1 seed registry and fail-closed evaluation control layer.

The protocol deliberately reserves every Week 1 seed namespace up front.  A
single master seed therefore defines all development, qualification, and locked
gate seeds before any result is observed.  Immutable phase, comparison, shard,
qualification, locked-gate, and terminal-decision artifacts keep later evidence
bound to that original protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:  # Unix is the supported training environment; this keeps imports portable.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-Unix hosts.
    fcntl = None


MANIFEST_VERSION = 1
PROTOCOL_KIND = "enoch-week1-seed-protocol"
LEDGER_KIND = "enoch-week1-seed-consumption-ledger"
FROZEN_PRODUCTION_REFERENCE_PREFIX = "c813c8a"
SEED_DERIVATION_DOMAIN = b"shengji/enoch-week1/seed/v1\0"
SEED_DERIVATION_DESCRIPTION = {
    "algorithm": "sha256-first-64-bits",
    "byte_order": "big",
    "domain": "shengji/enoch-week1/seed/v1",
    "index_encoding": "u64-big-endian",
    "master_seed_encoding": "u64-big-endian",
    "namespace_encoding": "u32-length-prefixed-utf8",
}
BLOCKED_EVALUATOR_ENV_PREFIXES = ("SHENGJI_", "GM_", "OMNI_", "GEN_")
DEFAULT_EVALUATOR_ENV_ALLOWLIST: tuple[str, ...] = ()
U64_MAX = (1 << 64) - 1

_NAMESPACE_RE = re.compile(r"[a-z0-9](?:[a-z0-9./-]*[a-z0-9])?")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IDENTIFIER_RE = re.compile(r"[a-z0-9](?:[a-z0-9._/-]*[a-z0-9])?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

ABLATION_ARMS = (
    "bid-ownership",
    "compound-follow",
    "failed-throw-better-player",
    "friend-revelation",
    "terminal-level-utility",
    "kitty-burial",
    "late-ruff-shape",
    "contextual-empty-trick",
    "relative-live-suit",
    "team-void-boss",
    "teammate-entry-return",
    "low-trump-handoff",
    "structural-family-coverage",
    "progressive-admission",
    "uncertain-legal-throws",
)

ARM_DESCRIPTIONS = {
    "bid-ownership": "bid-ownership evidence in determinization",
    "compound-follow": "compound pair/tractor follow evidence",
    "failed-throw-better-player": "failed-throw better_player evidence",
    "friend-revelation": "friend-revelation evidence",
    "terminal-level-utility": "exact terminal level utility for completed rollouts",
    "kitty-burial": "current Enoch kitty burial versus default point/shape burial",
    "late-ruff-shape": "late retained-trump ruff-shape planning",
    "contextual-empty-trick": "contextual empty-trick value",
    "relative-live-suit": "relative live-suit control and higher halters",
    "team-void-boss": "known team-void boss discounting",
    "teammate-entry-return": "teammate entry and return-suit planning",
    "low-trump-handoff": "low-trump handoff protection",
    "structural-family-coverage": "one representative per structural action family",
    "progressive-admission": "progressive admission beyond the initial top-K",
    "uncertain-legal-throws": "uncertain but legal throw admission",
}

CORRECTNESS_PREREQUISITES = (
    "arm-specific-deterministic-fixtures",
    "physical-copy-conservation",
    "authoritative-trick-decomposition",
    "mechanics-exhaustive-small-state-enumeration",
)

COMBINATION_PREREQUISITES = (
    "full-tactical-fixtures",
    "mechanics-tests",
    "hidden-information-tests",
    "model-contract-tests",
    "legal-coverage-prerequisites",
)

FAILURE_COUNTER_NAMES = (
    "illegal_action",
    "honesty_violation",
    "model_fallback",
    "model_contract_failure",
    "incomplete_pair",
    "hidden_information_leak",
    "artifact_mismatch",
    "cancellation",
    "fixture_failure",
    "machine_contention",
    "timeout",
)

# Every recorded timeout is an invalid evaluation event. Product-budget search
# may legitimately end because its wall-clock budget binds, but that is tracked
# separately in search telemetry; this counter means a decision/pair timed out.
INVALIDATING_FAILURE_COUNTERS = FAILURE_COUNTER_NAMES
WEEK1_EVALUATOR_CONTRACT = {
    "exact_seed_protocol": True,
    "in_process_candidate_control_isolation": True,
    "strict_no_fallback": True,
    "telemetry_required": True,
}
MODEL_SELECTION_EVIDENCE_IDS = {
    "fallback_disabled": "preflight/strict-evaluator-test",
    "intended_model_loaded": "preflight/expert-model-validation",
    "policy_selection_independent": "preflight/reference-model-contract-tests",
    "q_selection_independent": "preflight/reference-model-contract-tests",
    "value_selection_independent": "preflight/reference-model-contract-tests",
}
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_ALGORITHM = "paired-percentile-xorshift64star-v1"

# W1.2--W1.5 comparisons must retain one stable, non-empty style schema.  These
# are the metrics the in-process Rust evaluator currently supplies.  Keeping the
# registry here (rather than at a runner call site) makes omission or per-arm
# cherry-picking part of the hashed comparison contract.
WEEK1_STYLE_METRICS = (
    "compound-format-follow-rate",
    "empty-trick-ruff-rate",
    "failed-throw-rate",
    "follow-rate",
    "lead-rate",
    "multi-card-play-rate",
    "point-card-play-rate",
    "throw-rate",
    "trump-play-rate",
)

# Validation is intentionally expensive on first sight (all 35,111 seeds are
# rederived).  Exact canonical-byte caches retain fail-closed behavior while
# avoiding that work for the same unchanged manifest at every shard boundary.
_VALIDATED_PROTOCOL_CACHE: dict[int, tuple[str, str]] = {}
_VALIDATED_COMPARISON_CACHE: dict[int, tuple[str, str, str]] = {}

# Counts use the upper end wherever the plan specifies a range.  Consumers may
# use a prefix of a namespace, but the unused tail stays reserved forever.
SEED_NAMESPACE_COUNTS: tuple[tuple[str, int], ...] = (
    ("preflight/frozen-policy-equivalence", 100),
    ("baseline/searchless-outcome", 5_000),
    ("baseline/style", 5_000),
    ("smoke/product/001", 1),
    ("smoke/product/010", 10),
    ("smoke/product/100", 100),
    *tuple((f"dev/ablation/{arm}", 300) for arm in ABLATION_ARMS),
    *tuple((f"dev/survivor/{arm}", 800) for arm in ABLATION_ARMS),
    ("dev/combination/qualification", 300),
    ("dev/combination/screen", 800),
    ("qual/intended", 800),
    ("qual/equal", 800),
    ("qual/rank/low", 100),
    ("qual/rank/middle", 100),
    ("qual/rank/high", 100),
    ("qual/crossplay/assignment-01", 100),
    ("qual/crossplay/assignment-02", 100),
    ("qual/crossplay/assignment-03", 100),
    ("qual/crossplay/assignment-04", 100),
    ("qual/configuration/slot-01", 100),
    ("qual/configuration/slot-02", 100),
    ("qual/configuration/slot-03", 100),
    ("qual/finding-friends/contract-01", 100),
    ("qual/finding-friends/contract-02", 100),
    ("qual/scoring/kitty-ruleset-01", 100),
    ("qual/scoring/kitty-ruleset-02", 100),
    ("qual/scoring/kitty-ruleset-03", 100),
    ("qual/threshold/situation-01", 50),
    ("qual/threshold/situation-02", 50),
    ("qual/threshold/situation-03", 50),
    ("qual/threshold/situation-04", 50),
    ("locked/gate-1", 2_000),
    ("locked/confirmation", 2_000),
)


class ProtocolError(ValueError):
    """Raised when a protocol or ledger violates the frozen contract."""


class SeedReuseError(ProtocolError):
    """Raised when a seed has already been claimed by a consumer."""


def _require_u64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{label} must be an integer")
    if not 0 <= value <= U64_MAX:
        raise ProtocolError(f"{label} must be in [0, 2^64 - 1]")
    return value


def parse_u64(value: str) -> int:
    """Parse a decimal or ``0x``-prefixed unsigned 64-bit integer."""

    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid u64 seed: {value!r}") from exc
    try:
        return _require_u64(parsed, "master seed")
    except ProtocolError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_namespace(namespace: str) -> None:
    if not isinstance(namespace, str) or not _NAMESPACE_RE.fullmatch(namespace):
        raise ProtocolError(f"invalid seed namespace: {namespace!r}")
    if "//" in namespace or "/./" in namespace or "/../" in namespace:
        raise ProtocolError(f"invalid seed namespace: {namespace!r}")


def derive_seed(master_seed: int, namespace: str, index: int) -> int:
    """Derive one domain-separated deterministic unsigned 64-bit seed."""

    master_seed = _require_u64(master_seed, "master seed")
    _validate_namespace(namespace)
    index = _require_u64(index, "seed index")
    encoded_namespace = namespace.encode("utf-8")
    if len(encoded_namespace) > 0xFFFF_FFFF:
        raise ProtocolError("seed namespace is too long")
    payload = b"".join(
        (
            SEED_DERIVATION_DOMAIN,
            master_seed.to_bytes(8, "big"),
            len(encoded_namespace).to_bytes(4, "big"),
            encoded_namespace,
            index.to_bytes(8, "big"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole byte representation used for manifest hashing."""

    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not canonical-JSON serializable: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


WEEK1_STYLE_METRICS_SHA256 = canonical_json_sha256(WEEK1_STYLE_METRICS)


# These registries are data, not evaluator policy.  Their hashes are embedded in
# every downstream manifest so changing an arm name, phase, or gate silently is
# impossible.
ARM_REGISTRY: tuple[dict[str, Any], ...] = tuple(
    {
        "ablation_seed_namespace": f"dev/ablation/{arm_id}",
        "arm_id": arm_id,
        "description": ARM_DESCRIPTIONS[arm_id],
        "fixture_prerequisites": list(CORRECTNESS_PREREQUISITES),
        "independent": True,
        "inseparability_rationale": (
            "The retained-trump candidate, post-candidate kitty multiplier, and "
            "common-root exact terminal objective form one value-consistent action "
            "hypothesis; screening any component alone would compare incompatible "
            "value units or fail to expose the planned action."
            if arm_id == "late-ruff-shape"
            else None
        ),
        "ordinal": ordinal,
        "package_components": (
            [
                "common-root-terminal-level-objective",
                "post-candidate-kitty-multiplier",
                "retained-trump-structural-reservation",
            ]
            if arm_id == "late-ruff-shape"
            else []
        ),
        "survivor_seed_namespace": f"dev/survivor/{arm_id}",
    }
    for ordinal, arm_id in enumerate(ABLATION_ARMS, start=1)
)
ARM_REGISTRY_SHA256 = canonical_json_sha256(ARM_REGISTRY)

PHASE_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "phase": "W1.0",
        "predecessor_mode": "all",
        "predecessors": [],
        "seed_prefixes": [],
        "exit_artifact": "immutable-control-manifest",
    },
    {
        "phase": "W1.1",
        "predecessor_mode": "all",
        "predecessors": ["W1.0"],
        "seed_prefixes": ["preflight/", "baseline/", "smoke/"],
        "exit_artifact": "baseline-and-worker-report",
    },
    {
        "phase": "W1.2",
        "predecessor_mode": "all",
        "predecessors": ["W1.1"],
        "seed_prefixes": ["dev/ablation/"],
        "exit_artifact": "ranked-independent-ablation-table",
    },
    {
        "phase": "W1.3",
        "predecessor_mode": "all",
        "predecessors": ["W1.2"],
        "seed_prefixes": ["dev/survivor/"],
        "exit_artifact": "supported-independent-change-set",
    },
    {
        "phase": "W1.4",
        "predecessor_mode": "all",
        "predecessors": ["W1.3"],
        "seed_prefixes": ["dev/combination/"],
        "exit_artifact": "single-candidate-or-no-survivor",
    },
    {
        "phase": "W1.5",
        "predecessor_mode": "all",
        "predecessors": ["W1.4"],
        "seed_prefixes": ["qual/"],
        "exit_artifact": "qualification-decision",
    },
    {
        "phase": "W1.6",
        "predecessor_mode": "all",
        "predecessors": ["W1.5"],
        "seed_prefixes": ["locked/gate-1"],
        "exit_artifact": "primary-locked-gate-decision",
    },
    {
        "phase": "W1.7",
        "predecessor_mode": "all",
        "predecessors": ["W1.6"],
        "seed_prefixes": ["locked/confirmation"],
        "exit_artifact": "independent-confirmation-decision",
    },
    {
        "phase": "W1.8",
        "predecessor_mode": "any-one",
        "predecessors": ["W1.4", "W1.5", "W1.6", "W1.7"],
        "seed_prefixes": [],
        "exit_artifact": "freeze-or-no-confirmed-candidate-decision",
    },
)
PHASE_REGISTRY_SHA256 = canonical_json_sha256(PHASE_REGISTRY)

QUALIFICATION_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "category": "budget",
        "comparison_id": "intended",
        "namespace": "qual/intended",
        "pair_count": 800,
        "robustness": False,
    },
    {
        "category": "budget",
        "comparison_id": "equal",
        "namespace": "qual/equal",
        "pair_count": 800,
        "robustness": False,
    },
    *tuple(
        {
            "category": "rank",
            "comparison_id": f"rank-{rank}",
            "namespace": f"qual/rank/{rank}",
            "pair_count": 100,
            "robustness": True,
        }
        for rank in ("low", "middle", "high")
    ),
    *tuple(
        {
            "category": "crossplay",
            "comparison_id": f"crossplay-assignment-{index:02d}",
            "namespace": f"qual/crossplay/assignment-{index:02d}",
            "pair_count": 100,
            "robustness": True,
        }
        for index in range(1, 5)
    ),
    *tuple(
        {
            "category": "configuration",
            "comparison_id": f"configuration-slot-{index:02d}",
            "namespace": f"qual/configuration/slot-{index:02d}",
            "pair_count": 100,
            "robustness": True,
        }
        for index in range(1, 4)
    ),
    *tuple(
        {
            "category": "finding-friends",
            "comparison_id": f"finding-friends-contract-{index:02d}",
            "namespace": f"qual/finding-friends/contract-{index:02d}",
            "pair_count": 100,
            "robustness": True,
        }
        for index in range(1, 3)
    ),
    *tuple(
        {
            "category": "scoring-kitty",
            "comparison_id": f"scoring-kitty-ruleset-{index:02d}",
            "namespace": f"qual/scoring/kitty-ruleset-{index:02d}",
            "pair_count": 100,
            "robustness": True,
        }
        for index in range(1, 4)
    ),
    *tuple(
        {
            "category": "threshold",
            "comparison_id": f"threshold-situation-{index:02d}",
            "namespace": f"qual/threshold/situation-{index:02d}",
            "pair_count": 50,
            "robustness": True,
        }
        for index in range(1, 5)
    ),
)
QUALIFICATION_MATRIX_SHA256 = canonical_json_sha256(QUALIFICATION_MATRIX)

QUALIFICATION_THRESHOLDS = {
    "budget_level_utility_estimate_exclusive_min": 0.0,
    "budget_level_utility_lower_95_inclusive_min": -0.02,
    "budget_point_margin_estimate_exclusive_min": 0.0,
    "budget_win_rate_estimate_inclusive_min": -0.02,
    "pooled_robustness_level_utility_inclusive_min": 0.0,
    "stratum_level_utility_exclusive_min": -0.10,
    "zero_failure_counters": list(INVALIDATING_FAILURE_COUNTERS),
}
QUALIFICATION_THRESHOLDS_SHA256 = canonical_json_sha256(QUALIFICATION_THRESHOLDS)

LOCKED_SUPERIORITY_RULE = {
    "metric": "signed-level-utility-delta",
    "operator": ">",
    "statistic": "paired-bootstrap-lower-95",
    "threshold": 0.0,
}
LOCKED_SUPERIORITY_RULE_SHA256 = canonical_json_sha256(LOCKED_SUPERIORITY_RULE)


def _assert_static_control_contracts() -> None:
    if canonical_json_sha256(WEEK1_STYLE_METRICS) != WEEK1_STYLE_METRICS_SHA256:
        raise ProtocolError("in-process mutation of the Week-1 style schema detected")
    if canonical_json_sha256(ARM_REGISTRY) != ARM_REGISTRY_SHA256:
        raise ProtocolError("in-process mutation of the canonical arm registry detected")
    if canonical_json_sha256(PHASE_REGISTRY) != PHASE_REGISTRY_SHA256:
        raise ProtocolError("in-process mutation of the canonical phase registry detected")
    if canonical_json_sha256(QUALIFICATION_MATRIX) != QUALIFICATION_MATRIX_SHA256:
        raise ProtocolError("in-process mutation of the qualification matrix detected")
    if canonical_json_sha256(QUALIFICATION_THRESHOLDS) != QUALIFICATION_THRESHOLDS_SHA256:
        raise ProtocolError("in-process mutation of qualification thresholds detected")
    if canonical_json_sha256(LOCKED_SUPERIORITY_RULE) != LOCKED_SUPERIORITY_RULE_SHA256:
        raise ProtocolError("in-process mutation of the locked superiority rule detected")


def atomic_write_json(path: os.PathLike[str] | str, value: Any, *, overwrite: bool = False) -> None:
    """Atomically write canonical JSON, refusing replacement by default."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable file: {destination}")

    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Linking a complete same-directory temporary file is an atomic
            # create-if-absent operation.  Unlike an exists()/replace() pair,
            # this cannot overwrite a protocol created by a racing process.
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite immutable file: {destination}"
                ) from exc
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def load_json_object(path: os.PathLike[str] | str) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"could not load JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root in {path} must be an object")
    return value


def _validate_environment_allowlist(names: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(names))
    if len(set(normalized)) != len(normalized):
        raise ProtocolError("evaluator environment allowlist contains duplicates")
    for name in normalized:
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ProtocolError(f"invalid evaluator environment name: {name!r}")
        if not name.startswith(BLOCKED_EVALUATOR_ENV_PREFIXES):
            raise ProtocolError(
                f"allowlisted evaluator variable {name!r} does not use a blocked prefix"
            )
    return normalized


def sanitized_evaluator_environment(
    environment: Mapping[str, str] | None = None,
    *,
    allowlist: Iterable[str] = DEFAULT_EVALUATOR_ENV_ALLOWLIST,
    overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Remove ambient experiment knobs except explicitly allowlisted names.

    Returns ``(environment, removed_names)``.  Prefixed overrides must also be
    allowlisted, preventing an accidental bypass at the call site.
    """

    allowed = set(_validate_environment_allowlist(allowlist))
    source = os.environ if environment is None else environment
    cleaned: dict[str, str] = {}
    removed: list[str] = []
    for name, value in source.items():
        if name.startswith(BLOCKED_EVALUATOR_ENV_PREFIXES) and name not in allowed:
            removed.append(name)
        else:
            cleaned[name] = value
    for name, value in (overrides or {}).items():
        if not _ENV_NAME_RE.fullmatch(name):
            raise ProtocolError(f"invalid evaluator environment name: {name!r}")
        if name.startswith(BLOCKED_EVALUATOR_ENV_PREFIXES) and name not in allowed:
            raise ProtocolError(f"prefixed override {name!r} is not explicitly allowlisted")
        cleaned[name] = value
    return cleaned, tuple(sorted(removed))


def _assert_global_disjointness(namespaces: Sequence[Mapping[str, Any]]) -> None:
    owners: dict[int, tuple[str, int]] = {}
    for entry in namespaces:
        name = entry.get("name")
        seeds = entry.get("seeds")
        if not isinstance(name, str) or not isinstance(seeds, list):
            raise ProtocolError("malformed namespace while checking disjointness")
        for index, seed in enumerate(seeds):
            seed = _require_u64(seed, f"seed {name}[{index}]")
            previous = owners.get(seed)
            if previous is not None:
                raise ProtocolError(
                    f"seed collision: {name}[{index}] reuses seed {seed} from "
                    f"{previous[0]}[{previous[1]}]"
                )
            owners[seed] = (name, index)


def build_seed_registry(master_seed: int) -> dict[str, Any]:
    master_seed = _require_u64(master_seed, "master seed")
    namespaces: list[dict[str, Any]] = []
    for name, count in SEED_NAMESPACE_COUNTS:
        _validate_namespace(name)
        namespaces.append(
            {
                "count": count,
                "name": name,
                "seeds": [derive_seed(master_seed, name, index) for index in range(count)],
            }
        )
    _assert_global_disjointness(namespaces)
    return {
        "derivation": copy.deepcopy(SEED_DERIVATION_DESCRIPTION),
        "global_seed_count": sum(count for _, count in SEED_NAMESPACE_COUNTS),
        "master_seed": master_seed,
        "namespaces": namespaces,
    }


def build_protocol(
    master_seed: int,
    *,
    evaluator_env_allowlist: Iterable[str] = DEFAULT_EVALUATOR_ENV_ALLOWLIST,
) -> dict[str, Any]:
    registry = build_seed_registry(master_seed)
    allowlist = _validate_environment_allowlist(evaluator_env_allowlist)
    body: dict[str, Any] = {
        "automatic_production_promotion_allowed": False,
        "evaluator_environment_policy": {
            "allowlist": list(allowlist),
            "blocked_prefixes": list(BLOCKED_EVALUATOR_ENV_PREFIXES),
        },
        "manifest_version": MANIFEST_VERSION,
        "protocol_kind": PROTOCOL_KIND,
        "seed_registry": registry,
        "seed_registry_sha256": canonical_json_sha256(registry),
    }
    return {**body, "protocol_fingerprint": canonical_json_sha256(body)}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolError(f"{label} keys differ; missing={missing}, extra={extra}")


def validate_protocol(protocol: Mapping[str, Any]) -> str:
    """Validate all frozen invariants and return the protocol fingerprint."""

    if not isinstance(protocol, Mapping):
        raise ProtocolError("protocol must be an object")
    canonical_digest = canonical_json_sha256(protocol)
    cached = _VALIDATED_PROTOCOL_CACHE.get(id(protocol))
    if cached is not None and cached[0] == canonical_digest:
        return cached[1]
    _require_exact_keys(
        protocol,
        {
            "automatic_production_promotion_allowed",
            "evaluator_environment_policy",
            "manifest_version",
            "protocol_fingerprint",
            "protocol_kind",
            "seed_registry",
            "seed_registry_sha256",
        },
        "protocol",
    )
    if protocol["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported manifest version")
    if protocol["protocol_kind"] != PROTOCOL_KIND:
        raise ProtocolError("unexpected protocol kind")
    if protocol["automatic_production_promotion_allowed"] is not False:
        raise ProtocolError("automatic production promotion must remain disabled")

    policy = protocol["evaluator_environment_policy"]
    if not isinstance(policy, Mapping):
        raise ProtocolError("evaluator environment policy must be an object")
    _require_exact_keys(policy, {"allowlist", "blocked_prefixes"}, "environment policy")
    if policy["blocked_prefixes"] != list(BLOCKED_EVALUATOR_ENV_PREFIXES):
        raise ProtocolError("evaluator environment blocked prefixes changed")
    if not isinstance(policy["allowlist"], list):
        raise ProtocolError("evaluator environment allowlist must be a list")
    if list(_validate_environment_allowlist(policy["allowlist"])) != policy["allowlist"]:
        raise ProtocolError("evaluator environment allowlist must be sorted")

    registry = protocol["seed_registry"]
    if not isinstance(registry, Mapping):
        raise ProtocolError("seed registry must be an object")
    _require_exact_keys(
        registry,
        {"derivation", "global_seed_count", "master_seed", "namespaces"},
        "seed registry",
    )
    master_seed = _require_u64(registry["master_seed"], "master seed")
    if registry["derivation"] != SEED_DERIVATION_DESCRIPTION:
        raise ProtocolError("seed derivation contract changed")
    expected_total = sum(count for _, count in SEED_NAMESPACE_COUNTS)
    if registry["global_seed_count"] != expected_total:
        raise ProtocolError("global seed count changed")
    namespaces = registry["namespaces"]
    if not isinstance(namespaces, list) or len(namespaces) != len(SEED_NAMESPACE_COUNTS):
        raise ProtocolError("seed namespace set changed")

    for entry, (expected_name, expected_count) in zip(namespaces, SEED_NAMESPACE_COUNTS):
        if not isinstance(entry, Mapping):
            raise ProtocolError("seed namespace entry must be an object")
        _require_exact_keys(entry, {"count", "name", "seeds"}, "seed namespace")
        if entry["name"] != expected_name or entry["count"] != expected_count:
            raise ProtocolError(
                f"seed namespace contract changed at {expected_name!r}: "
                f"got {entry['name']!r}/{entry['count']!r}"
            )
        if not isinstance(entry["seeds"], list) or len(entry["seeds"]) != expected_count:
            raise ProtocolError(f"seed count changed for {expected_name}")
        expected_seeds = [
            derive_seed(master_seed, expected_name, index) for index in range(expected_count)
        ]
        if entry["seeds"] != expected_seeds:
            raise ProtocolError(f"derived seeds changed for {expected_name}")
    _assert_global_disjointness(namespaces)

    expected_registry_hash = canonical_json_sha256(registry)
    if protocol["seed_registry_sha256"] != expected_registry_hash:
        raise ProtocolError("seed registry hash mismatch")
    body = dict(protocol)
    fingerprint = body.pop("protocol_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != canonical_json_sha256(body):
        raise ProtocolError("protocol fingerprint mismatch")
    if len(_VALIDATED_PROTOCOL_CACHE) >= 16:
        _VALIDATED_PROTOCOL_CACHE.clear()
    _VALIDATED_PROTOCOL_CACHE[id(protocol)] = (canonical_digest, fingerprint)
    return fingerprint


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProtocolError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ProtocolError(f"{label} must be a canonical lowercase identifier")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = " nonnegative" if nonnegative else ""
        raise ProtocolError(f"{label} must be a finite{qualifier} number")
    return number


def _with_fingerprint(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: canonical_json_sha256(body)}


def _validate_fingerprint(value: Mapping[str, Any], field: str, label: str) -> str:
    if field not in value:
        raise ProtocolError(f"{label} is missing {field}")
    fingerprint = _require_sha256(value[field], f"{label} {field}")
    body = dict(value)
    body.pop(field)
    if fingerprint != canonical_json_sha256(body):
        raise ProtocolError(f"{label} fingerprint mismatch")
    return fingerprint


def canonical_arm_registry() -> list[dict[str, Any]]:
    """Return an isolated copy of the immutable 15-arm Week 1 registry."""

    _assert_static_control_contracts()
    return copy.deepcopy(list(ARM_REGISTRY))


def _phase_spec(phase: str) -> Mapping[str, Any]:
    for spec in PHASE_REGISTRY:
        if spec["phase"] == phase:
            return spec
    raise ProtocolError(f"unknown Week 1 phase: {phase!r}")


def _validate_artifact_references(artifacts: Any) -> None:
    if not isinstance(artifacts, list):
        raise ProtocolError("phase artifacts must be a list")
    previous: str | None = None
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ProtocolError("phase artifact reference must be an object")
        _require_exact_keys(artifact, {"artifact_id", "sha256"}, "artifact reference")
        artifact_id = _require_identifier(artifact["artifact_id"], "artifact id")
        _require_sha256(artifact["sha256"], f"artifact {artifact_id} hash")
        if previous is not None and artifact_id <= previous:
            raise ProtocolError("phase artifact references must be sorted and unique")
        previous = artifact_id


def _validate_phase_parent_records(
    phase_spec: Mapping[str, Any], parents: Any
) -> None:
    if not isinstance(parents, list):
        raise ProtocolError("phase parents must be a list")
    records: list[tuple[str, str]] = []
    for parent in parents:
        if not isinstance(parent, Mapping):
            raise ProtocolError("phase parent must be an object")
        _require_exact_keys(
            parent,
            {"phase", "phase_manifest_fingerprint"},
            "phase parent",
        )
        parent_phase = parent["phase"]
        _phase_spec(parent_phase)
        parent_fingerprint = _require_sha256(
            parent["phase_manifest_fingerprint"], "parent phase fingerprint"
        )
        records.append((parent_phase, parent_fingerprint))
    if records != sorted(set(records)):
        raise ProtocolError("phase parents must be sorted and unique")
    allowed = phase_spec["predecessors"]
    mode = phase_spec["predecessor_mode"]
    actual_phases = [phase for phase, _ in records]
    if mode == "all":
        if actual_phases != allowed:
            raise ProtocolError(
                f"{phase_spec['phase']} requires predecessor phases {allowed}, "
                f"got {actual_phases}"
            )
    elif mode == "any-one":
        if len(actual_phases) != 1 or actual_phases[0] not in allowed:
            raise ProtocolError(
                f"{phase_spec['phase']} requires exactly one terminal predecessor "
                f"from {allowed}, got {actual_phases}"
            )
    else:
        raise ProtocolError("unsupported phase predecessor mode")


def validate_phase_chain(
    protocol: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate one complete W1.0-to-terminal-predecessor lineage."""

    if not manifests:
        raise ProtocolError("phase chain cannot be empty")
    fingerprints: list[str] = []
    for index, manifest in enumerate(manifests):
        fingerprint = validate_phase_manifest(protocol, manifest)
        expected_phase = f"W1.{index}"
        if manifest["phase"] != expected_phase or expected_phase == "W1.8":
            raise ProtocolError(
                f"phase chain must be contiguous from W1.0; expected {expected_phase}"
            )
        if index > 0:
            expected_parent = [{
                "phase": manifests[index - 1]["phase"],
                "phase_manifest_fingerprint": fingerprints[index - 1],
            }]
            if manifest["parent_phases"] != expected_parent:
                raise ProtocolError("phase chain parent fingerprint does not match predecessor")
        fingerprints.append(fingerprint)
    return fingerprints


def build_phase_manifest(
    protocol: Mapping[str, Any],
    phase: str,
    *,
    artifacts: Mapping[str, str] | None = None,
    declarations: Mapping[str, Any] | None = None,
    parent_phase_manifests: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a generic immutable phase envelope for non-evaluation artifacts."""

    protocol_fingerprint = validate_protocol(protocol)
    _assert_static_control_contracts()
    phase_spec = _phase_spec(phase)
    artifact_records = sorted(
        (
            {"artifact_id": _require_identifier(name, "artifact id"), "sha256": _require_sha256(digest, f"artifact {name} hash")}
            for name, digest in (artifacts or {}).items()
        ),
        key=lambda item: item["artifact_id"],
    )
    _validate_artifact_references(artifact_records)
    if not artifact_records:
        raise ProtocolError(f"{phase} completion must bind at least one exit artifact")
    parent_records = []
    for parent in parent_phase_manifests:
        parent_fingerprint = validate_phase_manifest(protocol, parent)
        parent_records.append(
            {
                "phase": parent["phase"],
                "phase_manifest_fingerprint": parent_fingerprint,
            }
        )
    parent_records.sort(key=lambda record: record["phase"])
    _validate_phase_parent_records(phase_spec, parent_records)
    declaration_object = dict(declarations or {})
    canonical_json_bytes(declaration_object)
    if not declaration_object:
        raise ProtocolError(f"{phase} completion declarations cannot be empty")
    body = {
        "arm_registry_sha256": ARM_REGISTRY_SHA256,
        "artifacts": artifact_records,
        "automatic_production_promotion_allowed": False,
        "declarations": declaration_object,
        "declared_exit_artifact": phase_spec["exit_artifact"],
        "manifest_kind": "enoch-week1-phase-manifest",
        "manifest_version": MANIFEST_VERSION,
        "parent_phases": parent_records,
        "phase": phase,
        "phase_registry_sha256": PHASE_REGISTRY_SHA256,
        "protocol_fingerprint": protocol_fingerprint,
        "seed_registry_sha256": protocol["seed_registry_sha256"],
    }
    return _with_fingerprint(body, "phase_manifest_fingerprint")


def validate_phase_manifest(protocol: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    protocol_fingerprint = validate_protocol(protocol)
    _assert_static_control_contracts()
    if not isinstance(manifest, Mapping):
        raise ProtocolError("phase manifest must be an object")
    _require_exact_keys(
        manifest,
        {
            "arm_registry_sha256",
            "artifacts",
            "automatic_production_promotion_allowed",
            "declarations",
            "declared_exit_artifact",
            "manifest_kind",
            "manifest_version",
            "parent_phases",
            "phase",
            "phase_manifest_fingerprint",
            "phase_registry_sha256",
            "protocol_fingerprint",
            "seed_registry_sha256",
        },
        "phase manifest",
    )
    if manifest["manifest_kind"] != "enoch-week1-phase-manifest":
        raise ProtocolError("unexpected phase manifest kind")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported phase manifest version")
    if manifest["automatic_production_promotion_allowed"] is not False:
        raise ProtocolError("phase manifest cannot authorize production promotion")
    phase_spec = _phase_spec(manifest["phase"])
    if manifest["declared_exit_artifact"] != phase_spec["exit_artifact"]:
        raise ProtocolError("phase exit artifact declaration changed")
    if manifest["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("phase manifest protocol fingerprint mismatch")
    if manifest["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise ProtocolError("phase manifest seed registry mismatch")
    if manifest["arm_registry_sha256"] != ARM_REGISTRY_SHA256:
        raise ProtocolError("phase manifest arm registry mismatch")
    if manifest["phase_registry_sha256"] != PHASE_REGISTRY_SHA256:
        raise ProtocolError("phase manifest phase registry mismatch")
    _validate_artifact_references(manifest["artifacts"])
    if not manifest["artifacts"]:
        raise ProtocolError("completed phase manifest has no exit artifact")
    _validate_phase_parent_records(phase_spec, manifest["parent_phases"])
    if not isinstance(manifest["declarations"], Mapping):
        raise ProtocolError("phase declarations must be an object")
    canonical_json_bytes(manifest["declarations"])
    if not manifest["declarations"]:
        raise ProtocolError("completed phase declarations cannot be empty")
    return _validate_fingerprint(manifest, "phase_manifest_fingerprint", "phase manifest")


def _validate_w1_0_identity_bindings(
    policies: Mapping[str, Mapping[str, Any]],
    evaluator: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    search_knobs: Mapping[str, Any],
) -> None:
    binary_artifacts = {
        "enoch-0": "binary/enoch-0",
        "expert-0": "binary/expert-0",
        "grandmaster-0": "binary/grandmaster-0",
    }
    required = {
        *binary_artifacts.values(),
        "binary/week1-evaluator",
        "model/expert_model.onnx",
        "protocol/week1-seed-protocol",
        "source/production-reference",
        "source/week1-evaluator-file-list",
    }
    missing = required - set(artifact_hashes)
    if missing:
        raise ProtocolError(f"W1.0 identity artifacts are missing: {sorted(missing)}")
    reference_source = artifact_hashes["source/production-reference"]
    no_model = canonical_json_sha256(
        {"model": "none", "reason": "heuristic-policy-tier"}
    )
    for name, identity in policies.items():
        if identity["source_sha256"] != reference_source:
            raise ProtocolError(f"{name} source identity differs from production archive")
        if identity["binary_sha256"] != artifact_hashes[binary_artifacts[name]]:
            raise ProtocolError(f"{name} binary identity differs from frozen artifact")
        if name not in search_knobs:
            raise ProtocolError(f"{name} search configuration is missing")
        if identity["configuration_sha256"] != canonical_json_sha256(search_knobs[name]):
            raise ProtocolError(f"{name} configuration identity differs from search knobs")
    if policies["expert-0"]["model_sha256"] != artifact_hashes["model/expert_model.onnx"]:
        raise ProtocolError("Expert-0 model identity differs from frozen artifact")
    for name in ("enoch-0", "grandmaster-0"):
        if policies[name]["model_sha256"] != no_model:
            raise ProtocolError(f"{name} must bind the explicit no-model identity")
    if evaluator["binary_sha256"] != artifact_hashes["binary/week1-evaluator"]:
        raise ProtocolError("Week 1 evaluator binary identity differs from artifact")
    if evaluator["source_sha256"] != artifact_hashes["source/week1-evaluator-file-list"]:
        raise ProtocolError("Week 1 evaluator source identity differs from file-list artifact")
    if evaluator["configuration_sha256"] != canonical_json_sha256(WEEK1_EVALUATOR_CONTRACT):
        raise ProtocolError("Week 1 evaluator configuration identity changed")


def build_w1_0_control_manifest(
    protocol: Mapping[str, Any],
    *,
    production_reference: str,
    policy_identities: Mapping[str, Mapping[str, Any]],
    evaluator_identity: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    compiler: str,
    operating_system: str,
    hardware_summary: str,
    search_knobs: Mapping[str, Any],
    effective_environment: Mapping[str, str],
    replay_commands: Sequence[str],
    model_selection_contract: Mapping[str, bool],
    model_selection_evidence: Mapping[str, str],
) -> dict[str, Any]:
    """Build the fail-closed W1.0 frozen-control and environment manifest."""

    protocol_fingerprint = validate_protocol(protocol)
    _assert_static_control_contracts()
    if not isinstance(production_reference, str) or not re.fullmatch(
        r"[0-9a-f]{7,40}", production_reference
    ):
        raise ProtocolError("production reference must be a 7-40 digit lowercase git SHA")
    if not production_reference.startswith(FROZEN_PRODUCTION_REFERENCE_PREFIX):
        raise ProtocolError(
            f"Week 1 production reference must resolve from {FROZEN_PRODUCTION_REFERENCE_PREFIX}"
        )
    if set(policy_identities) != {"enoch-0", "expert-0", "grandmaster-0"}:
        raise ProtocolError("control manifest must freeze Enoch-0, Expert-0, and Grandmaster-0")
    normalized_policies: dict[str, dict[str, Any]] = {}
    for name in sorted(policy_identities):
        _validate_frozen_identity(policy_identities[name], POLICY_IDENTITY_FIELDS, name)
        normalized_policies[name] = copy.deepcopy(dict(policy_identities[name]))
    _validate_frozen_identity(evaluator_identity, EVALUATOR_IDENTITY_FIELDS, "evaluator")
    if not artifact_hashes:
        raise ProtocolError("control manifest must record artifact hashes")
    normalized_hashes: dict[str, str] = {}
    for name, digest in sorted(artifact_hashes.items()):
        normalized_hashes[_require_identifier(name, "artifact hash id")] = _require_sha256(
            digest, f"artifact {name} hash"
        )
    for value, label in (
        (compiler, "compiler"),
        (operating_system, "operating system"),
        (hardware_summary, "hardware summary"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"{label} must be a nonempty string")
    canonical_json_bytes(search_knobs)
    if not isinstance(search_knobs, Mapping) or not search_knobs:
        raise ProtocolError("control manifest must record every search knob")
    allowlist = set(protocol["evaluator_environment_policy"]["allowlist"])
    normalized_environment: dict[str, str] = {}
    for name, value in sorted(effective_environment.items()):
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ProtocolError(f"invalid effective environment name: {name!r}")
        if not isinstance(value, str):
            raise ProtocolError(f"effective environment value for {name} must be a string")
        if name.startswith(BLOCKED_EVALUATOR_ENV_PREFIXES) and name not in allowlist:
            raise ProtocolError(f"effective environment retained blocked variable {name!r}")
        normalized_environment[name] = value
    if not replay_commands or not all(
        isinstance(command, str) and command.strip() for command in replay_commands
    ):
        raise ProtocolError("control manifest requires nonempty replay commands")
    expected_contract = {
        "fallback_disabled": True,
        "intended_model_loaded": True,
        "policy_selection_independent": True,
        "q_selection_independent": True,
        "value_selection_independent": True,
    }
    if dict(model_selection_contract) != expected_contract:
        raise ProtocolError("model selection/fallback contract is not proven fail-closed")
    if dict(model_selection_evidence) != MODEL_SELECTION_EVIDENCE_IDS:
        raise ProtocolError("model selection evidence mapping changed")
    for artifact_id in MODEL_SELECTION_EVIDENCE_IDS.values():
        if artifact_id not in normalized_hashes:
            raise ProtocolError(f"model selection evidence artifact is missing: {artifact_id}")
    _validate_w1_0_identity_bindings(
        normalized_policies,
        evaluator_identity,
        normalized_hashes,
        search_knobs,
    )
    body = {
        "arm_registry_sha256": ARM_REGISTRY_SHA256,
        "artifact_hashes": normalized_hashes,
        "automatic_production_promotion_allowed": False,
        "compiler": compiler,
        "effective_environment": normalized_environment,
        "evaluator_identity": copy.deepcopy(dict(evaluator_identity)),
        "hardware_summary": hardware_summary,
        "manifest_kind": "enoch-week1-w1.0-control-manifest",
        "manifest_version": MANIFEST_VERSION,
        "model_selection_contract": expected_contract,
        "model_selection_evidence": dict(sorted(model_selection_evidence.items())),
        "operating_system": operating_system,
        "phase": "W1.0",
        "policy_identities": normalized_policies,
        "production_reference": production_reference,
        "protocol_fingerprint": protocol_fingerprint,
        "replay_commands": list(replay_commands),
        "search_knobs": copy.deepcopy(dict(search_knobs)),
        "seed_registry_sha256": protocol["seed_registry_sha256"],
    }
    manifest = _with_fingerprint(body, "control_manifest_fingerprint")
    validate_w1_0_control_manifest(protocol, manifest)
    return manifest


def validate_w1_0_control_manifest(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str:
    protocol_fingerprint = validate_protocol(protocol)
    _assert_static_control_contracts()
    if not isinstance(manifest, Mapping):
        raise ProtocolError("W1.0 control manifest must be an object")
    _require_exact_keys(
        manifest,
        {
            "arm_registry_sha256",
            "artifact_hashes",
            "automatic_production_promotion_allowed",
            "compiler",
            "control_manifest_fingerprint",
            "effective_environment",
            "evaluator_identity",
            "hardware_summary",
            "manifest_kind",
            "manifest_version",
            "model_selection_contract",
            "model_selection_evidence",
            "operating_system",
            "phase",
            "policy_identities",
            "production_reference",
            "protocol_fingerprint",
            "replay_commands",
            "search_knobs",
            "seed_registry_sha256",
        },
        "W1.0 control manifest",
    )
    if manifest["manifest_kind"] != "enoch-week1-w1.0-control-manifest":
        raise ProtocolError("unexpected W1.0 control manifest kind")
    if manifest["manifest_version"] != MANIFEST_VERSION or manifest["phase"] != "W1.0":
        raise ProtocolError("unexpected W1.0 control manifest version or phase")
    if manifest["automatic_production_promotion_allowed"] is not False:
        raise ProtocolError("W1.0 control manifest cannot authorize promotion")
    if manifest["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("W1.0 control manifest protocol mismatch")
    if manifest["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise ProtocolError("W1.0 control manifest seed registry mismatch")
    if manifest["arm_registry_sha256"] != ARM_REGISTRY_SHA256:
        raise ProtocolError("W1.0 control manifest arm registry mismatch")
    if not isinstance(manifest["production_reference"], str) or not re.fullmatch(
        r"[0-9a-f]{7,40}", manifest["production_reference"]
    ):
        raise ProtocolError("W1.0 production reference is not a canonical git SHA")
    if not manifest["production_reference"].startswith(FROZEN_PRODUCTION_REFERENCE_PREFIX):
        raise ProtocolError("W1.0 production reference differs from the frozen control")
    policies = manifest["policy_identities"]
    if not isinstance(policies, Mapping) or list(policies) != [
        "enoch-0",
        "expert-0",
        "grandmaster-0",
    ]:
        raise ProtocolError("W1.0 policy identities are incomplete or unsorted")
    for name, identity in policies.items():
        _validate_frozen_identity(identity, POLICY_IDENTITY_FIELDS, name)
    _validate_frozen_identity(
        manifest["evaluator_identity"], EVALUATOR_IDENTITY_FIELDS, "evaluator"
    )
    artifact_hashes = manifest["artifact_hashes"]
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise ProtocolError("W1.0 artifact hash set is empty")
    if list(artifact_hashes) != sorted(artifact_hashes):
        raise ProtocolError("W1.0 artifact hashes must be sorted")
    for name, digest in artifact_hashes.items():
        _require_identifier(name, "artifact hash id")
        _require_sha256(digest, f"artifact {name} hash")
    for field in ("compiler", "operating_system", "hardware_summary"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ProtocolError(f"W1.0 {field} must be a nonempty string")
    if not isinstance(manifest["search_knobs"], Mapping) or not manifest["search_knobs"]:
        raise ProtocolError("W1.0 search knob declaration is empty")
    canonical_json_bytes(manifest["search_knobs"])
    environment = manifest["effective_environment"]
    if not isinstance(environment, Mapping) or list(environment) != sorted(environment):
        raise ProtocolError("W1.0 effective environment must be a sorted object")
    allowlist = set(protocol["evaluator_environment_policy"]["allowlist"])
    for name, value in environment.items():
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ProtocolError(f"invalid W1.0 environment name: {name!r}")
        if not isinstance(value, str):
            raise ProtocolError(f"W1.0 environment value for {name} must be a string")
        if name.startswith(BLOCKED_EVALUATOR_ENV_PREFIXES) and name not in allowlist:
            raise ProtocolError(f"W1.0 environment retained blocked variable {name!r}")
    replay_commands = manifest["replay_commands"]
    if not isinstance(replay_commands, list) or not replay_commands or not all(
        isinstance(command, str) and command.strip() for command in replay_commands
    ):
        raise ProtocolError("W1.0 replay commands are missing or invalid")
    expected_contract = {
        "fallback_disabled": True,
        "intended_model_loaded": True,
        "policy_selection_independent": True,
        "q_selection_independent": True,
        "value_selection_independent": True,
    }
    if manifest["model_selection_contract"] != expected_contract:
        raise ProtocolError("W1.0 model selection/fallback contract is not fail-closed")
    if manifest["model_selection_evidence"] != dict(
        sorted(MODEL_SELECTION_EVIDENCE_IDS.items())
    ):
        raise ProtocolError("W1.0 model selection evidence mapping changed")
    for artifact_id in MODEL_SELECTION_EVIDENCE_IDS.values():
        if artifact_id not in artifact_hashes:
            raise ProtocolError(f"W1.0 model evidence artifact is missing: {artifact_id}")
    _validate_w1_0_identity_bindings(
        policies,
        manifest["evaluator_identity"],
        artifact_hashes,
        manifest["search_knobs"],
    )
    return _validate_fingerprint(
        manifest, "control_manifest_fingerprint", "W1.0 control manifest"
    )


def _protocol_seed(protocol: Mapping[str, Any], namespace: str, index: int) -> int:
    _validate_namespace(namespace)
    index = _require_u64(index, "seed index")
    for entry in protocol["seed_registry"]["namespaces"]:
        if entry["name"] == namespace:
            if index >= entry["count"]:
                raise ProtocolError(f"seed index {index} is outside namespace {namespace}")
            return entry["seeds"][index]
    raise ProtocolError(f"unknown seed namespace: {namespace}")


def _ledger_with_fingerprint(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "ledger_fingerprint": canonical_json_sha256(body)}


def new_seed_ledger(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol_fingerprint = validate_protocol(protocol)
    return _ledger_with_fingerprint(
        {
            "consumed": [],
            "ledger_kind": LEDGER_KIND,
            "manifest_version": MANIFEST_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "seed_registry_sha256": protocol["seed_registry_sha256"],
        }
    )


def validate_seed_ledger(protocol: Mapping[str, Any], ledger: Mapping[str, Any]) -> str:
    protocol_fingerprint = validate_protocol(protocol)
    if not isinstance(ledger, Mapping):
        raise ProtocolError("seed ledger must be an object")
    _require_exact_keys(
        ledger,
        {
            "consumed",
            "ledger_fingerprint",
            "ledger_kind",
            "manifest_version",
            "protocol_fingerprint",
            "seed_registry_sha256",
        },
        "seed ledger",
    )
    if ledger["manifest_version"] != MANIFEST_VERSION or ledger["ledger_kind"] != LEDGER_KIND:
        raise ProtocolError("unexpected seed ledger version or kind")
    if ledger["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("seed ledger belongs to a different protocol")
    if ledger["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise ProtocolError("seed ledger belongs to a different registry")
    body = dict(ledger)
    ledger_fingerprint = body.pop("ledger_fingerprint")
    if ledger_fingerprint != canonical_json_sha256(body):
        raise ProtocolError("seed ledger fingerprint mismatch")

    consumed = ledger["consumed"]
    if not isinstance(consumed, list):
        raise ProtocolError("seed ledger consumed field must be a list")
    seen_coordinates: set[tuple[str, int]] = set()
    seen_seeds: set[int] = set()
    for sequence, record in enumerate(consumed):
        if not isinstance(record, Mapping):
            raise ProtocolError("seed consumption record must be an object")
        _require_exact_keys(
            record, {"consumer", "index", "namespace", "seed", "sequence"}, "consumption record"
        )
        if record["sequence"] != sequence:
            raise ProtocolError("seed ledger sequence is not contiguous")
        consumer = record["consumer"]
        if not isinstance(consumer, str) or not consumer.strip() or len(consumer) > 256:
            raise ProtocolError("seed consumer must be a nonempty string of at most 256 characters")
        namespace = record["namespace"]
        index = record["index"]
        expected_seed = _protocol_seed(protocol, namespace, index)
        if record["seed"] != expected_seed:
            raise ProtocolError("seed ledger record does not match the protocol")
        coordinate = (namespace, index)
        if coordinate in seen_coordinates or expected_seed in seen_seeds:
            raise SeedReuseError(f"seed was reused in ledger: {namespace}[{index}]")
        seen_coordinates.add(coordinate)
        seen_seeds.add(expected_seed)
    return ledger_fingerprint


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def consume_seed_batch_once(
    ledger_path: os.PathLike[str] | str,
    protocol: Mapping[str, Any],
    claims: Sequence[tuple[str, int, str]],
) -> dict[str, Any]:
    """Atomically claim a batch of seeds, rejecting any prior or in-batch reuse."""

    validate_protocol(protocol)
    destination = Path(ledger_path)
    with _exclusive_lock(destination.with_name(f"{destination.name}.lock")):
        ledger = load_json_object(destination) if destination.exists() else new_seed_ledger(protocol)
        validate_seed_ledger(protocol, ledger)
        updated = copy.deepcopy(ledger)
        existing_coordinates = {
            (record["namespace"], record["index"]) for record in updated["consumed"]
        }
        existing_seeds = {record["seed"] for record in updated["consumed"]}
        for namespace, index, consumer in claims:
            if not isinstance(consumer, str) or not consumer.strip() or len(consumer) > 256:
                raise ProtocolError(
                    "seed consumer must be a nonempty string of at most 256 characters"
                )
            seed = _protocol_seed(protocol, namespace, index)
            coordinate = (namespace, index)
            if coordinate in existing_coordinates or seed in existing_seeds:
                raise SeedReuseError(f"seed already consumed: {namespace}[{index}]")
            updated["consumed"].append(
                {
                    "consumer": consumer,
                    "index": index,
                    "namespace": namespace,
                    "seed": seed,
                    "sequence": len(updated["consumed"]),
                }
            )
            existing_coordinates.add(coordinate)
            existing_seeds.add(seed)
        body = dict(updated)
        body.pop("ledger_fingerprint")
        updated = _ledger_with_fingerprint(body)
        validate_seed_ledger(protocol, updated)
        atomic_write_json(destination, updated, overwrite=True)
        return updated


def consume_seed_once(
    ledger_path: os.PathLike[str] | str,
    protocol: Mapping[str, Any],
    namespace: str,
    index: int,
    consumer: str,
) -> dict[str, Any]:
    return consume_seed_batch_once(ledger_path, protocol, [(namespace, index, consumer)])


def build_development_rule(
    rule_id: str,
    *,
    minimum_level_utility_estimate: float | None = None,
    minimum_level_utility_lower_95: float | None = None,
    minimum_point_margin_estimate: float | None = None,
    minimum_win_rate_estimate: float | None = None,
    maximum_candidate_p95_latency_ms: float | None = None,
    minimum_candidate_completed_worlds_mean: float | None = None,
    style_metric_bounds: Mapping[str, Mapping[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Declare an arm's development rule before its W1.2 observations exist."""

    rule = {
        "maximum_candidate_p95_latency_ms": maximum_candidate_p95_latency_ms,
        "minimum_candidate_completed_worlds_mean": minimum_candidate_completed_worlds_mean,
        "minimum_level_utility_estimate": minimum_level_utility_estimate,
        "minimum_level_utility_lower_95": minimum_level_utility_lower_95,
        "minimum_point_margin_estimate": minimum_point_margin_estimate,
        "minimum_win_rate_estimate": minimum_win_rate_estimate,
        "require_zero_invalidating_failures": True,
        "rule_id": rule_id,
        "style_metric_bounds": {
            name: {"maximum": bounds.get("maximum"), "minimum": bounds.get("minimum")}
            for name, bounds in sorted((style_metric_bounds or {}).items())
        },
    }
    validate_development_rule(rule)
    return rule


def validate_development_rule(rule: Mapping[str, Any]) -> str:
    if not isinstance(rule, Mapping):
        raise ProtocolError("development rule must be an object")
    _require_exact_keys(
        rule,
        {
            "maximum_candidate_p95_latency_ms",
            "minimum_candidate_completed_worlds_mean",
            "minimum_level_utility_estimate",
            "minimum_level_utility_lower_95",
            "minimum_point_margin_estimate",
            "minimum_win_rate_estimate",
            "require_zero_invalidating_failures",
            "rule_id",
            "style_metric_bounds",
        },
        "development rule",
    )
    _require_identifier(rule["rule_id"], "development rule id")
    if rule["require_zero_invalidating_failures"] is not True:
        raise ProtocolError("development rules must require zero invalidating failures")
    numeric_fields = (
        "minimum_level_utility_estimate",
        "minimum_level_utility_lower_95",
        "minimum_point_margin_estimate",
        "minimum_win_rate_estimate",
        "maximum_candidate_p95_latency_ms",
        "minimum_candidate_completed_worlds_mean",
    )
    for field in numeric_fields:
        value = rule[field]
        if value is not None:
            _require_finite_number(
                value,
                f"development rule {field}",
                nonnegative=field in {
                    "maximum_candidate_p95_latency_ms",
                    "minimum_candidate_completed_worlds_mean",
                },
            )
    bounds = rule["style_metric_bounds"]
    if not isinstance(bounds, Mapping):
        raise ProtocolError("style metric bounds must be an object")
    if list(bounds) != sorted(bounds):
        raise ProtocolError("style metric bounds must be sorted")
    for metric, limits in bounds.items():
        _require_identifier(metric, "style metric name")
        if not isinstance(limits, Mapping):
            raise ProtocolError(f"style bounds for {metric} must be an object")
        _require_exact_keys(limits, {"maximum", "minimum"}, f"style bounds for {metric}")
        if limits["minimum"] is None and limits["maximum"] is None:
            raise ProtocolError(f"style bounds for {metric} declare no bound")
        for direction in ("minimum", "maximum"):
            if limits[direction] is not None:
                _require_finite_number(limits[direction], f"{metric} {direction}")
        if (
            limits["minimum"] is not None
            and limits["maximum"] is not None
            and limits["minimum"] > limits["maximum"]
        ):
            raise ProtocolError(f"style bounds for {metric} are inverted")
    if not any(rule[field] is not None for field in numeric_fields) and not bounds:
        raise ProtocolError("development rule must declare at least one performance threshold")
    return canonical_json_sha256(rule)


def _registered_namespace_count(namespace: str) -> int:
    for name, count in SEED_NAMESPACE_COUNTS:
        if name == namespace:
            return count
    raise ProtocolError(f"unregistered Week 1 seed namespace: {namespace}")


def _arm_entry(arm_id: str) -> Mapping[str, Any]:
    for arm in ARM_REGISTRY:
        if arm["arm_id"] == arm_id:
            return arm
    raise ProtocolError(f"unknown Week 1 arm: {arm_id!r}")


def _validate_phase_seed_contract(
    phase: str, namespace: str, pair_count: int, subject_id: str
) -> None:
    spec = _phase_spec(phase)
    if not any(namespace.startswith(prefix) for prefix in spec["seed_prefixes"]):
        raise ProtocolError(f"namespace {namespace!r} is not permitted in {phase}")
    available = _registered_namespace_count(namespace)
    if pair_count > available:
        raise ProtocolError(f"pair count {pair_count} exceeds namespace capacity {available}")
    if phase == "W1.2":
        arm = _arm_entry(subject_id)
        if namespace != arm["ablation_seed_namespace"] or not 200 <= pair_count <= 300:
            raise ProtocolError("W1.2 requires 200-300 pairs in the arm's ablation namespace")
    elif phase == "W1.3":
        arm = _arm_entry(subject_id)
        if namespace != arm["survivor_seed_namespace"] or pair_count != 800:
            raise ProtocolError("W1.3 requires exactly 800 pairs in the arm's survivor namespace")
    elif phase == "W1.4":
        expected = {
            "dev/combination/qualification": range(200, 301),
            "dev/combination/screen": range(800, 801),
        }
        if namespace not in expected or pair_count not in expected[namespace]:
            raise ProtocolError("W1.4 pair count does not match the combination stage")
    elif phase == "W1.5" and pair_count != available:
        raise ProtocolError("W1.5 robustness and budget strata require their full namespace")
    elif phase == "W1.6":
        if namespace != "locked/gate-1" or not 1_500 <= pair_count <= 2_000:
            raise ProtocolError("W1.6 requires 1,500-2,000 pairs from locked/gate-1")
    elif phase == "W1.7":
        if namespace != "locked/confirmation" or not 1_500 <= pair_count <= 2_000:
            raise ProtocolError(
                "W1.7 requires 1,500-2,000 pairs from locked/confirmation"
            )
    elif phase == "W1.1" and namespace.startswith("smoke/product/"):
        if pair_count != available:
            raise ProtocolError("product smoke must consume its exact 1/10/100-pair namespace")


def _partition_indices(indices: Sequence[int], shard_count: int) -> list[dict[str, Any]]:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise ProtocolError("shard count must be an integer")
    if not 1 <= shard_count <= len(indices):
        raise ProtocolError("shard count must be between one and the pair count")
    base, remainder = divmod(len(indices), shard_count)
    assignments: list[dict[str, Any]] = []
    start = 0
    for ordinal in range(shard_count):
        width = base + (1 if ordinal < remainder else 0)
        assigned = list(indices[start : start + width])
        assignments.append(
            {
                "seed_indices": assigned,
                "shard_id": f"shard-{ordinal:03d}",
            }
        )
        start += width
    return assignments


def _seed_set(protocol: Mapping[str, Any], namespace: str, indices: Sequence[int]) -> list[dict[str, int]]:
    return [
        {"index": index, "seed": _protocol_seed(protocol, namespace, index)}
        for index in indices
    ]


def build_comparison_protocol_manifest(
    protocol: Mapping[str, Any],
    *,
    phase: str,
    comparison_id: str,
    subject_id: str,
    seed_namespace: str,
    pair_count: int,
    shard_count: int,
    candidate_fingerprint: str,
    control_fingerprint: str,
    evaluator_fingerprint: str,
    environment_fingerprint: str,
    configuration_fingerprint: str,
    development_rule: Mapping[str, Any] | None = None,
    required_style_metrics: Iterable[str] = WEEK1_STYLE_METRICS,
) -> dict[str, Any]:
    """Freeze one comparison, including its exact seed-to-shard assignment."""

    protocol_fingerprint = validate_protocol(protocol)
    _assert_static_control_contracts()
    comparison_id = _require_identifier(comparison_id, "comparison id")
    subject_id = _require_identifier(subject_id, "comparison subject id")
    _validate_namespace(seed_namespace)
    pair_count = _require_nonnegative_int(pair_count, "pair count")
    if pair_count == 0:
        raise ProtocolError("pair count must be positive")
    _validate_phase_seed_contract(phase, seed_namespace, pair_count, subject_id)
    bindings = {
        "candidate_fingerprint": _require_sha256(candidate_fingerprint, "candidate fingerprint"),
        "configuration_fingerprint": _require_sha256(
            configuration_fingerprint, "configuration fingerprint"
        ),
        "control_fingerprint": _require_sha256(control_fingerprint, "control fingerprint"),
        "environment_fingerprint": _require_sha256(
            environment_fingerprint, "environment fingerprint"
        ),
        "evaluator_fingerprint": _require_sha256(evaluator_fingerprint, "evaluator fingerprint"),
    }
    if phase == "W1.2":
        if development_rule is None:
            raise ProtocolError("W1.2 comparison must predeclare its advancement rule")
        validate_development_rule(development_rule)
    elif development_rule is not None:
        validate_development_rule(development_rule)
    requested_metrics = list(required_style_metrics)
    metrics = sorted(requested_metrics)
    if len(metrics) != len(set(metrics)):
        raise ProtocolError("required style metrics contain duplicates")
    for metric in metrics:
        _require_identifier(metric, "required style metric")
    if phase in {"W1.2", "W1.3", "W1.4", "W1.5"} and metrics != list(
        WEEK1_STYLE_METRICS
    ):
        raise ProtocolError(
            f"{phase} comparisons must retain the complete frozen Week-1 style schema"
        )
    indices = list(range(pair_count))
    seed_set = _seed_set(protocol, seed_namespace, indices)
    shards = _partition_indices(indices, shard_count)
    for shard in shards:
        shard["seed_set_sha256"] = canonical_json_sha256(
            _seed_set(protocol, seed_namespace, shard["seed_indices"])
        )
    body = {
        **bindings,
        "arm_registry_sha256": ARM_REGISTRY_SHA256,
        "automatic_production_promotion_allowed": False,
        "bootstrap": {
            "algorithm": BOOTSTRAP_ALGORITHM,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "comparison_id": comparison_id,
        "development_rule": copy.deepcopy(development_rule),
        "manifest_kind": "enoch-week1-comparison-protocol",
        "manifest_version": MANIFEST_VERSION,
        "pair_count": pair_count,
        "phase": phase,
        "protocol_fingerprint": protocol_fingerprint,
        "required_style_metrics": metrics,
        "seed_indices": indices,
        "seed_namespace": seed_namespace,
        "seed_registry_sha256": protocol["seed_registry_sha256"],
        "seed_set_sha256": canonical_json_sha256(seed_set),
        "shards": shards,
        "subject_id": subject_id,
    }
    return _with_fingerprint(body, "comparison_protocol_fingerprint")


# The longer name is kept as a readable public alias for callers that treat
# these as evaluator-run manifests rather than comparisons.
build_evaluation_protocol_manifest = build_comparison_protocol_manifest


def validate_comparison_protocol_manifest(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str:
    protocol_fingerprint = validate_protocol(protocol)
    _assert_static_control_contracts()
    if not isinstance(manifest, Mapping):
        raise ProtocolError("comparison protocol must be an object")
    canonical_digest = canonical_json_sha256(manifest)
    cached = _VALIDATED_COMPARISON_CACHE.get(id(manifest))
    if (
        cached is not None
        and cached[0] == canonical_digest
        and cached[1] == protocol_fingerprint
    ):
        return cached[2]
    expected_keys = {
        "arm_registry_sha256",
        "automatic_production_promotion_allowed",
        "bootstrap",
        "candidate_fingerprint",
        "comparison_id",
        "comparison_protocol_fingerprint",
        "configuration_fingerprint",
        "control_fingerprint",
        "development_rule",
        "environment_fingerprint",
        "evaluator_fingerprint",
        "manifest_kind",
        "manifest_version",
        "pair_count",
        "phase",
        "protocol_fingerprint",
        "required_style_metrics",
        "seed_indices",
        "seed_namespace",
        "seed_registry_sha256",
        "seed_set_sha256",
        "shards",
        "subject_id",
    }
    _require_exact_keys(manifest, expected_keys, "comparison protocol")
    if manifest["manifest_kind"] != "enoch-week1-comparison-protocol":
        raise ProtocolError("unexpected comparison protocol kind")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported comparison protocol version")
    if manifest["automatic_production_promotion_allowed"] is not False:
        raise ProtocolError("comparison protocol cannot authorize promotion")
    if manifest["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("comparison protocol seed protocol mismatch")
    if manifest["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise ProtocolError("comparison protocol seed registry mismatch")
    if manifest["arm_registry_sha256"] != ARM_REGISTRY_SHA256:
        raise ProtocolError("comparison protocol arm registry mismatch")
    for field in (
        "candidate_fingerprint",
        "configuration_fingerprint",
        "control_fingerprint",
        "environment_fingerprint",
        "evaluator_fingerprint",
    ):
        _require_sha256(manifest[field], field)
    comparison_id = _require_identifier(manifest["comparison_id"], "comparison id")
    subject_id = _require_identifier(manifest["subject_id"], "comparison subject id")
    _validate_namespace(manifest["seed_namespace"])
    pair_count = _require_nonnegative_int(manifest["pair_count"], "pair count")
    if pair_count == 0:
        raise ProtocolError("pair count must be positive")
    _validate_phase_seed_contract(
        manifest["phase"], manifest["seed_namespace"], pair_count, subject_id
    )
    expected_indices = list(range(pair_count))
    if manifest["seed_indices"] != expected_indices:
        raise ProtocolError("comparison seed indices must be the frozen namespace prefix")
    expected_seed_set = _seed_set(protocol, manifest["seed_namespace"], expected_indices)
    if manifest["seed_set_sha256"] != canonical_json_sha256(expected_seed_set):
        raise ProtocolError("comparison seed set hash mismatch")
    bootstrap = manifest["bootstrap"]
    if bootstrap != {"algorithm": BOOTSTRAP_ALGORITHM, "replicates": BOOTSTRAP_REPLICATES}:
        raise ProtocolError("comparison bootstrap declaration changed")
    metrics = manifest["required_style_metrics"]
    if not isinstance(metrics, list) or metrics != sorted(set(metrics)):
        raise ProtocolError("required style metrics must be sorted and unique")
    for metric in metrics:
        _require_identifier(metric, "required style metric")
    if manifest["phase"] in {"W1.2", "W1.3", "W1.4", "W1.5"} and metrics != list(
        WEEK1_STYLE_METRICS
    ):
        raise ProtocolError(
            f"{manifest['phase']} comparison does not retain the frozen Week-1 style schema"
        )
    if manifest["phase"] == "W1.2" and manifest["development_rule"] is None:
        raise ProtocolError("W1.2 comparison lacks a predeclared advancement rule")
    if manifest["development_rule"] is not None:
        validate_development_rule(manifest["development_rule"])
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ProtocolError("comparison must declare at least one shard")
    expected_shards = _partition_indices(expected_indices, len(shards))
    for expected, actual in zip(expected_shards, shards):
        if not isinstance(actual, Mapping):
            raise ProtocolError("comparison shard assignment must be an object")
        _require_exact_keys(
            actual, {"seed_indices", "seed_set_sha256", "shard_id"}, "shard assignment"
        )
        if actual["shard_id"] != expected["shard_id"]:
            raise ProtocolError("comparison shard ids are not canonical")
        if actual["seed_indices"] != expected["seed_indices"]:
            raise ProtocolError("comparison shard seed coverage changed")
        expected_hash = canonical_json_sha256(
            _seed_set(protocol, manifest["seed_namespace"], expected["seed_indices"])
        )
        if actual["seed_set_sha256"] != expected_hash:
            raise ProtocolError("comparison shard seed set hash mismatch")
    del comparison_id  # validated for canonical form; retained in the fingerprint.
    fingerprint = _validate_fingerprint(
        manifest, "comparison_protocol_fingerprint", "comparison protocol"
    )
    if len(_VALIDATED_COMPARISON_CACHE) >= 128:
        _VALIDATED_COMPARISON_CACHE.clear()
    _VALIDATED_COMPARISON_CACHE[id(manifest)] = (
        canonical_digest,
        protocol_fingerprint,
        fingerprint,
    )
    return fingerprint


validate_evaluation_protocol_manifest = validate_comparison_protocol_manifest


def _validate_pair_record(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_index: int,
) -> None:
    if not isinstance(record, Mapping):
        raise ProtocolError("paired result record must be an object")
    _require_exact_keys(
        record,
        {
            "candidate_completed_worlds",
            "candidate_latency_ms",
            "complete",
            "control_completed_worlds",
            "control_latency_ms",
            "effective_deal_seed",
            "failure_counters",
            "level_utility_delta",
            "orientations_completed",
            "point_margin_delta",
            "seed",
            "seed_index",
            "style_metrics",
            "win_rate_delta",
        },
        "paired result record",
    )
    _require_nonnegative_int(record["seed_index"], "paired record seed index")
    if record["seed_index"] != expected_index:
        raise ProtocolError("paired records are missing, duplicated, or out of canonical order")
    expected_seed = _protocol_seed(protocol, comparison["seed_namespace"], expected_index)
    _require_u64(record["seed"], f"paired record {expected_index} seed")
    if record["seed"] != expected_seed:
        raise ProtocolError(f"paired record seed mismatch at index {expected_index}")
    effective_deal_seed = _require_u64(
        record["effective_deal_seed"],
        f"paired record {expected_index} effective deal seed",
    )
    threshold_enriched = comparison["seed_namespace"].startswith("qual/threshold/")
    if not threshold_enriched and effective_deal_seed != expected_seed:
        raise ProtocolError(
            f"non-threshold paired record {expected_index} changed its frozen deal seed"
        )
    _require_nonnegative_int(
        record["orientations_completed"], f"paired record {expected_index} orientation count"
    )
    if record["complete"] is not True or record["orientations_completed"] != 2:
        raise ProtocolError(f"paired record {expected_index} is not a complete two-orientation unit")
    for field in ("level_utility_delta", "point_margin_delta", "win_rate_delta"):
        _require_finite_number(record[field], f"paired record {expected_index} {field}")
    if not -1.0 <= float(record["win_rate_delta"]) <= 1.0:
        raise ProtocolError("win-rate delta must be in [-1, 1]")
    for field in ("candidate_latency_ms", "control_latency_ms"):
        _require_finite_number(
            record[field], f"paired record {expected_index} {field}", nonnegative=True
        )
    for field in ("candidate_completed_worlds", "control_completed_worlds"):
        _require_nonnegative_int(record[field], f"paired record {expected_index} {field}")
    counters = record["failure_counters"]
    if not isinstance(counters, Mapping) or set(counters) != set(FAILURE_COUNTER_NAMES):
        raise ProtocolError("paired result failure counters do not match the frozen schema")
    for name in FAILURE_COUNTER_NAMES:
        _require_nonnegative_int(counters[name], f"failure counter {name}")
    style = record["style_metrics"]
    required_metrics = comparison["required_style_metrics"]
    if not isinstance(style, Mapping) or set(style) != set(required_metrics):
        raise ProtocolError("paired result style metrics do not match the comparison declaration")
    for name in required_metrics:
        _require_finite_number(style[name], f"style metric {name}")


def build_shard_result(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    shard_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    verified_external_evidence_fingerprint: str,
) -> dict[str, Any]:
    """Seal one complete shard.  Partial shards cannot be constructed."""

    comparison_fingerprint = validate_comparison_protocol_manifest(protocol, comparison)
    external_evidence_fingerprint = _require_sha256(
        verified_external_evidence_fingerprint,
        "verified external evidence fingerprint",
    )
    assignment = next(
        (item for item in comparison["shards"] if item["shard_id"] == shard_id), None
    )
    if assignment is None:
        raise ProtocolError(f"unknown shard id for comparison: {shard_id!r}")
    canonical_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolError("shard record must be an object")
        copied = copy.deepcopy(dict(record))
        _require_nonnegative_int(copied.get("seed_index"), "shard record seed index")
        canonical_records.append(copied)
    canonical_records.sort(key=lambda item: item["seed_index"])
    if len(canonical_records) != len(assignment["seed_indices"]):
        raise ProtocolError("shard does not contain its exact declared pair count")
    for record, expected_index in zip(canonical_records, assignment["seed_indices"]):
        _validate_pair_record(protocol, comparison, record, expected_index)
    body = {
        "candidate_fingerprint": comparison["candidate_fingerprint"],
        "comparison_protocol_fingerprint": comparison_fingerprint,
        "configuration_fingerprint": comparison["configuration_fingerprint"],
        "control_fingerprint": comparison["control_fingerprint"],
        "environment_fingerprint": comparison["environment_fingerprint"],
        "evaluator_fingerprint": comparison["evaluator_fingerprint"],
        "manifest_kind": "enoch-week1-shard-result",
        "manifest_version": MANIFEST_VERSION,
        "protocol_fingerprint": comparison["protocol_fingerprint"],
        "records": canonical_records,
        "seed_registry_sha256": comparison["seed_registry_sha256"],
        "shard_id": shard_id,
        "shard_seed_set_sha256": assignment["seed_set_sha256"],
        "verified_external_evidence_fingerprint": external_evidence_fingerprint,
    }
    result = _with_fingerprint(body, "shard_result_fingerprint")
    validate_shard_result(protocol, comparison, result)
    return result


def validate_shard_result(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> str:
    comparison_fingerprint = validate_comparison_protocol_manifest(protocol, comparison)
    if not isinstance(shard, Mapping):
        raise ProtocolError("shard result must be an object")
    _require_exact_keys(
        shard,
        {
            "candidate_fingerprint",
            "comparison_protocol_fingerprint",
            "configuration_fingerprint",
            "control_fingerprint",
            "environment_fingerprint",
            "evaluator_fingerprint",
            "manifest_kind",
            "manifest_version",
            "protocol_fingerprint",
            "records",
            "seed_registry_sha256",
            "shard_id",
            "shard_result_fingerprint",
            "shard_seed_set_sha256",
            "verified_external_evidence_fingerprint",
        },
        "shard result",
    )
    if shard["manifest_kind"] != "enoch-week1-shard-result":
        raise ProtocolError("unexpected shard result kind")
    if shard["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported shard result version")
    if shard["comparison_protocol_fingerprint"] != comparison_fingerprint:
        raise ProtocolError("shard comparison fingerprint mismatch")
    for field in (
        "candidate_fingerprint",
        "configuration_fingerprint",
        "control_fingerprint",
        "environment_fingerprint",
        "evaluator_fingerprint",
        "protocol_fingerprint",
        "seed_registry_sha256",
    ):
        if shard[field] != comparison[field]:
            raise ProtocolError(f"shard {field} does not match its comparison protocol")
    assignment = next(
        (item for item in comparison["shards"] if item["shard_id"] == shard["shard_id"]),
        None,
    )
    if assignment is None:
        raise ProtocolError("shard id is not declared by the comparison protocol")
    if shard["shard_seed_set_sha256"] != assignment["seed_set_sha256"]:
        raise ProtocolError("shard seed-set hash mismatch")
    _require_sha256(
        shard["verified_external_evidence_fingerprint"],
        "verified external evidence fingerprint",
    )
    records = shard["records"]
    if not isinstance(records, list) or len(records) != len(assignment["seed_indices"]):
        raise ProtocolError("shard does not have exact declared seed coverage")
    for record, expected_index in zip(records, assignment["seed_indices"]):
        _validate_pair_record(protocol, comparison, record, expected_index)
    return _validate_fingerprint(shard, "shard_result_fingerprint", "shard result")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ProtocolError("cannot summarize an empty paired result")
    return math.fsum(values) / len(values)


def _nearest_rank_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return float(ordered[rank - 1])


def _bootstrap_interval_95(
    values: Sequence[float], comparison_fingerprint: str
) -> tuple[float, float]:
    if all(value == values[0] for value in values[1:]):
        value = float(values[0])
        return value, value
    state = int.from_bytes(
        hashlib.sha256(
            b"shengji/enoch-week1/bootstrap/v1\0"
            + comparison_fingerprint.encode("ascii")
        ).digest()[:8],
        "big",
    ) or 0x9E3779B97F4A7C15
    mask = U64_MAX
    means: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for _ in values:
            state ^= state >> 12
            state ^= (state << 25) & mask
            state ^= state >> 27
            state &= mask
            random_u64 = (state * 0x2545F4914F6CDD1D) & mask
            total += values[random_u64 % len(values)]
        means.append(total / len(values))
    means.sort()
    lower_index = max(0, math.ceil(0.025 * BOOTSTRAP_REPLICATES) - 1)
    upper_index = min(
        BOOTSTRAP_REPLICATES - 1,
        max(0, math.ceil(0.975 * BOOTSTRAP_REPLICATES) - 1),
    )
    return means[lower_index], means[upper_index]


def _bootstrap_lower_95(values: Sequence[float], comparison_fingerprint: str) -> float:
    return _bootstrap_interval_95(values, comparison_fingerprint)[0]


def _minimum_detectable_effect(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    # Normal approximation, two-sided alpha=.05 and 80% power.
    return (1.959963984540054 + 0.8416212335729143) * math.sqrt(variance / len(values))


def _outcome_metric_summary(
    values: Sequence[float], comparison_fingerprint: str
) -> dict[str, Any]:
    lower, upper = _bootstrap_interval_95(values, comparison_fingerprint)
    return {
        "estimate": _mean(values),
        "mde_95_80": _minimum_detectable_effect(values),
        "paired_bootstrap_95": [lower, upper],
        "paired_bootstrap_lower_95": lower,
        "paired_bootstrap_upper_95": upper,
    }


def _summarize_records(
    records: Sequence[Mapping[str, Any]], comparison_fingerprint: str
) -> dict[str, Any]:
    levels = [float(record["level_utility_delta"]) for record in records]
    point_margins = [float(record["point_margin_delta"]) for record in records]
    win_rates = [float(record["win_rate_delta"]) for record in records]
    candidate_latencies = [float(record["candidate_latency_ms"]) for record in records]
    control_latencies = [float(record["control_latency_ms"]) for record in records]
    candidate_worlds = [float(record["candidate_completed_worlds"]) for record in records]
    control_worlds = [float(record["control_completed_worlds"]) for record in records]
    style_names = sorted(records[0]["style_metrics"])
    return {
        "bootstrap": {
            "algorithm": BOOTSTRAP_ALGORITHM,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "candidate_completed_worlds_mean": _mean(candidate_worlds),
        "candidate_latency_ms": {
            "p50": _nearest_rank_quantile(candidate_latencies, 0.50),
            "p95": _nearest_rank_quantile(candidate_latencies, 0.95),
        },
        "control_completed_worlds_mean": _mean(control_worlds),
        "control_latency_ms": {
            "p50": _nearest_rank_quantile(control_latencies, 0.50),
            "p95": _nearest_rank_quantile(control_latencies, 0.95),
        },
        "failure_counters": {
            name: sum(record["failure_counters"][name] for record in records)
            for name in FAILURE_COUNTER_NAMES
        },
        "level_utility": _outcome_metric_summary(levels, comparison_fingerprint),
        "pair_count": len(records),
        "point_margin": _outcome_metric_summary(point_margins, comparison_fingerprint),
        "point_margin_estimate": _mean(point_margins),
        "style_metric_estimates": {
            name: _mean([float(record["style_metrics"][name]) for record in records])
            for name in style_names
        },
        "win_rate": _outcome_metric_summary(win_rates, comparison_fingerprint),
        "win_rate_estimate": _mean(win_rates),
    }


def merge_shard_results(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge only the exact, complete shard set declared by the protocol."""

    comparison_fingerprint = validate_comparison_protocol_manifest(protocol, comparison)
    if len(shards) != len(comparison["shards"]):
        raise ProtocolError("merge requires every declared shard exactly once")
    by_id: dict[str, Mapping[str, Any]] = {}
    for shard in shards:
        fingerprint = validate_shard_result(protocol, comparison, shard)
        shard_id = shard["shard_id"]
        if shard_id in by_id:
            raise ProtocolError(f"duplicate shard in merge: {shard_id}")
        _require_sha256(fingerprint, "shard result fingerprint")
        by_id[shard_id] = shard
    expected_ids = [assignment["shard_id"] for assignment in comparison["shards"]]
    if set(by_id) != set(expected_ids):
        raise ProtocolError("merge shard ids do not exactly match the protocol")
    external_evidence_fingerprints = {
        shard["verified_external_evidence_fingerprint"] for shard in by_id.values()
    }
    if len(external_evidence_fingerprints) != 1:
        raise ProtocolError(
            "merge requires every shard to use the same verified external evidence"
        )
    external_evidence_fingerprint = next(iter(external_evidence_fingerprints))
    records = [
        copy.deepcopy(record)
        for shard_id in expected_ids
        for record in by_id[shard_id]["records"]
    ]
    records.sort(key=lambda record: record["seed_index"])
    if [record["seed_index"] for record in records] != comparison["seed_indices"]:
        raise ProtocolError("merged result does not have exact seed coverage")
    shard_references = [
        {
            "shard_id": shard_id,
            "shard_result_fingerprint": by_id[shard_id]["shard_result_fingerprint"],
            "shard_seed_set_sha256": by_id[shard_id]["shard_seed_set_sha256"],
        }
        for shard_id in expected_ids
    ]
    body = {
        "comparison_protocol_fingerprint": comparison_fingerprint,
        "manifest_kind": "enoch-week1-merged-result",
        "manifest_version": MANIFEST_VERSION,
        "metrics": _summarize_records(records, comparison_fingerprint),
        "protocol_fingerprint": comparison["protocol_fingerprint"],
        "records": records,
        "records_sha256": canonical_json_sha256(records),
        "seed_registry_sha256": comparison["seed_registry_sha256"],
        "seed_set_sha256": comparison["seed_set_sha256"],
        "shards": shard_references,
        "verified_external_evidence_fingerprint": external_evidence_fingerprint,
    }
    merged = _with_fingerprint(body, "merged_result_fingerprint")
    validate_merged_result(protocol, comparison, merged)
    return merged


def validate_merged_result(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> str:
    comparison_fingerprint = validate_comparison_protocol_manifest(protocol, comparison)
    if not isinstance(merged, Mapping):
        raise ProtocolError("merged result must be an object")
    _require_exact_keys(
        merged,
        {
            "comparison_protocol_fingerprint",
            "manifest_kind",
            "manifest_version",
            "merged_result_fingerprint",
            "metrics",
            "protocol_fingerprint",
            "records",
            "records_sha256",
            "seed_registry_sha256",
            "seed_set_sha256",
            "shards",
            "verified_external_evidence_fingerprint",
        },
        "merged result",
    )
    if merged["manifest_kind"] != "enoch-week1-merged-result":
        raise ProtocolError("unexpected merged result kind")
    if merged["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported merged result version")
    for field in (
        "comparison_protocol_fingerprint",
        "protocol_fingerprint",
        "seed_registry_sha256",
        "seed_set_sha256",
    ):
        expected = (
            comparison_fingerprint
            if field == "comparison_protocol_fingerprint"
            else comparison[field]
        )
        if merged[field] != expected:
            raise ProtocolError(f"merged result {field} mismatch")
    records = merged["records"]
    if not isinstance(records, list) or len(records) != comparison["pair_count"]:
        raise ProtocolError("merged result pair count is incomplete")
    for record, expected_index in zip(records, comparison["seed_indices"]):
        _validate_pair_record(protocol, comparison, record, expected_index)
    effective_seeds = [record["effective_deal_seed"] for record in records]
    if len(effective_seeds) != len(set(effective_seeds)):
        raise ProtocolError("merged result contains duplicate effective deal seeds")
    if merged["records_sha256"] != canonical_json_sha256(records):
        raise ProtocolError("merged raw-record hash mismatch")
    _require_sha256(
        merged["verified_external_evidence_fingerprint"],
        "verified external evidence fingerprint",
    )
    shard_references = merged["shards"]
    if not isinstance(shard_references, list) or len(shard_references) != len(
        comparison["shards"]
    ):
        raise ProtocolError("merged result shard reference set is incomplete")
    for reference, assignment in zip(shard_references, comparison["shards"]):
        if not isinstance(reference, Mapping):
            raise ProtocolError("merged shard reference must be an object")
        _require_exact_keys(
            reference,
            {"shard_id", "shard_result_fingerprint", "shard_seed_set_sha256"},
            "merged shard reference",
        )
        if reference["shard_id"] != assignment["shard_id"]:
            raise ProtocolError("merged shard references are not canonical")
        _require_sha256(reference["shard_result_fingerprint"], "shard result fingerprint")
        if reference["shard_seed_set_sha256"] != assignment["seed_set_sha256"]:
            raise ProtocolError("merged shard seed-set hash mismatch")
    expected_metrics = _summarize_records(records, comparison_fingerprint)
    if merged["metrics"] != expected_metrics:
        raise ProtocolError("merged statistics do not reconstruct from raw paired records")
    return _validate_fingerprint(merged, "merged_result_fingerprint", "merged result")


def _development_rule_failures(
    rule: Mapping[str, Any], metrics: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    comparisons = (
        (
            "minimum_level_utility_estimate",
            metrics["level_utility"]["estimate"],
            "level-utility-estimate-below-rule",
        ),
        (
            "minimum_level_utility_lower_95",
            metrics["level_utility"]["paired_bootstrap_lower_95"],
            "level-utility-lower-95-below-rule",
        ),
        (
            "minimum_point_margin_estimate",
            metrics["point_margin_estimate"],
            "point-margin-estimate-below-rule",
        ),
        (
            "minimum_win_rate_estimate",
            metrics["win_rate_estimate"],
            "win-rate-estimate-below-rule",
        ),
        (
            "minimum_candidate_completed_worlds_mean",
            metrics["candidate_completed_worlds_mean"],
            "completed-worlds-below-rule",
        ),
    )
    for rule_field, observed, reason in comparisons:
        threshold = rule[rule_field]
        if threshold is not None and observed < threshold:
            failures.append(reason)
    latency_max = rule["maximum_candidate_p95_latency_ms"]
    if latency_max is not None and metrics["candidate_latency_ms"]["p95"] > latency_max:
        failures.append("candidate-p95-latency-above-rule")
    for metric, bounds in rule["style_metric_bounds"].items():
        if metric not in metrics["style_metric_estimates"]:
            failures.append(f"style-metric-missing:{metric}")
            continue
        observed = metrics["style_metric_estimates"][metric]
        if bounds["minimum"] is not None and observed < bounds["minimum"]:
            failures.append(f"style-metric-below-rule:{metric}")
        if bounds["maximum"] is not None and observed > bounds["maximum"]:
            failures.append(f"style-metric-above-rule:{metric}")
    if rule["require_zero_invalidating_failures"]:
        for counter in INVALIDATING_FAILURE_COUNTERS:
            if metrics["failure_counters"][counter] != 0:
                failures.append(f"nonzero-failure-counter:{counter}")
    return failures


def _w1_3_advancement_body(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    merged: Mapping[str, Any],
    fixture_report: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_fingerprint = validate_comparison_protocol_manifest(protocol, comparison)
    if comparison["phase"] != "W1.2":
        raise ProtocolError("W1.3 advancement requires a W1.2 ablation result")
    merged_fingerprint = validate_merged_result(protocol, comparison, merged)
    if not isinstance(fixture_report, Mapping):
        raise ProtocolError("W1.3 advancement requires the sealed fixture report")
    try:
        try:
            from . import enoch_week1_fixtures
        except ImportError:  # pragma: no cover - direct-script import path.
            import enoch_week1_fixtures  # type: ignore[no-redef]

        fixture_report_fingerprint = enoch_week1_fixtures.validate_report(
            fixture_report
        )
    except enoch_week1_fixtures.FixtureError as exc:
        raise ProtocolError(f"invalid W1.3 fixture report: {exc}") from exc
    fixture_failures = _require_nonnegative_int(
        fixture_report["failure_count"], "fixture failure count"
    )
    fixture_source_files_sha256 = _require_sha256(
        fixture_report["source_files_sha256"], "fixture source identity"
    )
    rule = comparison["development_rule"]
    if rule is None:
        raise ProtocolError("W1.2 comparison has no predeclared development rule")
    validate_development_rule(rule)
    failures = _development_rule_failures(rule, merged["metrics"])
    if fixture_failures:
        failures.insert(0, "fixture-failures-nonzero")
    return {
        "arm_id": comparison["subject_id"],
        "comparison_protocol_fingerprint": comparison_fingerprint,
        "decision": "advance-to-w1.3" if not failures else "stop-and-record",
        "development_rule_sha256": canonical_json_sha256(rule),
        "fixture_failure_count": fixture_failures,
        "fixture_record_count": len(fixture_report["records"]),
        "fixture_report_fingerprint": fixture_report_fingerprint,
        "fixture_source_files_sha256": fixture_source_files_sha256,
        "manifest_kind": "enoch-week1-w1.3-advancement-decision",
        "manifest_version": MANIFEST_VERSION,
        "merged_result_fingerprint": merged_fingerprint,
        "protocol_fingerprint": comparison["protocol_fingerprint"],
        "reasons": failures or ["predeclared-rule-met-and-correctness-clean"],
    }


def build_w1_3_advancement_decision(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    merged: Mapping[str, Any],
    *,
    fixture_report: Mapping[str, Any],
) -> dict[str, Any]:
    body = _w1_3_advancement_body(
        protocol,
        comparison,
        merged,
        fixture_report,
    )
    return _with_fingerprint(body, "advancement_decision_fingerprint")


def validate_w1_3_advancement_decision(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    merged: Mapping[str, Any],
    fixture_report: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    if not isinstance(decision, Mapping):
        raise ProtocolError("W1.3 advancement decision must be an object")
    _require_exact_keys(
        decision,
        {
            "advancement_decision_fingerprint",
            "arm_id",
            "comparison_protocol_fingerprint",
            "decision",
            "development_rule_sha256",
            "fixture_failure_count",
            "fixture_record_count",
            "fixture_report_fingerprint",
            "fixture_source_files_sha256",
            "manifest_kind",
            "manifest_version",
            "merged_result_fingerprint",
            "protocol_fingerprint",
            "reasons",
        },
        "W1.3 advancement decision",
    )
    expected_body = _w1_3_advancement_body(
        protocol,
        comparison,
        merged,
        fixture_report,
    )
    actual_body = dict(decision)
    actual_body.pop("advancement_decision_fingerprint")
    if actual_body != expected_body:
        raise ProtocolError("W1.3 advancement decision does not reconstruct from its evidence")
    return _validate_fingerprint(
        decision, "advancement_decision_fingerprint", "W1.3 advancement decision"
    )


def _validate_serving_envelope(envelope: Mapping[str, Any]) -> None:
    if not isinstance(envelope, Mapping):
        raise ProtocolError("serving envelope must be an object")
    _require_exact_keys(
        envelope,
        {
            "candidate_completed_worlds_mean_min",
            "candidate_p50_latency_ms_max",
            "candidate_p95_latency_ms_max",
        },
        "serving envelope",
    )
    for field in envelope:
        _require_finite_number(envelope[field], f"serving envelope {field}", nonnegative=True)
    if envelope["candidate_p50_latency_ms_max"] > envelope["candidate_p95_latency_ms_max"]:
        raise ProtocolError("serving envelope p50 maximum cannot exceed its p95 maximum")


def _validate_w1_4_candidate_dependency(
    protocol: Mapping[str, Any], decision: Mapping[str, Any] | None
) -> str:
    """Validate the post-run W1.4 decision without an import cycle."""

    if decision is None:
        raise ProtocolError(
            "W1.5 qualification requires the eligible W1.4 candidate decision"
        )
    if not isinstance(decision, Mapping):
        raise ProtocolError("W1.4 candidate decision must be an object")
    try:
        try:
            from . import enoch_week1_campaign
        except ImportError:  # pragma: no cover - direct-script import path.
            import enoch_week1_campaign  # type: ignore[no-redef]

        fingerprint = enoch_week1_campaign.validate_w1_4_candidate_decision(
            protocol, decision
        )
        if decision["decision"] != "eligible-for-qualification":
            raise ProtocolError(
                "W1.5 qualification requires an eligible W1.4 candidate decision"
            )
        return fingerprint
    except ProtocolError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid W1.4 candidate decision: {exc}") from exc


def build_w1_5_qualification_manifest(
    protocol: Mapping[str, Any],
    *,
    candidate_fingerprint: str,
    control_fingerprint: str,
    evaluator_fingerprint: str,
    environment_fingerprint: str,
    configuration_fingerprints: Mapping[str, str],
    serving_envelope: Mapping[str, Any],
    intended_equal_byte_identical: bool,
    w1_4_candidate_decision: Mapping[str, Any] | None = None,
    shard_count: int = 8,
    required_style_metrics: Iterable[str] = WEEK1_STYLE_METRICS,
) -> dict[str, Any]:
    """Freeze the complete W1.5 product-budget qualification matrix."""

    protocol_fingerprint = validate_protocol(protocol)
    for value, label in (
        (candidate_fingerprint, "candidate fingerprint"),
        (control_fingerprint, "control fingerprint"),
        (evaluator_fingerprint, "evaluator fingerprint"),
        (environment_fingerprint, "environment fingerprint"),
    ):
        _require_sha256(value, label)
    candidate_decision_fingerprint = _validate_w1_4_candidate_dependency(
        protocol, w1_4_candidate_decision
    )
    assert w1_4_candidate_decision is not None
    for field, expected in (
        ("candidate_fingerprint", candidate_fingerprint),
        ("control_fingerprint", control_fingerprint),
        ("evaluator_fingerprint", evaluator_fingerprint),
        ("environment_fingerprint", environment_fingerprint),
    ):
        if w1_4_candidate_decision[field] != expected:
            raise ProtocolError(
                f"W1.5 {field} does not match the eligible W1.4 candidate decision"
            )
    if not isinstance(intended_equal_byte_identical, bool):
        raise ProtocolError("intended/equal byte-identity declaration must be boolean")
    if intended_equal_byte_identical:
        raise ProtocolError(
            "W1.5 equal-compute is a distinct fixed-work comparison and cannot be omitted"
        )
    expected_namespaces = {entry["namespace"] for entry in QUALIFICATION_MATRIX}
    if set(configuration_fingerprints) != expected_namespaces:
        raise ProtocolError(
            "qualification configuration hashes must cover every declared matrix namespace"
        )
    for namespace, digest in configuration_fingerprints.items():
        _require_sha256(digest, f"qualification configuration {namespace}")
    _validate_serving_envelope(serving_envelope)
    style_metrics = list(required_style_metrics)
    active_matrix = copy.deepcopy(list(QUALIFICATION_MATRIX))
    comparisons = [
        build_comparison_protocol_manifest(
            protocol,
            phase="W1.5",
            comparison_id=f"qual-{entry['comparison_id']}",
            subject_id="enoch-1",
            seed_namespace=entry["namespace"],
            pair_count=entry["pair_count"],
            shard_count=shard_count,
            candidate_fingerprint=candidate_fingerprint,
            control_fingerprint=control_fingerprint,
            evaluator_fingerprint=evaluator_fingerprint,
            environment_fingerprint=environment_fingerprint,
            configuration_fingerprint=configuration_fingerprints[entry["namespace"]],
            required_style_metrics=style_metrics,
        )
        for entry in active_matrix
    ]
    body = {
        "active_matrix": active_matrix,
        "candidate_fingerprint": candidate_fingerprint,
        "comparisons": comparisons,
        "configuration_fingerprints": {
            namespace: configuration_fingerprints[namespace]
            for namespace in sorted(configuration_fingerprints)
        },
        "control_fingerprint": control_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "evaluator_fingerprint": evaluator_fingerprint,
        "intended_equal_byte_identical": intended_equal_byte_identical,
        "manifest_kind": "enoch-week1-w1.5-qualification-manifest",
        "manifest_version": MANIFEST_VERSION,
        "matrix_sha256": QUALIFICATION_MATRIX_SHA256,
        "pair_count": sum(entry["pair_count"] for entry in active_matrix),
        "protocol_fingerprint": protocol_fingerprint,
        "seed_registry_sha256": protocol["seed_registry_sha256"],
        "serving_envelope": dict(serving_envelope),
        "thresholds": copy.deepcopy(QUALIFICATION_THRESHOLDS),
        "thresholds_sha256": QUALIFICATION_THRESHOLDS_SHA256,
        "w1_4_candidate_decision": copy.deepcopy(dict(w1_4_candidate_decision)),
        "w1_4_candidate_decision_fingerprint": candidate_decision_fingerprint,
    }
    manifest = _with_fingerprint(body, "qualification_manifest_fingerprint")
    validate_w1_5_qualification_manifest(protocol, manifest)
    return manifest


def validate_w1_5_qualification_manifest(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str:
    protocol_fingerprint = validate_protocol(protocol)
    if not isinstance(manifest, Mapping):
        raise ProtocolError("W1.5 qualification manifest must be an object")
    _require_exact_keys(
        manifest,
        {
            "active_matrix",
            "candidate_fingerprint",
            "comparisons",
            "configuration_fingerprints",
            "control_fingerprint",
            "environment_fingerprint",
            "evaluator_fingerprint",
            "intended_equal_byte_identical",
            "manifest_kind",
            "manifest_version",
            "matrix_sha256",
            "pair_count",
            "protocol_fingerprint",
            "qualification_manifest_fingerprint",
            "seed_registry_sha256",
            "serving_envelope",
            "thresholds",
            "thresholds_sha256",
            "w1_4_candidate_decision",
            "w1_4_candidate_decision_fingerprint",
        },
        "W1.5 qualification manifest",
    )
    if manifest["manifest_kind"] != "enoch-week1-w1.5-qualification-manifest":
        raise ProtocolError("unexpected W1.5 qualification manifest kind")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported W1.5 qualification manifest version")
    if manifest["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("qualification protocol fingerprint mismatch")
    if manifest["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise ProtocolError("qualification seed registry mismatch")
    for field in (
        "candidate_fingerprint",
        "control_fingerprint",
        "environment_fingerprint",
        "evaluator_fingerprint",
    ):
        _require_sha256(manifest[field], field)
    candidate_decision_fingerprint = _validate_w1_4_candidate_dependency(
        protocol, manifest["w1_4_candidate_decision"]
    )
    if (
        manifest["w1_4_candidate_decision_fingerprint"]
        != candidate_decision_fingerprint
    ):
        raise ProtocolError("qualification W1.4 candidate-decision fingerprint mismatch")
    for field in (
        "candidate_fingerprint",
        "control_fingerprint",
        "environment_fingerprint",
        "evaluator_fingerprint",
    ):
        if manifest["w1_4_candidate_decision"][field] != manifest[field]:
            raise ProtocolError(
                f"qualification {field} differs from its W1.4 candidate decision"
            )
    if manifest["matrix_sha256"] != QUALIFICATION_MATRIX_SHA256:
        raise ProtocolError("qualification matrix declaration changed")
    if manifest["thresholds"] != QUALIFICATION_THRESHOLDS:
        raise ProtocolError("qualification thresholds changed")
    if manifest["thresholds_sha256"] != QUALIFICATION_THRESHOLDS_SHA256:
        raise ProtocolError("qualification threshold hash mismatch")
    if not isinstance(manifest["intended_equal_byte_identical"], bool):
        raise ProtocolError("qualification byte-identity declaration must be boolean")
    if manifest["intended_equal_byte_identical"]:
        raise ProtocolError(
            "qualification cannot omit the distinct fixed-work equal-compute comparison"
        )
    expected_matrix = copy.deepcopy(list(QUALIFICATION_MATRIX))
    if manifest["active_matrix"] != expected_matrix:
        raise ProtocolError("qualification active matrix changed")
    expected_pair_count = sum(entry["pair_count"] for entry in expected_matrix)
    if manifest["pair_count"] != expected_pair_count or expected_pair_count != 3_300:
        raise ProtocolError("qualification matrix pair count must be exactly 3,300")
    configurations = manifest["configuration_fingerprints"]
    expected_namespaces = {entry["namespace"] for entry in QUALIFICATION_MATRIX}
    if not isinstance(configurations, Mapping) or set(configurations) != expected_namespaces:
        raise ProtocolError("qualification configuration hash set is incomplete")
    if list(configurations) != sorted(configurations):
        raise ProtocolError("qualification configuration hashes must be sorted")
    for namespace, digest in configurations.items():
        _require_sha256(digest, f"qualification configuration {namespace}")
    _validate_serving_envelope(manifest["serving_envelope"])
    comparisons = manifest["comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != len(expected_matrix):
        raise ProtocolError("qualification comparison set is incomplete")
    for entry, comparison in zip(expected_matrix, comparisons):
        validate_comparison_protocol_manifest(protocol, comparison)
        if comparison["phase"] != "W1.5":
            raise ProtocolError("qualification comparison has wrong phase")
        if comparison["comparison_id"] != f"qual-{entry['comparison_id']}":
            raise ProtocolError("qualification comparison id mismatch")
        if comparison["seed_namespace"] != entry["namespace"]:
            raise ProtocolError("qualification comparison namespace mismatch")
        if comparison["pair_count"] != entry["pair_count"]:
            raise ProtocolError("qualification comparison pair count mismatch")
        for field in (
            "candidate_fingerprint",
            "control_fingerprint",
            "environment_fingerprint",
            "evaluator_fingerprint",
        ):
            if comparison[field] != manifest[field]:
                raise ProtocolError(f"qualification comparison {field} mismatch")
        if comparison["configuration_fingerprint"] != configurations[entry["namespace"]]:
            raise ProtocolError("qualification comparison configuration mismatch")
    return _validate_fingerprint(
        manifest, "qualification_manifest_fingerprint", "W1.5 qualification manifest"
    )


def _w1_5_qualification_decision_body(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    merged_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_fingerprint = validate_w1_5_qualification_manifest(protocol, manifest)
    expected_ids = [comparison["comparison_id"] for comparison in manifest["comparisons"]]
    if set(merged_results) != set(expected_ids):
        raise ProtocolError("qualification decision requires one merged result per comparison")
    summaries: list[dict[str, Any]] = []
    robustness_weighted_sum = 0.0
    robustness_pairs = 0
    failures: list[str] = []
    matrix_by_protocol_id = {
        f"qual-{entry['comparison_id']}": entry for entry in manifest["active_matrix"]
    }
    for comparison in manifest["comparisons"]:
        comparison_id = comparison["comparison_id"]
        merged = merged_results[comparison_id]
        merged_fingerprint = validate_merged_result(protocol, comparison, merged)
        metrics = merged["metrics"]
        matrix_entry = matrix_by_protocol_id[comparison_id]
        serving_envelope_applicable = matrix_entry["category"] != "crossplay"
        summary = {
            "category": matrix_entry["category"],
            "comparison_id": comparison_id,
            "level_utility_estimate": metrics["level_utility"]["estimate"],
            "level_utility_lower_95": metrics["level_utility"][
                "paired_bootstrap_lower_95"
            ],
            "merged_result_fingerprint": merged_fingerprint,
            "pair_count": metrics["pair_count"],
            "point_margin_estimate": metrics["point_margin_estimate"],
            "robustness": matrix_entry["robustness"],
            "serving_envelope_applicable": serving_envelope_applicable,
            "win_rate_estimate": metrics["win_rate_estimate"],
        }
        summaries.append(summary)
        if matrix_entry["robustness"]:
            robustness_weighted_sum += (
                metrics["level_utility"]["estimate"] * metrics["pair_count"]
            )
            robustness_pairs += metrics["pair_count"]
            if (
                metrics["level_utility"]["estimate"]
                <= QUALIFICATION_THRESHOLDS["stratum_level_utility_exclusive_min"]
            ):
                failures.append(f"robustness-stratum-at-or-below-stop:{comparison_id}")
        else:
            if (
                metrics["level_utility"]["estimate"]
                <= QUALIFICATION_THRESHOLDS[
                    "budget_level_utility_estimate_exclusive_min"
                ]
            ):
                failures.append(f"budget-level-utility-not-positive:{comparison_id}")
            if (
                metrics["level_utility"]["paired_bootstrap_lower_95"]
                < QUALIFICATION_THRESHOLDS[
                    "budget_level_utility_lower_95_inclusive_min"
                ]
            ):
                failures.append(f"budget-lower-95-below-threshold:{comparison_id}")
            if (
                metrics["point_margin_estimate"]
                <= QUALIFICATION_THRESHOLDS[
                    "budget_point_margin_estimate_exclusive_min"
                ]
            ):
                failures.append(f"budget-point-margin-not-positive:{comparison_id}")
            if (
                metrics["win_rate_estimate"]
                < QUALIFICATION_THRESHOLDS[
                    "budget_win_rate_estimate_inclusive_min"
                ]
            ):
                failures.append(f"budget-win-rate-below-threshold:{comparison_id}")
        if serving_envelope_applicable:
            envelope = manifest["serving_envelope"]
            if (
                metrics["candidate_latency_ms"]["p50"]
                > envelope["candidate_p50_latency_ms_max"]
            ):
                failures.append(
                    f"candidate-p50-latency-outside-envelope:{comparison_id}"
                )
            if (
                metrics["candidate_latency_ms"]["p95"]
                > envelope["candidate_p95_latency_ms_max"]
            ):
                failures.append(
                    f"candidate-p95-latency-outside-envelope:{comparison_id}"
                )
            if (
                metrics["candidate_completed_worlds_mean"]
                < envelope["candidate_completed_worlds_mean_min"]
            ):
                failures.append(
                    f"completed-worlds-outside-envelope:{comparison_id}"
                )
        for counter in QUALIFICATION_THRESHOLDS["zero_failure_counters"]:
            if metrics["failure_counters"][counter] != 0:
                failures.append(f"nonzero-failure-counter:{comparison_id}:{counter}")
    if robustness_pairs == 0:
        raise ProtocolError("qualification matrix contains no robustness pairs")
    pooled_robustness = robustness_weighted_sum / robustness_pairs
    if (
        pooled_robustness
        < QUALIFICATION_THRESHOLDS["pooled_robustness_level_utility_inclusive_min"]
    ):
        failures.append("pooled-robustness-level-utility-negative")
    return {
        "comparison_summaries": summaries,
        "decision": "eligible-for-locked-gate" if not failures else "reject-candidate",
        "manifest_kind": "enoch-week1-w1.5-qualification-decision",
        "manifest_version": MANIFEST_VERSION,
        "pooled_robustness_level_utility": pooled_robustness,
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "qualification_manifest_fingerprint": manifest_fingerprint,
        "reasons": failures or ["all-predeclared-qualification-criteria-passed"],
        "thresholds_sha256": QUALIFICATION_THRESHOLDS_SHA256,
        "w1_4_candidate_decision_fingerprint": manifest[
            "w1_4_candidate_decision_fingerprint"
        ],
    }


def build_w1_5_qualification_decision(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    merged_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body = _w1_5_qualification_decision_body(protocol, manifest, merged_results)
    return _with_fingerprint(body, "qualification_decision_fingerprint")


def validate_w1_5_qualification_decision_artifact(decision: Mapping[str, Any]) -> str:
    if not isinstance(decision, Mapping):
        raise ProtocolError("W1.5 qualification decision must be an object")
    _require_exact_keys(
        decision,
        {
            "comparison_summaries",
            "decision",
            "manifest_kind",
            "manifest_version",
            "pooled_robustness_level_utility",
            "protocol_fingerprint",
            "qualification_decision_fingerprint",
            "qualification_manifest_fingerprint",
            "reasons",
            "thresholds_sha256",
            "w1_4_candidate_decision_fingerprint",
        },
        "W1.5 qualification decision",
    )
    if decision["manifest_kind"] != "enoch-week1-w1.5-qualification-decision":
        raise ProtocolError("unexpected W1.5 qualification decision kind")
    if decision["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported W1.5 qualification decision version")
    if decision["decision"] not in {"eligible-for-locked-gate", "reject-candidate"}:
        raise ProtocolError("invalid W1.5 qualification decision")
    for field in (
        "protocol_fingerprint",
        "qualification_manifest_fingerprint",
        "thresholds_sha256",
        "w1_4_candidate_decision_fingerprint",
    ):
        _require_sha256(decision[field], field)
    if decision["thresholds_sha256"] != QUALIFICATION_THRESHOLDS_SHA256:
        raise ProtocolError("qualification decision threshold hash mismatch")
    _require_finite_number(
        decision["pooled_robustness_level_utility"], "pooled robustness estimate"
    )
    if not isinstance(decision["comparison_summaries"], list) or not isinstance(
        decision["reasons"], list
    ):
        raise ProtocolError("qualification decision summaries and reasons must be lists")
    if not decision["comparison_summaries"] or not decision["reasons"]:
        raise ProtocolError("qualification decision must retain summaries and reasons")
    seen_comparisons: set[str] = set()
    for summary in decision["comparison_summaries"]:
        if not isinstance(summary, Mapping):
            raise ProtocolError("qualification comparison summary must be an object")
        _require_exact_keys(
            summary,
            {
                "category",
                "comparison_id",
                "level_utility_estimate",
                "level_utility_lower_95",
                "merged_result_fingerprint",
                "pair_count",
                "point_margin_estimate",
                "robustness",
                "serving_envelope_applicable",
                "win_rate_estimate",
            },
            "qualification comparison summary",
        )
        comparison_id = _require_identifier(summary["comparison_id"], "comparison id")
        if comparison_id in seen_comparisons:
            raise ProtocolError("qualification comparison summaries contain duplicates")
        seen_comparisons.add(comparison_id)
        _require_identifier(summary["category"], "qualification category")
        _require_sha256(summary["merged_result_fingerprint"], "merged result fingerprint")
        _require_nonnegative_int(summary["pair_count"], "qualification pair count")
        if summary["pair_count"] == 0 or not isinstance(summary["robustness"], bool):
            raise ProtocolError("qualification summary pair count or robustness flag is invalid")
        if not isinstance(summary["serving_envelope_applicable"], bool):
            raise ProtocolError(
                "qualification serving-envelope applicability must be boolean"
            )
        if summary["serving_envelope_applicable"] != (
            summary["category"] != "crossplay"
        ):
            raise ProtocolError(
                "qualification serving-envelope applicability/category mismatch"
            )
        for field in (
            "level_utility_estimate",
            "level_utility_lower_95",
            "point_margin_estimate",
            "win_rate_estimate",
        ):
            _require_finite_number(summary[field], f"qualification summary {field}")
    if not all(isinstance(reason, str) and reason for reason in decision["reasons"]):
        raise ProtocolError("qualification reasons must be nonempty strings")
    return _validate_fingerprint(
        decision, "qualification_decision_fingerprint", "W1.5 qualification decision"
    )


def validate_w1_5_qualification_decision(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    merged_results: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    validate_w1_5_qualification_decision_artifact(decision)
    expected_body = _w1_5_qualification_decision_body(protocol, manifest, merged_results)
    actual_body = dict(decision)
    actual_body.pop("qualification_decision_fingerprint")
    if actual_body != expected_body:
        raise ProtocolError("qualification decision does not reconstruct from raw results")
    return decision["qualification_decision_fingerprint"]


POLICY_IDENTITY_FIELDS = {
    "binary_sha256",
    "configuration_sha256",
    "model_sha256",
    "source_sha256",
}
EVALUATOR_IDENTITY_FIELDS = {
    "binary_sha256",
    "configuration_sha256",
    "source_sha256",
}


def _validate_frozen_identity(
    identity: Mapping[str, Any], expected_fields: set[str], label: str
) -> str:
    if not isinstance(identity, Mapping):
        raise ProtocolError(f"{label} identity must be an object")
    _require_exact_keys(identity, expected_fields, f"{label} identity")
    for field in expected_fields:
        _require_sha256(identity[field], f"{label} {field}")
    return canonical_json_sha256(identity)


def build_frozen_policy_identity(
    *, source_sha256: str, binary_sha256: str, model_sha256: str, configuration_sha256: str
) -> dict[str, str]:
    identity = {
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "model_sha256": model_sha256,
        "source_sha256": source_sha256,
    }
    _validate_frozen_identity(identity, POLICY_IDENTITY_FIELDS, "policy")
    return identity


def build_frozen_evaluator_identity(
    *, source_sha256: str, binary_sha256: str, configuration_sha256: str
) -> dict[str, str]:
    identity = {
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "source_sha256": source_sha256,
    }
    _validate_frozen_identity(identity, EVALUATOR_IDENTITY_FIELDS, "evaluator")
    return identity


def _validate_locked_secondary_requirements(requirements: Mapping[str, Any]) -> None:
    validate_development_rule(requirements)
    for field in (
        "minimum_point_margin_estimate",
        "minimum_win_rate_estimate",
        "maximum_candidate_p95_latency_ms",
        "minimum_candidate_completed_worlds_mean",
    ):
        if requirements[field] is None:
            raise ProtocolError(f"locked secondary requirements must declare {field}")


def _build_locked_gate_manifest(
    protocol: Mapping[str, Any],
    *,
    gate_role: str,
    pair_count: int,
    shard_count: int,
    candidate_identity: Mapping[str, Any],
    control_identity: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
    environment_sha256: str,
    qualification_decision_fingerprint: str,
    qualification_intended_configuration_fingerprint: str,
    locked_configuration_fingerprint: str,
    secondary_requirements: Mapping[str, Any],
    required_style_metrics: Iterable[str],
) -> dict[str, Any]:
    protocol_fingerprint = validate_protocol(protocol)
    if gate_role not in {"primary", "confirmation"}:
        raise ProtocolError("locked gate role must be primary or confirmation")
    candidate_fingerprint = _validate_frozen_identity(
        candidate_identity, POLICY_IDENTITY_FIELDS, "candidate"
    )
    control_fingerprint = _validate_frozen_identity(
        control_identity, POLICY_IDENTITY_FIELDS, "control"
    )
    evaluator_fingerprint = _validate_frozen_identity(
        evaluator_identity, EVALUATOR_IDENTITY_FIELDS, "evaluator"
    )
    environment_sha256 = _require_sha256(environment_sha256, "locked environment hash")
    qualification_decision_fingerprint = _require_sha256(
        qualification_decision_fingerprint, "qualification decision fingerprint"
    )
    qualification_intended_configuration_fingerprint = _require_sha256(
        qualification_intended_configuration_fingerprint,
        "qualification intended configuration fingerprint",
    )
    locked_configuration_fingerprint = _require_sha256(
        locked_configuration_fingerprint, "locked configuration fingerprint"
    )
    _validate_locked_secondary_requirements(secondary_requirements)
    phase = "W1.6" if gate_role == "primary" else "W1.7"
    namespace = "locked/gate-1" if gate_role == "primary" else "locked/confirmation"
    frozen_lock = {
        "arm_registry_sha256": ARM_REGISTRY_SHA256,
        "candidate_identity": copy.deepcopy(dict(candidate_identity)),
        "control_identity": copy.deepcopy(dict(control_identity)),
        "environment_sha256": environment_sha256,
        "evaluator_identity": copy.deepcopy(dict(evaluator_identity)),
        "protocol_fingerprint": protocol_fingerprint,
        "qualification_decision_fingerprint": qualification_decision_fingerprint,
        "qualification_intended_configuration_fingerprint": (
            qualification_intended_configuration_fingerprint
        ),
        "locked_configuration_fingerprint": locked_configuration_fingerprint,
        "seed_registry_sha256": protocol["seed_registry_sha256"],
    }
    frozen_lock_sha256 = canonical_json_sha256(frozen_lock)
    comparison = build_comparison_protocol_manifest(
        protocol,
        phase=phase,
        comparison_id=f"locked-{gate_role}",
        subject_id="enoch-1",
        seed_namespace=namespace,
        pair_count=pair_count,
        shard_count=shard_count,
        candidate_fingerprint=candidate_fingerprint,
        control_fingerprint=control_fingerprint,
        evaluator_fingerprint=evaluator_fingerprint,
        environment_fingerprint=environment_sha256,
        configuration_fingerprint=locked_configuration_fingerprint,
        required_style_metrics=required_style_metrics,
    )
    body = {
        "comparison": comparison,
        "frozen_lock": frozen_lock,
        "frozen_lock_sha256": frozen_lock_sha256,
        "gate_role": gate_role,
        "manifest_kind": "enoch-week1-locked-gate-manifest",
        "manifest_version": MANIFEST_VERSION,
        "phase": phase,
        "secondary_requirements": copy.deepcopy(dict(secondary_requirements)),
        "superiority_rule": copy.deepcopy(LOCKED_SUPERIORITY_RULE),
        "superiority_rule_sha256": LOCKED_SUPERIORITY_RULE_SHA256,
    }
    manifest = _with_fingerprint(body, "locked_gate_manifest_fingerprint")
    validate_locked_gate_manifest(protocol, manifest)
    return manifest


def build_locked_gate_manifests(
    protocol: Mapping[str, Any],
    qualification_decision: Mapping[str, Any],
    *,
    qualification_manifest: Mapping[str, Any],
    qualification_merged_results: Mapping[str, Mapping[str, Any]],
    candidate_identity: Mapping[str, Any],
    control_identity: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
    environment_sha256: str,
    locked_configuration_fingerprint: str,
    secondary_requirements: Mapping[str, Any],
    pair_count: int = 2_000,
    shard_count: int = 10,
    required_style_metrics: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Predeclare both disjoint locked gates with an identical frozen lock."""

    qualification_fingerprint = validate_w1_5_qualification_decision(
        protocol,
        qualification_manifest,
        qualification_merged_results,
        qualification_decision,
    )
    if qualification_decision["decision"] != "eligible-for-locked-gate":
        raise ProtocolError("locked gates require a passing W1.5 qualification decision")
    qualification_bindings = {
        "candidate_fingerprint": _validate_frozen_identity(
            candidate_identity, POLICY_IDENTITY_FIELDS, "candidate"
        ),
        "control_fingerprint": _validate_frozen_identity(
            control_identity, POLICY_IDENTITY_FIELDS, "control"
        ),
        "evaluator_fingerprint": _validate_frozen_identity(
            evaluator_identity, EVALUATOR_IDENTITY_FIELDS, "evaluator"
        ),
        "environment_fingerprint": _require_sha256(
            environment_sha256, "locked environment hash"
        ),
    }
    for field, expected in qualification_bindings.items():
        if qualification_manifest[field] != expected:
            raise ProtocolError(
                f"locked {field} differs from the qualified candidate protocol"
            )
    qualification_intended_configuration_fingerprint = qualification_manifest[
        "configuration_fingerprints"
    ]["qual/intended"]
    locked_configuration_fingerprint = _require_sha256(
        locked_configuration_fingerprint, "locked configuration fingerprint"
    )
    if locked_configuration_fingerprint == qualification_intended_configuration_fingerprint:
        raise ProtocolError(
            "locked standard-scenario configuration must differ from intended qualification"
        )
    style_metrics = list(required_style_metrics)
    primary = _build_locked_gate_manifest(
        protocol,
        gate_role="primary",
        pair_count=pair_count,
        shard_count=shard_count,
        candidate_identity=candidate_identity,
        control_identity=control_identity,
        evaluator_identity=evaluator_identity,
        environment_sha256=environment_sha256,
        qualification_decision_fingerprint=qualification_fingerprint,
        qualification_intended_configuration_fingerprint=(
            qualification_intended_configuration_fingerprint
        ),
        locked_configuration_fingerprint=locked_configuration_fingerprint,
        secondary_requirements=secondary_requirements,
        required_style_metrics=style_metrics,
    )
    confirmation = _build_locked_gate_manifest(
        protocol,
        gate_role="confirmation",
        pair_count=pair_count,
        shard_count=shard_count,
        candidate_identity=candidate_identity,
        control_identity=control_identity,
        evaluator_identity=evaluator_identity,
        environment_sha256=environment_sha256,
        qualification_decision_fingerprint=qualification_fingerprint,
        qualification_intended_configuration_fingerprint=(
            qualification_intended_configuration_fingerprint
        ),
        locked_configuration_fingerprint=locked_configuration_fingerprint,
        secondary_requirements=secondary_requirements,
        required_style_metrics=style_metrics,
    )
    validate_locked_gate_pair(protocol, primary, confirmation)
    return primary, confirmation


def validate_locked_gate_manifest(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str:
    _assert_static_control_contracts()
    protocol_fingerprint = validate_protocol(protocol)
    if not isinstance(manifest, Mapping):
        raise ProtocolError("locked gate manifest must be an object")
    _require_exact_keys(
        manifest,
        {
            "comparison",
            "frozen_lock",
            "frozen_lock_sha256",
            "gate_role",
            "locked_gate_manifest_fingerprint",
            "manifest_kind",
            "manifest_version",
            "phase",
            "secondary_requirements",
            "superiority_rule",
            "superiority_rule_sha256",
        },
        "locked gate manifest",
    )
    if manifest["manifest_kind"] != "enoch-week1-locked-gate-manifest":
        raise ProtocolError("unexpected locked gate manifest kind")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported locked gate manifest version")
    role = manifest["gate_role"]
    expected_phase = "W1.6" if role == "primary" else "W1.7" if role == "confirmation" else None
    if expected_phase is None or manifest["phase"] != expected_phase:
        raise ProtocolError("locked gate role and phase do not agree")
    if manifest["superiority_rule"] != LOCKED_SUPERIORITY_RULE:
        raise ProtocolError("locked superiority rule changed")
    if (
        manifest["superiority_rule_sha256"] != LOCKED_SUPERIORITY_RULE_SHA256
        or manifest["superiority_rule_sha256"]
        != canonical_json_sha256(manifest["superiority_rule"])
    ):
        raise ProtocolError("locked superiority rule hash mismatch")
    _validate_locked_secondary_requirements(manifest["secondary_requirements"])
    frozen_lock = manifest["frozen_lock"]
    if not isinstance(frozen_lock, Mapping):
        raise ProtocolError("frozen lock must be an object")
    _require_exact_keys(
        frozen_lock,
        {
            "arm_registry_sha256",
            "candidate_identity",
            "control_identity",
            "environment_sha256",
            "evaluator_identity",
            "locked_configuration_fingerprint",
            "protocol_fingerprint",
            "qualification_decision_fingerprint",
            "qualification_intended_configuration_fingerprint",
            "seed_registry_sha256",
        },
        "frozen lock",
    )
    candidate_fingerprint = _validate_frozen_identity(
        frozen_lock["candidate_identity"], POLICY_IDENTITY_FIELDS, "candidate"
    )
    control_fingerprint = _validate_frozen_identity(
        frozen_lock["control_identity"], POLICY_IDENTITY_FIELDS, "control"
    )
    evaluator_fingerprint = _validate_frozen_identity(
        frozen_lock["evaluator_identity"], EVALUATOR_IDENTITY_FIELDS, "evaluator"
    )
    _require_sha256(frozen_lock["environment_sha256"], "locked environment hash")
    _require_sha256(
        frozen_lock["qualification_decision_fingerprint"],
        "qualification decision fingerprint",
    )
    _require_sha256(
        frozen_lock["qualification_intended_configuration_fingerprint"],
        "qualification intended configuration fingerprint",
    )
    _require_sha256(
        frozen_lock["locked_configuration_fingerprint"],
        "locked configuration fingerprint",
    )
    if (
        frozen_lock["locked_configuration_fingerprint"]
        == frozen_lock["qualification_intended_configuration_fingerprint"]
    ):
        raise ProtocolError(
            "locked standard-scenario configuration reuses intended qualification"
        )
    if frozen_lock["arm_registry_sha256"] != ARM_REGISTRY_SHA256:
        raise ProtocolError("locked gate arm registry mismatch")
    if frozen_lock["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("locked gate seed protocol mismatch")
    if frozen_lock["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise ProtocolError("locked gate seed registry mismatch")
    if manifest["frozen_lock_sha256"] != canonical_json_sha256(frozen_lock):
        raise ProtocolError("frozen candidate/evaluator lock hash mismatch")
    comparison = manifest["comparison"]
    validate_comparison_protocol_manifest(protocol, comparison)
    expected_namespace = "locked/gate-1" if role == "primary" else "locked/confirmation"
    expected_comparison_id = f"locked-{role}"
    if comparison["phase"] != expected_phase or comparison["seed_namespace"] != expected_namespace:
        raise ProtocolError("locked gate comparison uses the wrong phase or seed namespace")
    if comparison["comparison_id"] != expected_comparison_id:
        raise ProtocolError("locked gate comparison id does not match its gate role")
    if comparison["subject_id"] != "enoch-1":
        raise ProtocolError("locked gate comparison subject must be enoch-1")
    if comparison["development_rule"] is not None:
        raise ProtocolError("locked gate comparison cannot carry a second development rule")
    expected_bindings = {
        "candidate_fingerprint": candidate_fingerprint,
        "configuration_fingerprint": frozen_lock["locked_configuration_fingerprint"],
        "control_fingerprint": control_fingerprint,
        "environment_fingerprint": frozen_lock["environment_sha256"],
        "evaluator_fingerprint": evaluator_fingerprint,
    }
    for field, expected in expected_bindings.items():
        if comparison[field] != expected:
            raise ProtocolError(f"locked comparison {field} escaped the frozen lock")
    return _validate_fingerprint(
        manifest, "locked_gate_manifest_fingerprint", "locked gate manifest"
    )


def validate_locked_gate_pair(
    protocol: Mapping[str, Any],
    primary: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> tuple[str, str]:
    primary_fingerprint = validate_locked_gate_manifest(protocol, primary)
    confirmation_fingerprint = validate_locked_gate_manifest(protocol, confirmation)
    if primary["gate_role"] != "primary" or confirmation["gate_role"] != "confirmation":
        raise ProtocolError("locked gate pair must be primary followed by confirmation")
    if primary["frozen_lock_sha256"] != confirmation["frozen_lock_sha256"]:
        raise ProtocolError("locked primary and confirmation do not share the frozen candidate")
    if primary["comparison"]["pair_count"] != confirmation["comparison"]["pair_count"]:
        raise ProtocolError("locked primary and confirmation pair counts differ")
    if primary["secondary_requirements"] != confirmation["secondary_requirements"]:
        raise ProtocolError("locked primary and confirmation secondary rules differ")
    primary_schema = {
        "bootstrap": primary["comparison"]["bootstrap"],
        "development_rule": primary["comparison"]["development_rule"],
        "pair_count": primary["comparison"]["pair_count"],
        "required_style_metrics": primary["comparison"]["required_style_metrics"],
        "shard_count": len(primary["comparison"]["shards"]),
        "subject_id": primary["comparison"]["subject_id"],
    }
    confirmation_schema = {
        "bootstrap": confirmation["comparison"]["bootstrap"],
        "development_rule": confirmation["comparison"]["development_rule"],
        "pair_count": confirmation["comparison"]["pair_count"],
        "required_style_metrics": confirmation["comparison"]["required_style_metrics"],
        "shard_count": len(confirmation["comparison"]["shards"]),
        "subject_id": confirmation["comparison"]["subject_id"],
    }
    if primary_schema != confirmation_schema:
        raise ProtocolError("locked primary and confirmation comparison schemas differ")
    primary_seeds = {
        _protocol_seed(protocol, "locked/gate-1", index)
        for index in primary["comparison"]["seed_indices"]
    }
    confirmation_seeds = {
        _protocol_seed(protocol, "locked/confirmation", index)
        for index in confirmation["comparison"]["seed_indices"]
    }
    if not primary_seeds.isdisjoint(confirmation_seeds):
        raise ProtocolError("locked primary and confirmation seed sets overlap")
    return primary_fingerprint, confirmation_fingerprint


def _locked_gate_decision_body(
    protocol: Mapping[str, Any],
    gate_manifest: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> dict[str, Any]:
    gate_fingerprint = validate_locked_gate_manifest(protocol, gate_manifest)
    comparison = gate_manifest["comparison"]
    merged_fingerprint = validate_merged_result(protocol, comparison, merged)
    failures = _development_rule_failures(
        gate_manifest["secondary_requirements"], merged["metrics"]
    )
    superiority_rule = gate_manifest["superiority_rule"]
    observed_superiority = merged["metrics"]["level_utility"][
        "paired_bootstrap_lower_95"
    ]
    if not observed_superiority > superiority_rule["threshold"]:
        failures.insert(0, "paired-bootstrap-lower-95-not-greater-than-zero")
    return {
        "candidate_fingerprint": comparison["candidate_fingerprint"],
        "decision": "pass" if not failures else "fail",
        "frozen_lock_sha256": gate_manifest["frozen_lock_sha256"],
        "gate_role": gate_manifest["gate_role"],
        "locked_gate_manifest_fingerprint": gate_fingerprint,
        "manifest_kind": "enoch-week1-locked-gate-decision",
        "manifest_version": MANIFEST_VERSION,
        "merged_result_fingerprint": merged_fingerprint,
        "protocol_fingerprint": comparison["protocol_fingerprint"],
        "reasons": failures or ["primary-and-secondary-locked-gate-criteria-passed"],
    }


def build_locked_gate_decision(
    protocol: Mapping[str, Any],
    gate_manifest: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> dict[str, Any]:
    body = _locked_gate_decision_body(protocol, gate_manifest, merged)
    return _with_fingerprint(body, "locked_gate_decision_fingerprint")


def validate_locked_gate_decision_artifact(decision: Mapping[str, Any]) -> str:
    if not isinstance(decision, Mapping):
        raise ProtocolError("locked gate decision must be an object")
    _require_exact_keys(
        decision,
        {
            "candidate_fingerprint",
            "decision",
            "frozen_lock_sha256",
            "gate_role",
            "locked_gate_decision_fingerprint",
            "locked_gate_manifest_fingerprint",
            "manifest_kind",
            "manifest_version",
            "merged_result_fingerprint",
            "protocol_fingerprint",
            "reasons",
        },
        "locked gate decision",
    )
    if decision["manifest_kind"] != "enoch-week1-locked-gate-decision":
        raise ProtocolError("unexpected locked gate decision kind")
    if decision["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported locked gate decision version")
    if decision["gate_role"] not in {"primary", "confirmation"}:
        raise ProtocolError("invalid locked gate decision role")
    if decision["decision"] not in {"pass", "fail"}:
        raise ProtocolError("invalid locked gate decision")
    for field in (
        "candidate_fingerprint",
        "frozen_lock_sha256",
        "locked_gate_manifest_fingerprint",
        "merged_result_fingerprint",
        "protocol_fingerprint",
    ):
        _require_sha256(decision[field], field)
    if not isinstance(decision["reasons"], list) or not decision["reasons"]:
        raise ProtocolError("locked gate decision must record at least one reason")
    return _validate_fingerprint(
        decision, "locked_gate_decision_fingerprint", "locked gate decision"
    )


def validate_locked_gate_decision(
    protocol: Mapping[str, Any],
    gate_manifest: Mapping[str, Any],
    merged: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    validate_locked_gate_decision_artifact(decision)
    expected_body = _locked_gate_decision_body(protocol, gate_manifest, merged)
    actual_body = dict(decision)
    actual_body.pop("locked_gate_decision_fingerprint")
    if actual_body != expected_body:
        raise ProtocolError("locked gate decision does not reconstruct from raw results")
    return decision["locked_gate_decision_fingerprint"]


NO_CANDIDATE_REASONS = (
    "no-survivor",
    "combination-regressed",
    "candidate-prerequisites-incomplete",
    "qualification-failed",
    "primary-locked-gate-failed",
    "confirmation-not-passed",
    "candidate-withdrawn",
)


def build_week1_decision_artifact(
    protocol: Mapping[str, Any],
    *,
    phase_manifests: Sequence[Mapping[str, Any]],
    control_manifest: Mapping[str, Any],
    enoch0_fingerprint: str,
    candidate_fingerprint: str | None,
    qualification_manifest: Mapping[str, Any] | None = None,
    qualification_merged_results: Mapping[str, Mapping[str, Any]] | None = None,
    qualification_decision: Mapping[str, Any] | None = None,
    primary_gate_decision: Mapping[str, Any] | None,
    confirmation_gate_decision: Mapping[str, Any] | None,
    primary_gate_manifest: Mapping[str, Any] | None = None,
    primary_merged_result: Mapping[str, Any] | None = None,
    confirmation_gate_manifest: Mapping[str, Any] | None = None,
    confirmation_merged_result: Mapping[str, Any] | None = None,
    prerequisites_complete: bool,
    no_candidate_reason: str | None = None,
    evidence_fingerprints: Iterable[str] = (),
) -> dict[str, Any]:
    """Record the only Week 1 terminal choice: freeze or retain Enoch-0.

    A freeze changes the downstream yardstick only.  This artifact can never
    authorize a production promotion or deployment.
    """

    protocol_fingerprint = validate_protocol(protocol)
    phase_fingerprints = validate_phase_chain(protocol, phase_manifests)
    control_manifest_fingerprint = validate_w1_0_control_manifest(
        protocol, control_manifest
    )
    enoch0_fingerprint = _require_sha256(enoch0_fingerprint, "Enoch-0 fingerprint")
    frozen_enoch0_fingerprint = canonical_json_sha256(
        control_manifest["policy_identities"]["enoch-0"]
    )
    if enoch0_fingerprint != frozen_enoch0_fingerprint:
        raise ProtocolError("terminal Enoch-0 differs from the frozen W1.0 control")
    if control_manifest_fingerprint not in {
        artifact["sha256"] for artifact in phase_manifests[0]["artifacts"]
    }:
        raise ProtocolError("W1.0 phase does not bind the frozen control manifest")
    if candidate_fingerprint is not None:
        candidate_fingerprint = _require_sha256(
            candidate_fingerprint, "candidate fingerprint"
        )
    if not isinstance(prerequisites_complete, bool):
        raise ProtocolError("candidate prerequisite declaration must be boolean")
    primary_fingerprint: str | None = None
    confirmation_fingerprint: str | None = None
    qualification_fingerprint: str | None = None
    evaluation_control_fingerprint: str | None = None
    qualification_inputs = (
        qualification_manifest,
        qualification_merged_results,
        qualification_decision,
    )
    if any(value is not None for value in qualification_inputs):
        if any(value is None for value in qualification_inputs):
            raise ProtocolError("qualification evidence must include manifest, results, and decision")
        qualification_fingerprint = validate_w1_5_qualification_decision(
            protocol,
            qualification_manifest,
            qualification_merged_results,
            qualification_decision,
        )
        evaluation_control_fingerprint = qualification_manifest[
            "control_fingerprint"
        ]
    if len(phase_manifests) >= 6 and qualification_fingerprint is None:
        raise ProtocolError("W1.5-or-later terminal decisions require typed qualification evidence")
    if primary_gate_decision is not None:
        if primary_gate_manifest is None or primary_merged_result is None:
            raise ProtocolError("primary decision requires its manifest and raw merged result")
        primary_fingerprint = validate_locked_gate_decision(
            protocol,
            primary_gate_manifest,
            primary_merged_result,
            primary_gate_decision,
        )
        if primary_gate_decision["gate_role"] != "primary":
            raise ProtocolError("primary decision artifact has the wrong gate role")
    elif primary_gate_manifest is not None or primary_merged_result is not None:
        raise ProtocolError("primary gate evidence was supplied without a decision")
    if confirmation_gate_decision is not None:
        if confirmation_gate_manifest is None or confirmation_merged_result is None:
            raise ProtocolError(
                "confirmation decision requires its manifest and raw merged result"
            )
        confirmation_fingerprint = validate_locked_gate_decision(
            protocol,
            confirmation_gate_manifest,
            confirmation_merged_result,
            confirmation_gate_decision,
        )
        if confirmation_gate_decision["gate_role"] != "confirmation":
            raise ProtocolError("confirmation decision artifact has the wrong gate role")
    elif confirmation_gate_manifest is not None or confirmation_merged_result is not None:
        raise ProtocolError("confirmation evidence was supplied without a decision")
    if confirmation_gate_decision is not None and primary_gate_decision is None:
        raise ProtocolError("confirmation cannot exist without a primary locked decision")
    if len(phase_manifests) >= 7 and primary_gate_decision is None:
        raise ProtocolError("W1.6-or-later terminal decisions require a primary gate decision")
    if len(phase_manifests) >= 8 and confirmation_gate_decision is None:
        raise ProtocolError("W1.7 terminal decisions require a confirmation decision")
    for gate in (primary_gate_decision, confirmation_gate_decision):
        if gate is None:
            continue
        if gate["protocol_fingerprint"] != protocol_fingerprint:
            raise ProtocolError("locked decision belongs to a different seed protocol")
        if candidate_fingerprint is None or gate["candidate_fingerprint"] != candidate_fingerprint:
            raise ProtocolError("locked decision candidate fingerprint mismatch")
    if primary_gate_manifest is not None:
        frozen_control_fingerprint = canonical_json_sha256(
            primary_gate_manifest["frozen_lock"]["control_identity"]
        )
        if frozen_control_fingerprint != evaluation_control_fingerprint:
            raise ProtocolError(
                "locked gate control differs from the qualified runtime evaluation control"
            )
        if (
            qualification_fingerprint is None
            or primary_gate_manifest["frozen_lock"]["qualification_decision_fingerprint"]
            != qualification_fingerprint
        ):
            raise ProtocolError("locked gate does not bind the validated qualification decision")
    if primary_gate_decision is not None and confirmation_gate_decision is not None:
        validate_locked_gate_pair(protocol, primary_gate_manifest, confirmation_gate_manifest)
        if (
            primary_gate_decision["frozen_lock_sha256"]
            != confirmation_gate_decision["frozen_lock_sha256"]
        ):
            raise ProtocolError("locked decisions do not share one frozen candidate")
    confirmed = bool(
        prerequisites_complete
        and primary_gate_decision is not None
        and confirmation_gate_decision is not None
        and primary_gate_decision["decision"] == "pass"
        and confirmation_gate_decision["decision"] == "pass"
    )
    if primary_gate_decision is not None and qualification_decision["decision"] != "eligible-for-locked-gate":
        raise ProtocolError("locked gate evidence requires a passing qualification decision")
    if confirmed:
        if no_candidate_reason is not None:
            raise ProtocolError("confirmed candidate cannot have a no-candidate reason")
        if candidate_fingerprint == enoch0_fingerprint:
            raise ProtocolError("confirmed candidate must be distinct from permanent Enoch-0")
        if candidate_fingerprint == evaluation_control_fingerprint:
            raise ProtocolError(
                "confirmed candidate must be distinct from its runtime evaluation control"
            )
        decision = "freeze-enoch-1"
        reason = None
        downstream = candidate_fingerprint
        candidate_status = "confirmed"
        expected_terminal_phase = "W1.7"
    else:
        if no_candidate_reason not in NO_CANDIDATE_REASONS:
            raise ProtocolError("non-freeze decision requires an explicit canonical reason")
        if not prerequisites_complete and no_candidate_reason != "candidate-prerequisites-incomplete":
            raise ProtocolError("incomplete prerequisites require their explicit decision reason")
        if no_candidate_reason == "qualification-failed" and (
            qualification_decision is None
            or qualification_decision["decision"] != "reject-candidate"
        ):
            raise ProtocolError("qualification-failed requires a typed rejected qualification")
        if no_candidate_reason == "primary-locked-gate-failed" and (
            primary_gate_decision is None or primary_gate_decision["decision"] != "fail"
        ):
            raise ProtocolError("primary failure reason requires a failed primary decision")
        if (
            primary_gate_decision is not None
            and primary_gate_decision["decision"] == "fail"
            and no_candidate_reason != "primary-locked-gate-failed"
        ):
            raise ProtocolError("failed primary gate requires its explicit decision reason")
        if (
            primary_gate_decision is not None
            and primary_gate_decision["decision"] == "pass"
            and (
                confirmation_gate_decision is None
                or confirmation_gate_decision["decision"] != "pass"
            )
            and no_candidate_reason != "confirmation-not-passed"
        ):
            raise ProtocolError("unconfirmed primary pass must be recorded as non-confirmation")
        if (
            no_candidate_reason == "confirmation-not-passed"
            and (
                confirmation_gate_decision is None
                or confirmation_gate_decision["decision"] != "fail"
            )
        ):
            raise ProtocolError("non-confirmation requires a failed confirmation decision")
        decision = "no-confirmed-candidate"
        reason = no_candidate_reason
        downstream = enoch0_fingerprint
        candidate_status = (
            "provisional" if no_candidate_reason == "candidate-prerequisites-incomplete" else "not-confirmed"
        )
        expected_terminal_phase = {
            "no-survivor": "W1.4",
            "combination-regressed": "W1.4",
            "candidate-prerequisites-incomplete": "W1.4",
            "qualification-failed": "W1.5",
            "primary-locked-gate-failed": "W1.6",
            "confirmation-not-passed": "W1.7",
        }.get(no_candidate_reason)
        if no_candidate_reason == "candidate-withdrawn":
            expected_terminal_phase = phase_manifests[-1]["phase"]
            if expected_terminal_phase not in {"W1.4", "W1.5", "W1.6", "W1.7"}:
                raise ProtocolError("candidate withdrawal needs W1.4-or-later phase evidence")
    if phase_manifests[-1]["phase"] != expected_terminal_phase:
        raise ProtocolError(
            f"terminal decision reason requires phase evidence through "
            f"{expected_terminal_phase}, got {phase_manifests[-1]['phase']}"
        )
    evidence = set(evidence_fingerprints)
    if qualification_fingerprint is not None:
        evidence.add(qualification_fingerprint)
    if primary_fingerprint is not None:
        evidence.add(primary_fingerprint)
    if confirmation_fingerprint is not None:
        evidence.add(confirmation_fingerprint)
    phase_artifact_hashes = {
        artifact["sha256"]
        for manifest in phase_manifests
        for artifact in manifest["artifacts"]
    }
    if qualification_fingerprint is not None:
        qualification_phase = phase_manifests[5] if len(phase_manifests) > 5 else None
        if qualification_phase is None or qualification_fingerprint not in {
            artifact["sha256"] for artifact in qualification_phase["artifacts"]
        }:
            raise ProtocolError("W1.5 phase does not bind the qualification decision")
    if primary_fingerprint is not None:
        primary_phase = phase_manifests[6] if len(phase_manifests) > 6 else None
        if primary_phase is None or primary_fingerprint not in {
            artifact["sha256"] for artifact in primary_phase["artifacts"]
        }:
            raise ProtocolError("W1.6 phase does not bind the primary gate decision")
    if confirmation_fingerprint is not None:
        confirmation_phase = phase_manifests[7] if len(phase_manifests) > 7 else None
        if confirmation_phase is None or confirmation_fingerprint not in {
            artifact["sha256"] for artifact in confirmation_phase["artifacts"]
        }:
            raise ProtocolError("W1.7 phase does not bind the confirmation decision")
    for fingerprint in evidence:
        _require_sha256(fingerprint, "Week 1 evidence fingerprint")
    evidence.update(phase_fingerprints)
    evidence.update(phase_artifact_hashes)
    frozen_phase_manifests = copy.deepcopy(list(phase_manifests))
    body = {
        "candidate_fingerprint": candidate_fingerprint,
        "candidate_status": candidate_status,
        "confirmation_gate_decision": copy.deepcopy(confirmation_gate_decision),
        "confirmation_gate_manifest": copy.deepcopy(confirmation_gate_manifest),
        "confirmation_merged_result": copy.deepcopy(confirmation_merged_result),
        "control_manifest": copy.deepcopy(control_manifest),
        "control_manifest_fingerprint": control_manifest_fingerprint,
        "decision": decision,
        "downstream_primary_fingerprint": downstream,
        "evaluation_control_fingerprint": evaluation_control_fingerprint,
        "evaluation_control_relation": (
            "current-no-feature-evaluation-adapter-for-permanent-enoch-0"
            if evaluation_control_fingerprint is not None
            else None
        ),
        "evidence_fingerprints": sorted(evidence),
        "human_operator_review_required": True,
        "manifest_kind": "enoch-week1-terminal-decision",
        "manifest_version": MANIFEST_VERSION,
        "no_candidate_reason": reason,
        "permanent_scientific_control_fingerprint": enoch0_fingerprint,
        "phase_chain_sha256": canonical_json_sha256(frozen_phase_manifests),
        "phase_manifests": frozen_phase_manifests,
        "primary_gate_decision": copy.deepcopy(primary_gate_decision),
        "primary_gate_manifest": copy.deepcopy(primary_gate_manifest),
        "primary_merged_result": copy.deepcopy(primary_merged_result),
        "production_promotion_authorized": False,
        "production_promotion_disposition": "not-authorized-by-week-1",
        "protocol_fingerprint": protocol_fingerprint,
        "qualification_decision": copy.deepcopy(qualification_decision),
        "qualification_manifest": copy.deepcopy(qualification_manifest),
        "qualification_merged_results": copy.deepcopy(
            qualification_merged_results
        ),
        "stage2_rebaseline_authorized_after_human_review": True,
    }
    artifact = _with_fingerprint(body, "week1_decision_fingerprint")
    validate_week1_decision_artifact(protocol, artifact)
    return artifact


def validate_week1_decision_artifact(
    protocol: Mapping[str, Any], artifact: Mapping[str, Any]
) -> str:
    protocol_fingerprint = validate_protocol(protocol)
    if not isinstance(artifact, Mapping):
        raise ProtocolError("Week 1 decision artifact must be an object")
    _require_exact_keys(
        artifact,
        {
            "candidate_fingerprint",
            "candidate_status",
            "confirmation_gate_decision",
            "confirmation_gate_manifest",
            "confirmation_merged_result",
            "control_manifest",
            "control_manifest_fingerprint",
            "decision",
            "downstream_primary_fingerprint",
            "evaluation_control_fingerprint",
            "evaluation_control_relation",
            "evidence_fingerprints",
            "human_operator_review_required",
            "manifest_kind",
            "manifest_version",
            "no_candidate_reason",
            "permanent_scientific_control_fingerprint",
            "phase_chain_sha256",
            "phase_manifests",
            "primary_gate_decision",
            "primary_gate_manifest",
            "primary_merged_result",
            "production_promotion_authorized",
            "production_promotion_disposition",
            "protocol_fingerprint",
            "qualification_decision",
            "qualification_manifest",
            "qualification_merged_results",
            "stage2_rebaseline_authorized_after_human_review",
            "week1_decision_fingerprint",
        },
        "Week 1 decision artifact",
    )
    if artifact["manifest_kind"] != "enoch-week1-terminal-decision":
        raise ProtocolError("unexpected Week 1 decision artifact kind")
    if artifact["manifest_version"] != MANIFEST_VERSION:
        raise ProtocolError("unsupported Week 1 decision artifact version")
    if artifact["protocol_fingerprint"] != protocol_fingerprint:
        raise ProtocolError("Week 1 decision belongs to a different protocol")
    if artifact["production_promotion_authorized"] is not False:
        raise ProtocolError("Week 1 decision cannot authorize production promotion")
    if artifact["production_promotion_disposition"] != "not-authorized-by-week-1":
        raise ProtocolError("Week 1 production promotion disposition changed")
    if artifact["human_operator_review_required"] is not True:
        raise ProtocolError("Week 1 decision must require human operator review")
    if artifact["stage2_rebaseline_authorized_after_human_review"] is not True:
        raise ProtocolError("Week 1 decision must gate Stage 2 on human review")
    control_manifest = artifact["control_manifest"]
    control_manifest_fingerprint = validate_w1_0_control_manifest(
        protocol, control_manifest
    )
    if artifact["control_manifest_fingerprint"] != control_manifest_fingerprint:
        raise ProtocolError("terminal W1.0 control manifest fingerprint mismatch")
    phase_manifests = artifact["phase_manifests"]
    if not isinstance(phase_manifests, list):
        raise ProtocolError("Week 1 decision phase evidence must be a list")
    phase_fingerprints = validate_phase_chain(protocol, phase_manifests)
    if artifact["phase_chain_sha256"] != canonical_json_sha256(phase_manifests):
        raise ProtocolError("Week 1 phase-chain hash mismatch")
    if control_manifest_fingerprint not in {
        item["sha256"] for item in phase_manifests[0]["artifacts"]
    }:
        raise ProtocolError("Week 1 phase chain omits the frozen control manifest")
    control = _require_sha256(
        artifact["permanent_scientific_control_fingerprint"], "Enoch-0 fingerprint"
    )
    if control != canonical_json_sha256(
        control_manifest["policy_identities"]["enoch-0"]
    ):
        raise ProtocolError("terminal permanent control differs from W1.0 Enoch-0")
    evaluation_control = artifact["evaluation_control_fingerprint"]
    evaluation_control_relation = artifact["evaluation_control_relation"]
    if evaluation_control is None:
        if evaluation_control_relation is not None:
            raise ProtocolError(
                "terminal evaluation-control relation exists without an identity"
            )
    else:
        _require_sha256(
            evaluation_control, "runtime evaluation-control fingerprint"
        )
        if (
            evaluation_control_relation
            != "current-no-feature-evaluation-adapter-for-permanent-enoch-0"
        ):
            raise ProtocolError("terminal evaluation-control relation changed")
    candidate = artifact["candidate_fingerprint"]
    if candidate is not None:
        _require_sha256(candidate, "candidate fingerprint")
    decision = artifact["decision"]
    qualification = artifact["qualification_decision"]
    qualification_manifest = artifact["qualification_manifest"]
    qualification_merged_results = artifact["qualification_merged_results"]
    primary = artifact["primary_gate_decision"]
    primary_manifest = artifact["primary_gate_manifest"]
    primary_merged_result = artifact["primary_merged_result"]
    confirmation = artifact["confirmation_gate_decision"]
    confirmation_manifest = artifact["confirmation_gate_manifest"]
    confirmation_merged_result = artifact["confirmation_merged_result"]
    qualification_fingerprint = None
    primary_fingerprint = None
    confirmation_fingerprint = None
    if qualification is not None:
        if qualification_manifest is None or qualification_merged_results is None:
            raise ProtocolError(
                "terminal qualification decision lacks its manifest or merged results"
            )
        qualification_fingerprint = validate_w1_5_qualification_decision(
            protocol,
            qualification_manifest,
            qualification_merged_results,
            qualification,
        )
        if qualification_manifest["control_fingerprint"] != evaluation_control:
            raise ProtocolError(
                "terminal qualification uses another runtime evaluation control"
            )
    elif qualification_manifest is not None or qualification_merged_results is not None:
        raise ProtocolError("terminal qualification evidence lacks a decision")
    elif evaluation_control is not None:
        raise ProtocolError(
            "terminal runtime evaluation control lacks qualification evidence"
        )
    if primary is not None:
        if primary_manifest is None or primary_merged_result is None:
            raise ProtocolError(
                "terminal primary decision lacks its manifest or merged result"
            )
        primary_fingerprint = validate_locked_gate_decision(
            protocol, primary_manifest, primary_merged_result, primary
        )
        if primary["gate_role"] != "primary":
            raise ProtocolError("terminal primary gate decision has the wrong role")
        if (
            primary["candidate_fingerprint"]
            != canonical_json_sha256(primary_manifest["frozen_lock"]["candidate_identity"])
            or primary["frozen_lock_sha256"] != primary_manifest["frozen_lock_sha256"]
        ):
            raise ProtocolError("terminal primary decision identity differs from manifest")
    elif primary_manifest is not None or primary_merged_result is not None:
        raise ProtocolError("terminal primary evidence lacks a decision")
    if confirmation is not None:
        if confirmation_manifest is None or confirmation_merged_result is None:
            raise ProtocolError(
                "terminal confirmation decision lacks its manifest or merged result"
            )
        confirmation_fingerprint = validate_locked_gate_decision(
            protocol,
            confirmation_manifest,
            confirmation_merged_result,
            confirmation,
        )
        if confirmation["gate_role"] != "confirmation":
            raise ProtocolError("terminal confirmation decision has the wrong role")
        if (
            confirmation["candidate_fingerprint"]
            != canonical_json_sha256(
                confirmation_manifest["frozen_lock"]["candidate_identity"]
            )
            or confirmation["frozen_lock_sha256"]
            != confirmation_manifest["frozen_lock_sha256"]
        ):
            raise ProtocolError("terminal confirmation identity differs from manifest")
    elif confirmation_manifest is not None or confirmation_merged_result is not None:
        raise ProtocolError("terminal confirmation evidence lacks a decision")
    if len(phase_manifests) >= 6 and qualification is None:
        raise ProtocolError("W1.5-or-later terminal artifact lacks qualification evidence")
    if len(phase_manifests) >= 7 and primary is None:
        raise ProtocolError("W1.6-or-later terminal artifact lacks primary evidence")
    if len(phase_manifests) >= 8 and confirmation is None:
        raise ProtocolError("W1.7 terminal artifact lacks confirmation evidence")
    if primary is not None:
        if qualification is None or qualification["decision"] != "eligible-for-locked-gate":
            raise ProtocolError("terminal locked evidence lacks passing qualification")
        if primary["candidate_fingerprint"] != candidate:
            raise ProtocolError("terminal primary candidate identity mismatch")
        if (
            canonical_json_sha256(
                primary_manifest["frozen_lock"]["control_identity"]
            )
            != evaluation_control
        ):
            raise ProtocolError(
                "terminal primary control differs from the qualified runtime evaluation control"
            )
        if (
            primary_manifest["frozen_lock"]["qualification_decision_fingerprint"]
            != qualification_fingerprint
        ):
            raise ProtocolError("terminal primary gate binds another qualification")
    if confirmation is not None:
        if primary is None or confirmation["candidate_fingerprint"] != candidate:
            raise ProtocolError("terminal confirmation candidate identity mismatch")
        if confirmation["frozen_lock_sha256"] != primary["frozen_lock_sha256"]:
            raise ProtocolError("terminal locked decisions use different frozen locks")
        validate_locked_gate_pair(protocol, primary_manifest, confirmation_manifest)
    if decision == "freeze-enoch-1":
        if candidate is None or artifact["candidate_status"] != "confirmed":
            raise ProtocolError("freeze decision lacks a confirmed candidate")
        if candidate == control:
            raise ProtocolError("freeze candidate must differ from permanent Enoch-0")
        if candidate == evaluation_control:
            raise ProtocolError(
                "freeze candidate must differ from its runtime evaluation control"
            )
        if (
            qualification is None
            or qualification["decision"] != "eligible-for-locked-gate"
            or primary is None
            or primary["decision"] != "pass"
            or confirmation is None
            or confirmation["decision"] != "pass"
        ):
            raise ProtocolError("freeze decision lacks two typed passing locked gates")
        if artifact["downstream_primary_fingerprint"] != candidate:
            raise ProtocolError("freeze decision downstream identity is not the candidate")
        if artifact["no_candidate_reason"] is not None:
            raise ProtocolError("freeze decision cannot record a no-candidate reason")
    elif decision == "no-confirmed-candidate":
        if artifact["candidate_status"] not in {"not-confirmed", "provisional"}:
            raise ProtocolError("no-candidate decision has an invalid candidate status")
        if artifact["downstream_primary_fingerprint"] != control:
            raise ProtocolError("no-candidate decision must retain Enoch-0 downstream")
        if artifact["no_candidate_reason"] not in NO_CANDIDATE_REASONS:
            raise ProtocolError("no-candidate decision lacks a canonical reason")
        if (
            artifact["candidate_status"] == "provisional"
            and artifact["no_candidate_reason"] != "candidate-prerequisites-incomplete"
        ):
            raise ProtocolError("provisional status requires incomplete prerequisites")
        if (
            artifact["no_candidate_reason"] == "candidate-prerequisites-incomplete"
            and artifact["candidate_status"] != "provisional"
        ):
            raise ProtocolError("incomplete prerequisites require provisional status")
        if (
            artifact["no_candidate_reason"] != "candidate-prerequisites-incomplete"
            and artifact["candidate_status"] != "not-confirmed"
        ):
            raise ProtocolError("non-prerequisite failures must be not-confirmed")
        reason = artifact["no_candidate_reason"]
        if reason in {"no-survivor", "combination-regressed", "candidate-prerequisites-incomplete"}:
            if any(value is not None for value in (qualification, primary, confirmation)):
                raise ProtocolError("W1.4 terminal reason contains later-phase evidence")
        elif reason == "qualification-failed":
            if qualification is None or qualification["decision"] != "reject-candidate":
                raise ProtocolError("qualification failure lacks typed rejection")
            if primary is not None or confirmation is not None:
                raise ProtocolError("qualification failure contains locked-gate evidence")
        elif reason == "primary-locked-gate-failed":
            if primary is None or primary["decision"] != "fail" or confirmation is not None:
                raise ProtocolError("primary failure lacks one typed failed primary gate")
        elif reason == "confirmation-not-passed":
            if (
                primary is None
                or primary["decision"] != "pass"
                or confirmation is None
                or confirmation["decision"] != "fail"
            ):
                raise ProtocolError("non-confirmation lacks pass/fail typed gate evidence")
    else:
        raise ProtocolError("Week 1 terminal decision must be freeze or no-candidate")
    expected_terminal_phase = (
        "W1.7"
        if decision == "freeze-enoch-1"
        else {
            "no-survivor": "W1.4",
            "combination-regressed": "W1.4",
            "candidate-prerequisites-incomplete": "W1.4",
            "qualification-failed": "W1.5",
            "primary-locked-gate-failed": "W1.6",
            "confirmation-not-passed": "W1.7",
            "candidate-withdrawn": phase_manifests[-1]["phase"],
        }[artifact["no_candidate_reason"]]
    )
    if (
        artifact["no_candidate_reason"] == "candidate-withdrawn"
        and phase_manifests[-1]["phase"] not in {"W1.4", "W1.5", "W1.6", "W1.7"}
    ):
        raise ProtocolError("candidate withdrawal requires W1.4-or-later phase evidence")
    if phase_manifests[-1]["phase"] != expected_terminal_phase:
        raise ProtocolError("Week 1 terminal reason and phase evidence disagree")
    evidence = artifact["evidence_fingerprints"]
    if not isinstance(evidence, list) or evidence != sorted(set(evidence)):
        raise ProtocolError("Week 1 evidence fingerprints must be sorted and unique")
    for fingerprint in evidence:
        _require_sha256(fingerprint, "Week 1 evidence fingerprint")
    if not set(phase_fingerprints).issubset(evidence):
        raise ProtocolError("Week 1 evidence omits phase-chain fingerprints")
    phase_artifact_hashes = {
        item["sha256"]
        for manifest in phase_manifests
        for item in manifest["artifacts"]
    }
    if not phase_artifact_hashes.issubset(evidence):
        raise ProtocolError("Week 1 evidence omits phase artifact fingerprints")
    for fingerprint in (
        qualification_fingerprint,
        primary_fingerprint,
        confirmation_fingerprint,
        control_manifest_fingerprint,
    ):
        if fingerprint is not None and fingerprint not in evidence:
            raise ProtocolError("Week 1 evidence omits typed decision fingerprints")
    return _validate_fingerprint(
        artifact, "week1_decision_fingerprint", "Week 1 decision artifact"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init-seeds", help="freeze all Week 1 seed namespaces from one master seed"
    )
    initialize.add_argument("--master-seed", required=True, type=parse_u64)
    initialize.add_argument("--output", required=True, type=Path)
    initialize.add_argument("--ledger", type=Path, help="also create an empty consumption ledger")
    initialize.add_argument(
        "--allow-env",
        action="append",
        default=[],
        metavar="NAME",
        help="explicitly retain one SHENGJI_/GM_/OMNI_/GEN_ evaluator variable",
    )

    validate = subparsers.add_parser("validate", help="validate a frozen protocol and optional ledger")
    validate.add_argument("--protocol", required=True, type=Path)
    validate.add_argument("--ledger", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "init-seeds":
            protocol = build_protocol(
                arguments.master_seed, evaluator_env_allowlist=arguments.allow_env
            )
            atomic_write_json(arguments.output, protocol)
            if arguments.ledger is not None:
                atomic_write_json(arguments.ledger, new_seed_ledger(protocol))
            print(
                f"initialized {protocol['seed_registry']['global_seed_count']} seeds; "
                f"protocol_fingerprint={protocol['protocol_fingerprint']}"
            )
            return 0

        protocol = load_json_object(arguments.protocol)
        fingerprint = validate_protocol(protocol)
        if arguments.ledger is not None:
            ledger = load_json_object(arguments.ledger)
            validate_seed_ledger(protocol, ledger)
        print(f"valid protocol_fingerprint={fingerprint}")
        return 0
    except (OSError, ProtocolError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
