"""Read-only integrity checks for the three immutable course result roots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SealedResult:
    label: str
    identity: str
    relative_root: str
    manifest_sha256: str

    @property
    def root(self) -> Path:
        return PROJECT_ROOT / self.relative_root


Q2R = SealedResult(
    label="Q2-R",
    identity="q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69",
    relative_root=(
        "results/q2_revision_formal/"
        "q2r-9c98b90049545d0508fb20bb488019608a4356d55fe91ff809bce8dc5c5e0c69"
    ),
    manifest_sha256="896fdbf45bbf19d04f74a802e78739714d286beabf322db6ef8f3594679411d1",
)

Q2F = SealedResult(
    label="Q2-F",
    identity="q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7",
    relative_root=(
        "results/q2f_course_extension/"
        "q2f-33955d1f3f1c1c4a435ba0945848224f32ff32d8f1a3b80f630858d9aa91e2c7"
    ),
    manifest_sha256="dfa0313b198824f5925f00494a307311445bd7ac08786f1e16570993d919f99b",
)

Q2F_FINAL = SealedResult(
    label="Q2-F final improvement",
    identity="q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1",
    relative_root=(
        "results/q2f_final_improvement/"
        "q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1"
    ),
    manifest_sha256="9e82add7292779256e0b36ca47bf7a5c0284a39f3656cfc128ff5920e9f23a10",
)

SEALED_RESULTS = (Q2R, Q2F, Q2F_FINAL)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sealed_result(specification: SealedResult) -> dict[str, Any]:
    """Fail closed unless the manifest and every listed artifact are intact."""

    manifest_path = specification.root / "hashes" / "result_manifest.json"
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != specification.manifest_sha256:
        raise RuntimeError(
            f"sealed_manifest_mismatch:{specification.label}:{actual_manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry.get("scope") == "external_report":
            relative_path = entry["project_relative_path"]
            path = PROJECT_ROOT / relative_path
        else:
            relative_path = entry["relative_path"]
            path = specification.root / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"sealed_artifact_missing:{specification.label}:{relative_path}"
            )
        if path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(
                f"sealed_artifact_size_mismatch:{specification.label}:{relative_path}"
            )
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(
                f"sealed_artifact_hash_mismatch:{specification.label}:{relative_path}"
            )
    return {
        "label": specification.label,
        "identity": specification.identity,
        "manifest_sha256": actual_manifest_hash,
        "artifact_count": len(manifest["files"]),
        "status": "PASS",
    }


def verify_all_sealed_results() -> list[dict[str, Any]]:
    """Verify Q2-R, Q2-F, and the final-improvement evidence."""

    return [verify_sealed_result(specification) for specification in SEALED_RESULTS]
