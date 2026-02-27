from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_json(obj: Any) -> bytes:
    # Canonical, stable serialization for hashing
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _get(obj: Dict[str, Any], *keys: str, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass(frozen=True)
class AuditRecordKey:
    galaxy: str
    framework_version: str
    timestamp: str


# -----------------------------
# Public API used by tests
# -----------------------------
REQUIRED_TOP_LEVEL_FIELDS = ("galaxia", "framework_version")
REQUIRED_MODEL_FIELDS = ("xi", "vif_hinge")
REQUIRED_CSV_COLUMNS = ("galaxia", "xi", "vif_hinge")


def validate_audit_json(path: str) -> Tuple[bool, List[str], AuditRecordKey | None, str]:
    """
    Returns:
      ok: bool
      errors: list[str]
      key: (galaxia, framework_version, timestamp) if available
      content_hash: sha256 of canonical json bytes (full doc)
    """
    errors: List[str] = []
    doc = _read_json(path)

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in doc or doc[field] in (None, ""):
            errors.append(f"Missing required field '{field}'")

    # Allow timestamp missing, but if present should be non-empty
    ts = doc.get("timestamp", "")
    if "timestamp" in doc and (ts is None or str(ts).strip() == ""):
        errors.append("Field 'timestamp' is present but empty")

    # We accept two possible nestings depending on your export style:
    #   - doc["resultados_modelo"][...]
    #   - doc["resultados"][...]
    model = doc.get("resultados_modelo")
    if model is None:
        model = doc.get("resultados")

    if not isinstance(model, dict):
        errors.append("Missing model results dict: expected 'resultados_modelo' or 'resultados'")
        model = {}

    # xi can be float or dict with "valor"
    xi_val = model.get("xi")
    if isinstance(xi_val, dict):
        xi_val = xi_val.get("valor")

    if xi_val is None:
        errors.append("Missing model field 'xi' (or 'xi.valor')")
    else:
        try:
            _ = float(xi_val)
        except Exception:
            errors.append(f"Field 'xi' is not numeric: {xi_val!r}")

    if "vif_hinge" not in model:
        errors.append("Missing model field 'vif_hinge'")
    else:
        try:
            _ = float(model["vif_hinge"])
        except Exception:
            errors.append(f"Field 'vif_hinge' is not numeric: {model['vif_hinge']!r}")

    # Optional but recommended: datos_curva schema if exists
    datos_curva = doc.get("datos_curva")
    if datos_curva is not None and isinstance(datos_curva, dict):
        # If present, must include vrot/e arrays of same length
        vrot = datos_curva.get("vrot_km_s")
        ev   = datos_curva.get("e_vrot_km_s")
        rads = datos_curva.get("cobertura_radial_kpc")
        if any(x is None for x in (vrot, ev, rads)):
            errors.append("datos_curva present but missing one of: cobertura_radial_kpc, vrot_km_s, e_vrot_km_s")
        else:
            if not (isinstance(vrot, list) and isinstance(ev, list) and isinstance(rads, list)):
                errors.append("datos_curva arrays must be lists")
            else:
                if not (len(vrot) == len(ev) == len(rads)):
                    errors.append(
                        f"datos_curva array length mismatch: len(r)={len(rads)}, len(v)={len(vrot)}, len(e)={len(ev)}"
                    )

    galaxy = str(doc.get("galaxia", "")).strip()
    ver = str(doc.get("framework_version", "")).strip()
    ts_s = str(ts).strip() if ts is not None else ""
    key = AuditRecordKey(galaxy, ver, ts_s) if (galaxy and ver) else None

    content_hash = _sha256_bytes(_canonical_json(doc))
    ok = len(errors) == 0
    return ok, errors, key, content_hash


def find_duplicates(audit_paths: List[str]) -> Dict[str, List[str]]:
    """
    Detect duplicates by:
      - identical content hash
      - identical (galaxy, framework_version, timestamp) if timestamp exists
    Returns dict category -> list of human-readable duplicate reports.
    """
    dup_reports: Dict[str, List[str]] = {"content_hash": [], "record_key": []}

    by_hash: Dict[str, List[str]] = {}
    by_key: Dict[AuditRecordKey, List[str]] = {}

    for p in audit_paths:
        ok, _errors, key, h = validate_audit_json(p)  # errors handled elsewhere
        by_hash.setdefault(h, []).append(p)
        if key and key.timestamp:
            by_key.setdefault(key, []).append(p)

    for h, paths in by_hash.items():
        if len(paths) > 1:
            dup_reports["content_hash"].append(f"{h[:12]}... -> {paths}")

    for k, paths in by_key.items():
        if len(paths) > 1:
            dup_reports["record_key"].append(f"{k} -> {paths}")

    return dup_reports


def validate_reports_csv(path: str, required_cols: List[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in required_cols if c not in cols]
        if missing:
            errors.append(f"Missing required columns in {os.path.basename(path)}: {missing}")
            return False, errors

        # basic duplicate row check for (galaxia, framework_version) if present
        seen = set()
        for i, row in enumerate(reader, start=2):
            g = row.get("galaxia") or row.get("galaxy") or row.get("galaxia_id")
            v = row.get("framework_version") or row.get("version")
            if g and v:
                key = (g.strip(), v.strip())
                if key in seen:
                    errors.append(f"Duplicate row key {key} at line {i}")
                else:
                    seen.add(key)

    return len(errors) == 0, errors


def discover_artifacts(repo_root: str = ".") -> Dict[str, List[str]]:
    audits = sorted(glob.glob(os.path.join(repo_root, "audits", "validated", "*.json")))
    reports_json = sorted(glob.glob(os.path.join(repo_root, "reports", "*.json")))
    reports_csv = sorted(glob.glob(os.path.join(repo_root, "reports", "*.csv")))
    return {"audits": audits, "reports_json": reports_json, "reports_csv": reports_csv}


# -----------------------------
# CLI entry point
# -----------------------------
def _cli(repo_root: str = ".") -> int:
    """Run all integrity checks and return exit code (0=pass, 1=fail)."""
    artifacts = discover_artifacts(repo_root)
    audits = artifacts["audits"]
    csvs = artifacts["reports_csv"]

    failed = False

    # --- audit JSON validation ---
    if not audits:
        print("ERROR: no JSON files found in audits/validated/")
        failed = True
    else:
        for p in audits:
            ok, errors, _key, _h = validate_audit_json(p)
            if ok:
                print(f"  OK  {p}")
            else:
                print(f"  FAIL {p}")
                for e in errors:
                    print(f"       • {e}")
                failed = True

        dups = find_duplicates(audits)
        for kind, reports in dups.items():
            for r in reports:
                print(f"  DUPLICATE [{kind}] {r}")
                failed = True

    # --- reports CSV validation ---
    if csvs:
        required = list(REQUIRED_CSV_COLUMNS)
        for p in csvs:
            ok, errors = validate_reports_csv(p, required_cols=required)
            if ok:
                print(f"  OK  {p}")
            else:
                print(f"  FAIL {p}")
                for e in errors:
                    print(f"       • {e}")
                failed = True
    else:
        print("  (no CSV files in reports/ — skipping CSV checks)")

    if failed:
        print("\nIntegrity check FAILED.")
        return 1
    print("\nIntegrity check PASSED.")
    return 0


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(_cli(root))
