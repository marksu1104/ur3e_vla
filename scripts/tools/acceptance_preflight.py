"""Record CPU checks, the exact working tree and optional read-only H5 checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


PROJECT = Path(__file__).resolve().parents[2]
MINIMUM_UNIT_TESTS = 191


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=PROJECT, timeout=15)


def workspace_snapshot() -> dict:
    status = git("status", "--porcelain=v1", "--untracked-files=all", "-z").decode()
    records = iter(status.split("\0"))
    files = []
    for entry in records:
        if not entry:
            continue
        item = {"status": entry[:2], "path": entry[3:]}
        if "R" in entry[:2] or "C" in entry[:2]:
            item["original_path"] = next(records)
        path = PROJECT / item["path"]
        item["sha256"] = file_sha256(path) if path.is_file() else None
        files.append(item)
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "branch": git("branch", "--show-current").decode().strip(),
        "dirty": bool(files),
        "files": files,
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
    }


def run_check(
    name: str,
    argv: list[str],
    output: Path,
    *,
    minimum_tests: int | None = None,
) -> dict:
    log_path = output / f"{name}.log"
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                argv, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                timeout=120, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        record = {"passed": result.returncode == 0, "returncode": result.returncode,
                  "argv": argv, "log": str(log_path)}
        if name == "unit_tests":
            text = log_path.read_text(errors="replace")
            count = re.search(r"Ran (\d+) tests?", text)
            record["tests"] = int(count.group(1)) if count else 0
            skipped = re.search(r"skipped=(\d+)", text)
            record["skipped"] = int(skipped.group(1)) if skipped else 0
            errors = []
            if minimum_tests is not None and record["tests"] < minimum_tests:
                errors.append(
                    f"Only {record['tests']} unit tests ran; expected at least {minimum_tests}"
                )
            elif record["tests"] == 0:
                errors.append("No unit tests ran")
            if record["skipped"]:
                errors.append(f"{record['skipped']} unit tests were skipped")
            if errors:
                record["passed"] = False
                record["error"] = "; ".join(errors)
        return record
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"passed": False, "error": str(error), "argv": argv, "log": str(log_path)}


def h5_check(path: Path) -> dict:
    validation_result = None
    validation_error = None
    try:
        before = file_sha256(path)
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        from scripts.tools.validate_h5 import validate_h5

        validation_result = validate_h5(path)
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if "before" not in locals():
            return {"passed": False, "path": str(path), "error": str(error)}
        validation_error = error

    try:
        after = file_sha256(path)
    except OSError as error:
        messages = [f"H5 unavailable after validation: {error}"]
        if validation_error is not None:
            messages.append(f"validation failed: {validation_error}")
        return {
            "passed": False,
            "path": str(path),
            "sha256": before,
            "sha256_before": before,
            "sha256_after": None,
            "unchanged": False,
            "error": "; ".join(messages),
        }

    unchanged = before == after
    record = {
        **(validation_result or {}),
        "passed": validation_error is None and unchanged,
        "path": str(path),
        "sha256": before,
        "sha256_before": before,
        "sha256_after": after,
        "unchanged": unchanged,
    }
    errors = []
    if not unchanged:
        errors.append("H5 changed while it was being checked")
    if validation_error is not None:
        errors.append(f"validation failed: {validation_error}")
    if errors:
        record["error"] = "; ".join(errors)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, action="append", default=[],
                        help="H5 file to validate read-only; repeat for multiple files")
    parser.add_argument("--output-dir", type=Path, help="New directory for logs and report; must not exist")
    parser.add_argument("--skip-unit-tests", action="store_true", help="Explicitly omit the unit test suite")
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    output = args.output_dir or PROJECT / "outputs/test/preflight" / (
        f"preflight_{started.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    output = output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"FAIL: cannot create fresh output directory: {error}", file=sys.stderr)
        return 1

    report = {"started_at_utc": started.isoformat(), "python": sys.executable,
              "scope": "CPU checks and read-only H5 validation; manual acceptance is separate", "checks": {}}
    checks = report["checks"]
    try:
        report["workspace_before"] = workspace_snapshot()
        commands = [
            ("git_diff", ["git", "diff", "HEAD", "--check"]),
            ("compile", [sys.executable, "-m", "compileall", "-q", "vla_sim", "scripts", "tests", "rlds_builder"]),
        ]
        if not args.skip_unit_tests:
            commands.append(("unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"]))
        else:
            report["unit_tests_skipped"] = True
        for name, argv in commands:
            checks[name] = run_check(
                name,
                argv,
                output,
                minimum_tests=MINIMUM_UNIT_TESTS if name == "unit_tests" else None,
            )
            print(f"{'PASS' if checks[name]['passed'] else 'FAIL'}: {name}", flush=True)
        paths = list(dict.fromkeys(path.resolve() for path in args.h5))
        if not paths:
            report["h5_checks_skipped"] = True
        for index, path in enumerate(paths):
            name = f"h5_{index}"
            checks[name] = h5_check(path)
            print(f"{'PASS' if checks[name]['passed'] else 'FAIL'}: {path}", flush=True)
        report["workspace_after"] = workspace_snapshot()
        unchanged = report["workspace_before"] == report["workspace_after"]
        checks["stable_workspace"] = {"passed": unchanged}
        if not unchanged:
            checks["stable_workspace"]["error"] = "Working tree changed during checks; rerun on a stable version"
            print("FAIL: working tree changed during checks", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        checks["preflight"] = {"passed": False, "error": str(error)}
        print(f"FAIL: {error}", file=sys.stderr)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = bool(checks) and all(check["passed"] for check in checks.values())
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"{'PASS' if report['passed'] else 'FAIL'}: preflight; report={report_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
