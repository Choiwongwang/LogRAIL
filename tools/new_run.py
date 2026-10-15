#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment run helper for LogRAIL on the school server.

Layout (outside the git repo by default):
  ~/lograil/runs/<run_id>/
      meta.yaml      # params + git sha + env snapshot
      stdout.log     # tee'd command output (when using `exec`)
      metrics.json   # filled by `close` or manually
      artifacts/     # preds, checkpoints, reports, ...
  ~/lograil/index/runs.csv
  ~/lograil/journal/lab_notes.md

Examples:
  # create an empty run card
  python tools/new_run.py create --tag s1_baseline \\
      --param seed=42 --param window_size=20 --note "Stage-1 baseline"

  # create + run a command (logs to stdout.log, updates meta)
  python tools/new_run.py exec --tag s1_smoke -- \\
      python -c "print('hello')"

  # attach metrics and append index row
  python tools/new_run.py close 20261016_001_s1_baseline \\
      --metrics-json path/to/metrics.json --status ok

  python tools/new_run.py list
  python tools/new_run.py show latest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("[x] PyYAML required: pip install PyYAML") from e


DEFAULT_WORKSPACE = Path(os.path.expanduser("~/lograil"))
REPO_ROOT = Path(__file__).resolve().parents[1]


def _workspace(p: Optional[Path] = None) -> Path:
    return Path(p or os.environ.get("LOGRAIL_WORKSPACE", DEFAULT_WORKSPACE)).expanduser()


def _paths(ws: Path) -> Dict[str, Path]:
    return {
        "ws": ws,
        "runs": ws / "runs",
        "index_dir": ws / "index",
        "index_csv": ws / "index" / "runs.csv",
        "journal": ws / "journal" / "lab_notes.md",
    }


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _git_info(repo: Path) -> Dict[str, str]:
    def _run(args: List[str]) -> str:
        try:
            out = subprocess.check_output(
                args, cwd=str(repo), stderr=subprocess.DEVNULL, text=True
            )
            return out.strip()
        except Exception:
            return ""

    dirty = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_commit_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": _run(["git", "branch", "--show-current"]),
        "git_dirty": "true" if dirty else "false",
        "git_remote_origin": _run(["git", "remote", "get-url", "origin"]),
    }


def _yaml_safe(x: Any) -> Any:
    """Coerce values to types PyYAML SafeDumper accepts."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _yaml_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_yaml_safe(v) for v in x]
    return str(x)


def _env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "LOGRAIL_MODELS": os.environ.get("LOGRAIL_MODELS", ""),
    }
    try:
        import torch

        snap["torch"] = str(torch.__version__)
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["gpu_name"] = str(torch.cuda.get_device_name(0))
            snap["gpu_count"] = int(torch.cuda.device_count())
    except Exception:
        snap["torch"] = ""
    for pkg in ("transformers", "sentence_transformers", "chromadb"):
        try:
            mod = __import__(pkg)
            snap[pkg] = str(getattr(mod, "__version__", "?"))
        except Exception:
            snap[pkg] = ""
    return _yaml_safe(snap)


def _next_seq(runs_dir: Path, day: str) -> int:
    runs_dir.mkdir(parents=True, exist_ok=True)
    mx = 0
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith(day + "_") and len(name) > len(day) + 1:
            part = name[len(day) + 1 :].split("_", 1)[0]
            if part.isdigit():
                mx = max(mx, int(part))
    return mx + 1


def _make_run_id(runs_dir: Path, tag: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in tag.strip())
    safe = safe.strip("-_") or "run"
    day = _today()
    seq = _next_seq(runs_dir, day)
    return f"{day}_{seq:03d}_{safe}"


def _resolve_run(runs_dir: Path, run_id: str) -> Path:
    if run_id in ("latest", "@latest"):
        dirs = sorted(
            [p for p in runs_dir.iterdir() if p.is_dir() and (p / "meta.yaml").exists()],
            key=lambda p: p.name,
        )
        if not dirs:
            raise SystemExit("[x] no runs yet")
        return dirs[-1]
    path = runs_dir / run_id
    if not path.is_dir():
        raise SystemExit(f"[x] run not found: {path}")
    return path


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    with open(run_dir / "meta.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    with open(run_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_yaml_safe(meta), f, sort_keys=False, allow_unicode=True)


def _parse_params(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise SystemExit(f"[x] --param must be key=value, got: {raw}")
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        # light typing
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                if "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


INDEX_FIELDS = [
    "run_id",
    "created_at",
    "finished_at",
    "status",
    "tag",
    "git_commit_short",
    "git_branch",
    "git_dirty",
    "command",
    "duration_sec",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "note",
    "run_dir",
]


def _ensure_index(index_csv: Path) -> None:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    if not index_csv.exists():
        with open(index_csv, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=INDEX_FIELDS).writeheader()


def _upsert_index_row(index_csv: Path, row: Dict[str, Any]) -> None:
    _ensure_index(index_csv)
    rows: List[Dict[str, str]] = []
    with open(index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (r.get(k) or "") for k in INDEX_FIELDS})

    rid = str(row.get("run_id", ""))
    new_row = {k: "" for k in INDEX_FIELDS}
    for k in INDEX_FIELDS:
        if k in row and row[k] is not None:
            new_row[k] = str(row[k])

    replaced = False
    for i, r in enumerate(rows):
        if r.get("run_id") == rid:
            # keep old cells if new blank
            merged = dict(r)
            for k, v in new_row.items():
                if v != "":
                    merged[k] = v
            rows[i] = merged
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    with open(index_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)


def cmd_init_workspace(args: argparse.Namespace) -> None:
    p = _paths(_workspace(args.workspace))
    p["runs"].mkdir(parents=True, exist_ok=True)
    p["index_dir"].mkdir(parents=True, exist_ok=True)
    _ensure_index(p["index_csv"])
    p["journal"].parent.mkdir(parents=True, exist_ok=True)
    if not p["journal"].exists():
        p["journal"].write_text(
            "# LogRAIL lab notes\n\n"
            "每条实验写 3–5 行：目的 / 结果 / 是否采纳 / 下一步。\n\n"
            "## Template\n\n"
            "### YYYY-MM-DD — <run_id>\n"
            "- Goal:\n"
            "- Result:\n"
            "- Keep?: yes/no\n"
            "- Next:\n\n",
            encoding="utf-8",
        )
    # starter metrics schema
    schema = p["index_dir"] / "metrics.schema.json"
    if not schema.exists():
        schema.write_text(
            json.dumps(
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                    "n": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "extra": {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"[✓] workspace ready: {p['ws']}")
    print(f"  runs:    {p['runs']}")
    print(f"  index:   {p['index_csv']}")
    print(f"  journal: {p['journal']}")


def _create_run(
    ws: Path,
    tag: str,
    note: str,
    params: Dict[str, Any],
    command: Optional[List[str]] = None,
) -> Path:
    p = _paths(ws)
    p["runs"].mkdir(parents=True, exist_ok=True)
    _ensure_index(p["index_csv"])

    run_id = _make_run_id(p["runs"], tag)
    run_dir = p["runs"] / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    git = _git_info(REPO_ROOT)
    created = datetime.now().isoformat(timespec="seconds")
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "tag": tag,
        "note": note,
        "status": "created",
        "created_at": created,
        "finished_at": None,
        "duration_sec": None,
        "command": " ".join(shlex.quote(c) for c in command) if command else "",
        "command_argv": command or [],
        "params": params,
        "paths": {
            "repo": str(REPO_ROOT),
            "workspace": str(ws),
            "run_dir": str(run_dir),
            "artifacts": str(run_dir / "artifacts"),
            "models": os.environ.get("LOGRAIL_MODELS", str(ws / "models")),
        },
        "git": git,
        "env": _env_snapshot(),
        "metrics": {},
    }
    _save_meta(run_dir, meta)
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    _upsert_index_row(
        p["index_csv"],
        {
            "run_id": run_id,
            "created_at": created,
            "status": "created",
            "tag": tag,
            "git_commit_short": git.get("git_commit_short", ""),
            "git_branch": git.get("git_branch", ""),
            "git_dirty": git.get("git_dirty", ""),
            "command": meta["command"],
            "note": note,
            "run_dir": str(run_dir),
        },
    )
    print(f"[✓] created run: {run_id}")
    print(f"    dir: {run_dir}")
    return run_dir


def cmd_create(args: argparse.Namespace) -> None:
    ws = _workspace(args.workspace)
    cmd_init_workspace(argparse.Namespace(workspace=ws))
    _create_run(ws, args.tag, args.note or "", _parse_params(args.param))


def cmd_exec(args: argparse.Namespace) -> None:
    ws = _workspace(args.workspace)
    cmd_init_workspace(argparse.Namespace(workspace=ws))
    if not args.cmd:
        raise SystemExit("[x] provide a command after --")

    run_dir = _create_run(
        ws,
        args.tag,
        args.note or "",
        _parse_params(args.param),
        command=args.cmd,
    )
    meta = _load_meta(run_dir)
    meta["status"] = "running"
    meta["started_at"] = datetime.now().isoformat(timespec="seconds")
    _save_meta(run_dir, meta)

    log_path = run_dir / "stdout.log"
    argv = args.cmd
    print(f"[i] exec: {' '.join(shlex.quote(c) for c in argv)}")
    print(f"[i] log:  {log_path}")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# run_id={run_dir.name}\n")
        logf.write(f"# started={meta['started_at']}\n")
        logf.write(f"# cmd={' '.join(shlex.quote(c) for c in argv)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            argv,
            cwd=args.cwd or str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        rc = proc.wait()

    dur = round(time.time() - t0, 3)
    finished = datetime.now().isoformat(timespec="seconds")
    status = "ok" if rc == 0 else "failed"
    meta = _load_meta(run_dir)
    meta["status"] = status
    meta["exit_code"] = rc
    meta["finished_at"] = finished
    meta["duration_sec"] = dur
    _save_meta(run_dir, meta)

    p = _paths(ws)
    _upsert_index_row(
        p["index_csv"],
        {
            "run_id": run_dir.name,
            "finished_at": finished,
            "status": status,
            "duration_sec": dur,
            "command": meta.get("command", ""),
        },
    )
    print(f"[{'✓' if rc == 0 else 'x'}] status={status} exit={rc} duration={dur}s")
    if args.auto_close_metrics and (run_dir / "artifacts" / "metrics.json").exists():
        _close_with_metrics(
            ws,
            run_dir,
            run_dir / "artifacts" / "metrics.json",
            status=status,
            note=None,
        )
    sys.exit(rc)


def _metrics_from_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "final" in data and isinstance(data["final"], dict):
        # eval_final export_report style
        return data["final"]
    if not isinstance(data, dict):
        raise SystemExit(f"[x] metrics json must be an object: {path}")
    return data


def _close_with_metrics(
    ws: Path,
    run_dir: Path,
    metrics_path: Optional[Path],
    status: str,
    note: Optional[str],
) -> None:
    meta = _load_meta(run_dir)
    metrics: Dict[str, Any] = dict(meta.get("metrics") or {})
    if metrics_path:
        metrics.update(_metrics_from_file(metrics_path))
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if note:
        meta["note"] = note
    meta["metrics"] = metrics
    meta["status"] = status
    if not meta.get("finished_at"):
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_meta(run_dir, meta)

    p = _paths(ws)
    _upsert_index_row(
        p["index_csv"],
        {
            "run_id": run_dir.name,
            "finished_at": meta.get("finished_at", ""),
            "status": status,
            "tag": meta.get("tag", ""),
            "duration_sec": meta.get("duration_sec", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "f1": metrics.get("f1", ""),
            "accuracy": metrics.get("accuracy", ""),
            "note": meta.get("note", ""),
            "run_dir": str(run_dir),
        },
    )
    print(f"[✓] closed {run_dir.name}  status={status}")
    if metrics:
        print(
            f"    P={metrics.get('precision', '-')}  "
            f"R={metrics.get('recall', '-')}  "
            f"F1={metrics.get('f1', '-')}"
        )


def cmd_close(args: argparse.Namespace) -> None:
    ws = _workspace(args.workspace)
    p = _paths(ws)
    run_dir = _resolve_run(p["runs"], args.run_id)
    metrics_path = Path(args.metrics_json).expanduser() if args.metrics_json else None
    if metrics_path and not metrics_path.exists():
        raise SystemExit(f"[x] metrics file not found: {metrics_path}")
    _close_with_metrics(ws, run_dir, metrics_path, args.status, args.note)


def cmd_list(args: argparse.Namespace) -> None:
    ws = _workspace(args.workspace)
    p = _paths(ws)
    _ensure_index(p["index_csv"])
    with open(p["index_csv"], "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("(no runs yet)")
        return
    rows = rows[-args.tail :]
    hdr = f"{'run_id':<36} {'status':<8} {'f1':>8} {'tag'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.get('run_id',''):<36} {r.get('status',''):<8} "
            f"{(r.get('f1') or '-'):>8} {r.get('tag','')}"
        )


def cmd_show(args: argparse.Namespace) -> None:
    ws = _workspace(args.workspace)
    p = _paths(ws)
    run_dir = _resolve_run(p["runs"], args.run_id)
    meta = _load_meta(run_dir)
    print(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print(f"run_dir: {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Create/exec/close experiment runs under ~/lograil/runs"
    )
    ap.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="default ~/lograil (or $LOGRAIL_WORKSPACE)",
    )
    sp = ap.add_subparsers(dest="cmd_name", required=True)

    p_init = sp.add_parser("init", help="create runs/ index/ journal/ skeletons")
    p_init.set_defaults(func=cmd_init_workspace)

    p_c = sp.add_parser("create", help="create a new run directory + meta.yaml")
    p_c.add_argument("--tag", required=True, help="short label, e.g. s1_baseline")
    p_c.add_argument("--note", default="", help="one-line purpose")
    p_c.add_argument(
        "--param",
        action="append",
        default=[],
        help="key=value (repeatable)",
    )
    p_c.set_defaults(func=cmd_create)

    p_e = sp.add_parser(
        "exec",
        help="create a run and execute a command with stdout tee'd to stdout.log",
    )
    p_e.add_argument("--tag", required=True)
    p_e.add_argument("--note", default="")
    p_e.add_argument("--param", action="append", default=[])
    p_e.add_argument(
        "--cwd",
        default=None,
        help="working directory for the command (default: LogRAIL repo root)",
    )
    p_e.add_argument(
        "--auto-close-metrics",
        action="store_true",
        help="if artifacts/metrics.json exists after cmd, call close automatically",
    )
    p_e.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="command after -- , e.g. -- python train.py",
    )
    p_e.set_defaults(func=cmd_exec)

    p_cl = sp.add_parser("close", help="write metrics + update index/runs.csv")
    p_cl.add_argument("run_id", help="run id or 'latest'")
    p_cl.add_argument("--metrics-json", default=None)
    p_cl.add_argument("--status", default="ok", choices=["ok", "failed", "aborted"])
    p_cl.add_argument("--note", default=None)
    p_cl.set_defaults(func=cmd_close)

    p_l = sp.add_parser("list", help="list recent runs from index/runs.csv")
    p_l.add_argument("--tail", type=int, default=20)
    p_l.set_defaults(func=cmd_list)

    p_s = sp.add_parser("show", help="print meta.yaml for a run")
    p_s.add_argument("run_id", help="run id or 'latest'")
    p_s.set_defaults(func=cmd_show)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    # argparse puts leading '--' into remainder sometimes
    if getattr(args, "cmd", None) is not None:
        if args.cmd and args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
    args.func(args)


if __name__ == "__main__":
    main()
