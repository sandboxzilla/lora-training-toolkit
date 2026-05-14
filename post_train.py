#!/usr/bin/env python3
"""
post_train.py — Config-driven post-training pipeline for LoRA adapters.

Reads the same YAML job config as lora_train.py.  Runs three independently
skippable steps:

  1. GGUF conversion  — convert HuggingFace adapter → GGUF via
                        convert_lora_to_gguf.py from the llama.cpp source tree
  2. Hot-swap deploy  — POST the new GGUF to a running llama.cpp server
                        via the /lora endpoint
  3. Inference tests  — run test cases defined in post_training.tests,
                        POST to /v1/chat/completions, match response against
                        each test's expect_pattern regex

Results are written to:
  {run_dir}/{job_name}_post_train_report.json

Exit codes:
    0  — all enabled steps passed
    1  — config/GGUF/server error (pipeline could not run)
    2  — pipeline ran but one or more inference tests failed
"""
import os, sys, json, re, argparse, subprocess, textwrap, time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).parent.resolve()

# ── CLI ───────────────────────────────────────────────────────────────────────
_EPILOG = """\
EXAMPLES
--------
  Full pipeline (GGUF → deploy → tests):
    python3 post_train.py --config jobs/my_job.yaml

  Use a specific completed run:
    python3 post_train.py --config jobs/my_job.yaml --run-id r003

  Skip GGUF conversion (adapter already converted):
    python3 post_train.py --config jobs/my_job.yaml --skip-gguf

  Tests only (no conversion, no deploy):
    python3 post_train.py --config jobs/my_job.yaml --skip-gguf --skip-deploy

  Dry-run GGUF conversion (check paths without writing):
    python3 post_train.py --config jobs/my_job.yaml --dry-run

CONFIG — post_training SECTION (required for full pipeline)
-----------------------------------------------------------
  post_training.gguf.llama_cpp_dir      Path to llama.cpp source tree
  post_training.gguf.base_model         HuggingFace base model dir (for conversion)
  post_training.gguf.outtype            GGUF quantization: f16 | q8_0 | f32 (default f16)
  post_training.gguf.output_dir         Where GGUF file is written; default: {run_dir}/
  post_training.gguf.venv               Virtualenv to activate before conversion (optional)
  post_training.server.base_url         llama.cpp server base URL (e.g. http://localhost:8080)
  post_training.server.lora_endpoint    Hot-swap endpoint (default: /lora)
  post_training.server.chat_endpoint    Inference endpoint (default: /v1/chat/completions)
  post_training.server.adapter_id       Adapter slot id for hot-swap (default: 0)
  post_training.server.scale            Adapter scale: 1.0 = active, 0.0 = dormant (default: 1.0)
  post_training.server.timeout_s        HTTP timeout in seconds (default: 60)
  post_training.tests[]                 List of inference test cases (see below)

TEST CASE FORMAT (each item under post_training.tests):
-------------------------------------------------------
  name:            Test name (used in report)
  system:          System prompt string
  user:            User message string
  expect_pattern:  Python regex; test passes if the response body matches
  max_tokens:      Max tokens to generate (default: 256)
  temperature:     Sampling temperature; 0.0 = deterministic (default: 0.0)

REPORT FORMAT
-------------
  Written to {run_dir}/{job_name}_post_train_report.json
  Fields: job, run_id, run_dir, timestamp, steps{gguf,deploy,tests}, verdict

EXIT CODES
----------
   0  All enabled steps passed
   1  Config error, missing required fields, GGUF/server setup failure
   2  One or more inference tests failed (pipeline ran, tests reported)

See USER_MANUAL.md for full documentation."""

parser = argparse.ArgumentParser(
    description=(
        "Config-driven post-training pipeline: GGUF conversion → adapter "
        "hot-swap → inference tests.  Reads the same YAML config as lora_train.py."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=_EPILOG,
    add_help=False,
)
parser.add_argument(
    "-h", "--help", "-?",
    action="help",
    default=argparse.SUPPRESS,
    help="Show this help message and exit",
)
parser.add_argument(
    "--config", metavar="YAML", required=True,
    help="Path to the job YAML config file (same file used for lora_train.py)",
)
parser.add_argument(
    "--run-id", metavar="ID",
    help=(
        "Run directory to use (e.g. r003).  Default: auto-detect the latest "
        "completed run that contains an adapter/ directory."
    ),
)
parser.add_argument(
    "--skip-gguf", action="store_true",
    help=(
        "Skip GGUF conversion.  Use when the adapter has already been converted "
        "or when you want to test an existing GGUF file."
    ),
)
parser.add_argument(
    "--skip-deploy", action="store_true",
    help=(
        "Skip hot-swap deployment.  Use when the server already has the correct "
        "adapter loaded, or to run tests without changing the active adapter."
    ),
)
parser.add_argument(
    "--dry-run", action="store_true",
    help=(
        "Print resolved paths and actions without executing them.  "
        "Useful for verifying config before a real run."
    ),
)
args = parser.parse_args()

# ── Config loading ────────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed.  Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

cfg_path = Path(args.config).expanduser().resolve()
if not cfg_path.exists():
    print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
    sys.exit(1)

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

def _get(d, *keys, default=None, required=False, expand=False):
    """Safely walk nested dict; optionally expand ~ in path values."""
    v = d
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            if required:
                print(f"ERROR: required config field missing: {'.'.join(str(k) for k in keys)}", file=sys.stderr)
                sys.exit(1)
            return default
        v = v[k]
    if expand and isinstance(v, str):
        v = str(Path(v).expanduser())
    return v

# ── Resolve job identity and run directory ────────────────────────────────────
job_name = _get(cfg, "job", "name", required=True)
base_dir  = Path(_get(cfg, "output", "base_dir", required=True, expand=True))
runs_dir  = base_dir / job_name / "runs"

def _find_latest_run(runs_dir: Path) -> Path | None:
    """Return the run dir with the most recent adapter/ subdirectory, or None."""
    candidates = sorted(runs_dir.glob("r[0-9][0-9][0-9]"), reverse=True)
    for c in candidates:
        if (c / "adapter").exists():
            return c
    return None

if args.run_id:
    run_dir = runs_dir / args.run_id
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
else:
    run_dir = _find_latest_run(runs_dir)
    if run_dir is None:
        # Also check legacy flat adapter path (pre-config-driven layout)
        legacy = Path(_get(cfg, "output", "adapter_dir", default="", expand=True))
        if legacy and Path(legacy).exists():
            run_dir = Path(legacy).parent
            print(f"INFO: no versioned runs found, using legacy adapter path: {legacy}")
        else:
            print(f"ERROR: no completed run (with adapter/) found under {runs_dir}", file=sys.stderr)
            sys.exit(1)

adapter_hf_dir = run_dir / "adapter"
# Also accept legacy direct adapter_dir from config
if not adapter_hf_dir.exists():
    legacy_dir = _get(cfg, "output", "adapter_dir", default="", expand=True)
    if legacy_dir and Path(legacy_dir).exists():
        adapter_hf_dir = Path(legacy_dir)
    else:
        print(f"ERROR: adapter directory not found: {adapter_hf_dir}", file=sys.stderr)
        sys.exit(1)

print(f"[post_train] job={job_name}  run={run_dir.name}  adapter={adapter_hf_dir}")

# ── Report scaffolding ────────────────────────────────────────────────────────
report_path = run_dir / f"{job_name}_post_train_report.json"
report = {
    "job":       job_name,
    "run_id":    run_dir.name,
    "run_dir":   str(run_dir),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "steps": {
        "gguf":   {"status": "skipped", "detail": ""},
        "deploy": {"status": "skipped", "detail": ""},
        "tests":  {"status": "skipped", "results": []},
    },
    "verdict": "PENDING",
}

def _save_report():
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

# ── STEP 1: GGUF conversion ───────────────────────────────────────────────────
gguf_path: Path | None = None

if not args.skip_gguf:
    pt = _get(cfg, "post_training", "gguf")
    if pt is None:
        print("ERROR: post_training.gguf section required for GGUF conversion; use --skip-gguf to skip", file=sys.stderr)
        sys.exit(1)

    llama_cpp_dir = Path(_get(cfg, "post_training", "gguf", "llama_cpp_dir", required=True, expand=True))
    base_model    = Path(_get(cfg, "post_training", "gguf", "base_model", required=True, expand=True))
    outtype       = _get(cfg, "post_training", "gguf", "outtype", default="f16")
    output_dir_s  = _get(cfg, "post_training", "gguf", "output_dir", default="", expand=True)
    output_dir    = Path(output_dir_s) if output_dir_s else run_dir
    venv          = _get(cfg, "post_training", "gguf", "venv", default="", expand=True)
    python_exe    = _get(cfg, "post_training", "gguf", "python", default="python3")

    converter = llama_cpp_dir / "convert_lora_to_gguf.py"
    if not converter.exists():
        print(f"ERROR: convert_lora_to_gguf.py not found: {converter}", file=sys.stderr)
        sys.exit(1)

    gguf_name = f"{job_name}-{outtype}.gguf"
    gguf_path = output_dir / gguf_name

    if venv:
        venv_python = Path(venv) / "bin" / "python3"
        python_bin = str(venv_python) if venv_python.exists() else python_exe
    else:
        python_bin = python_exe

    cmd = [
        python_bin, str(converter),
        str(adapter_hf_dir),
        "--base",    str(base_model),
        "--outfile", str(gguf_path),
        "--outtype", outtype,
        "--verbose",
    ]

    print(f"[gguf] converting {adapter_hf_dir.name} → {gguf_path.name}")
    if args.dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        report["steps"]["gguf"] = {"status": "dry-run", "cmd": " ".join(cmd), "output": gguf_path.name}
    else:
        _save_report()
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr[-1000:] or result.stdout[-1000:]
            print(f"ERROR: GGUF conversion failed:\n{detail}", file=sys.stderr)
            report["steps"]["gguf"] = {"status": "failed", "detail": detail}
            report["verdict"] = "FAIL"
            _save_report()
            sys.exit(1)
        print(f"[gguf] OK → {gguf_path}")
        report["steps"]["gguf"] = {"status": "ok", "gguf_path": str(gguf_path)}
        _save_report()
else:
    # Try to find an existing GGUF for this job
    output_dir_s = _get(cfg, "post_training", "gguf", "output_dir", default="", expand=True)
    output_dir   = Path(output_dir_s) if output_dir_s else run_dir
    outtype      = _get(cfg, "post_training", "gguf", "outtype", default="f16")
    gguf_name    = f"{job_name}-{outtype}.gguf"
    candidate    = output_dir / gguf_name
    if candidate.exists():
        gguf_path = candidate
        print(f"[gguf] skipped; using existing {gguf_path}")
    else:
        print(f"[gguf] skipped; no existing GGUF at {candidate} (deploy will be skipped too)")
    report["steps"]["gguf"]["detail"] = "skipped by --skip-gguf"

# ── STEP 2: Hot-swap deploy ───────────────────────────────────────────────────
if not args.skip_deploy:
    if gguf_path is None or not gguf_path.exists():
        print("WARN: no GGUF path available; skipping deploy")
        report["steps"]["deploy"] = {"status": "skipped", "detail": "no GGUF path"}
    else:
        pt_srv = _get(cfg, "post_training", "server")
        if pt_srv is None:
            print("ERROR: post_training.server section required for deploy; use --skip-deploy to skip", file=sys.stderr)
            sys.exit(1)

        base_url      = _get(cfg, "post_training", "server", "base_url", required=True).rstrip("/")
        lora_endpoint = _get(cfg, "post_training", "server", "lora_endpoint", default="/lora")
        adapter_id    = _get(cfg, "post_training", "server", "adapter_id", default=0)
        scale         = _get(cfg, "post_training", "server", "scale", default=1.0)
        timeout_s     = _get(cfg, "post_training", "server", "timeout_s", default=60)

        lora_url = f"{base_url}{lora_endpoint}"
        payload = json.dumps([{
            "id":    adapter_id,
            "path":  str(gguf_path),
            "scale": float(scale),
        }]).encode()

        print(f"[deploy] POST {lora_url}  adapter_id={adapter_id}  scale={scale}")
        if args.dry_run:
            print(f"  [dry-run] payload: {payload.decode()}")
            report["steps"]["deploy"] = {"status": "dry-run", "url": lora_url}
        else:
            try:
                req = urlrequest.Request(
                    lora_url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                    body = resp.read().decode(errors="replace")
                print(f"[deploy] OK  response: {body[:200]}")
                report["steps"]["deploy"] = {"status": "ok", "response": body[:500]}
                _save_report()
            except HTTPError as e:
                detail = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
                print(f"ERROR: deploy failed: {detail}", file=sys.stderr)
                report["steps"]["deploy"] = {"status": "failed", "detail": detail}
                report["verdict"] = "FAIL"
                _save_report()
                sys.exit(1)
            except URLError as e:
                detail = f"Connection error: {e.reason}"
                print(f"ERROR: deploy failed: {detail}", file=sys.stderr)
                report["steps"]["deploy"] = {"status": "failed", "detail": detail}
                report["verdict"] = "FAIL"
                _save_report()
                sys.exit(1)
else:
    report["steps"]["deploy"]["detail"] = "skipped by --skip-deploy"

# ── STEP 3: Inference tests ───────────────────────────────────────────────────
test_cases = _get(cfg, "post_training", "tests", default=[])
if not isinstance(test_cases, list) or len(test_cases) == 0:
    print("[tests] no test cases defined in post_training.tests — skipping")
    report["steps"]["tests"] = {"status": "skipped", "detail": "no test cases", "results": []}
else:
    pt_srv    = _get(cfg, "post_training", "server")
    if pt_srv is None:
        print("WARN: post_training.server not configured — cannot run tests")
        report["steps"]["tests"] = {"status": "skipped", "detail": "no server config", "results": []}
    else:
        base_url      = _get(cfg, "post_training", "server", "base_url", required=True).rstrip("/")
        chat_endpoint = _get(cfg, "post_training", "server", "chat_endpoint", default="/v1/chat/completions")
        timeout_s     = _get(cfg, "post_training", "server", "timeout_s", default=60)
        chat_url      = f"{base_url}{chat_endpoint}"

        print(f"[tests] running {len(test_cases)} test case(s) against {chat_url}")
        results = []
        any_fail = False

        for tc in test_cases:
            name    = tc.get("name", "unnamed")
            system  = tc.get("system", "")
            user    = tc.get("user", "")
            pattern = tc.get("expect_pattern", "")
            max_tok = int(tc.get("max_tokens", 256))
            temp    = float(tc.get("temperature", 0.0))

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})

            payload_d = {
                "messages":    messages,
                "max_tokens":  max_tok,
                "temperature": temp,
                "stream":      False,
            }
            payload = json.dumps(payload_d).encode()

            print(f"  [{name}] ...", end="", flush=True)
            if args.dry_run:
                print(f" [dry-run] would POST to {chat_url}")
                results.append({"name": name, "status": "dry-run", "pattern": pattern})
                continue

            try:
                req = urlrequest.Request(
                    chat_url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                t0 = time.monotonic()
                with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                    body = json.loads(resp.read().decode(errors="replace"))
                elapsed = time.monotonic() - t0

                content = (
                    body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                )
                matched = bool(re.search(pattern, content, re.DOTALL)) if pattern else True
                status  = "PASS" if matched else "FAIL"
                if not matched:
                    any_fail = True

                result = {
                    "name":       name,
                    "status":     status,
                    "pattern":    pattern,
                    "matched":    matched,
                    "elapsed_s":  round(elapsed, 2),
                    "response":   content[:1000],
                }
                print(f" {status}  ({elapsed:.1f}s)")
                if not matched:
                    print(f"    pattern  : {pattern!r}")
                    print(f"    response : {content[:300]!r}")

            except (HTTPError, URLError) as e:
                detail = str(e)
                print(f" ERROR: {detail}")
                result = {"name": name, "status": "ERROR", "detail": detail}
                any_fail = True

            results.append(result)

        pass_count = sum(1 for r in results if r.get("status") == "PASS")
        fail_count = len(results) - pass_count
        print(f"[tests] {pass_count}/{len(results)} passed")

        report["steps"]["tests"] = {
            "status":     "ok" if not any_fail else "failed",
            "passed":     pass_count,
            "failed":     fail_count,
            "total":      len(results),
            "results":    results,
        }
        _save_report()

# ── Final verdict ─────────────────────────────────────────────────────────────
steps = report["steps"]
all_ok = all(
    s.get("status") in ("ok", "skipped", "dry-run")
    for s in steps.values()
)
report["verdict"] = "PASS" if all_ok else "FAIL"
_save_report()

print(f"\n[post_train] verdict={report['verdict']}  report={report_path}")
if not all_ok:
    sys.exit(2 if steps["tests"].get("status") == "failed" else 1)
