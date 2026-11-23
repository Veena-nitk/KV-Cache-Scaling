"""Generate the receipt-gated, per-probe-checkpoint Colab bisection notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


cells = [
    md("""# KV cache boundary bisect — Colab runbook

**Outside the assignment protocol.** The original study already used its single deliberate OOM. This notebook is for a separate draft multi-OOM study. See docs/ASSIGNMENT_COMPLETION.md for the submitted results.

**Do not share your Colab account or credentials. You run these cells.** This notebook only dispatches after a TA-approved amendment and a completed, pushed receipt. It runs the default arm first, then expandable segments on the same physical GPU UUID. Each invocation performs one probe, exports a hashed capsule, downloads it, and waits for you to confirm the external save before continuing.

If the receipt is still a template, the registered UUID differs from the allocated T4, or the runtime is not Python 3.11 / PyTorch 2.6 + CUDA 12.4 / Transformers 4.51.3, stop without running any probe."""),
    code("""# Cell 1 — runtime and one-GPU identity; no model is loaded here.
import hashlib, json, os, pathlib, shutil, subprocess, uuid, zipfile

BRANCH = os.environ.get('KV_CACHE_PAIRED_BRANCH', 'veena-submission')
RECEIPT = os.environ.get('KV_CACHE_PAIRED_RECEIPT', 'prereg/bisect-2026-10-28/receipt.json')
REPO = pathlib.Path('/content/KV-Cache-Scaling-paired')
REPO_URL = 'https://github.com/DevanshSharma351/KV-Cache-Scaling.git'
PYTHON = pathlib.Path(os.environ.get(
    'KV_CACHE_PAIRED_PYTHON', '/content/kvcache-paired-venv/bin/python'))
subprocess.run(['git', 'check-ref-format', '--branch', BRANCH], check=True)
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO)], check=True)
source = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=REPO, check=True,
                        text=True, capture_output=True).stdout.strip()
if source != REPO_URL:
    raise RuntimeError('Use a clean clone of the reviewed GitHub repository')
receipt_path = REPO / RECEIPT
if not receipt_path.is_file():
    raise RuntimeError('Missing committed, TA-approved paired receipt; stop before dependency setup or model probes')
receipt = json.loads(receipt_path.read_text())
if receipt.get('remote_branch') != BRANCH:
    raise RuntimeError('Notebook branch must match the reviewed receipt branch')

gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
gpu_rows = gpu.strip().splitlines()
if len(gpu_rows) != 1 or gpu_rows[0].split(',')[1].strip() not in ('Tesla T4', 'T4'):
    raise RuntimeError('This experiment requires exactly one Tesla T4')
SESSION_GPU_UUID = gpu_rows[0].split(',')[0].strip()

# Use a separate Python 3.11 environment. Keep Colab's notebook kernel CPU-only.
if not PYTHON.is_file():
    venv = str(PYTHON.parent.parent)
    if shutil.which('python3.11'):
        subprocess.run(['python3.11', '-m', 'venv', venv], check=True)
    elif shutil.which('uv'):
        subprocess.run(['uv', 'venv', '--python', '3.11', '--seed', venv], check=True)
    else:
        raise RuntimeError('Provide a dedicated Python 3.11 environment as described in the runbook')
ENV = dict(os.environ, PYTHONPATH=str(REPO / 'src'), USE_TF='0', USE_FLAX='0')
def run_source(*args, **kwargs):
    return subprocess.run([str(PYTHON), *args], cwd=REPO, env=ENV, **kwargs)

# Verify the receipt, preregistration, source, and exact GPU before installing dependencies.
gate_code = (
    "import sys; from kv_cache_scaling.bisect_gate import load_verified_bisection; "
    "v=load_verified_bisection('.',sys.argv[1]); "
    "assert v.gpu_uuid.lower()==sys.argv[2].lower(), 'GPU UUID differs from the frozen inventory'; "
    "print('Paired receipt and GPU identity verified')"
)
run_source('-c', "import sys; assert sys.version_info[:2]==(3,11), sys.version", check=True)
run_source('-c', gate_code, RECEIPT, SESSION_GPU_UUID, check=True)
run_source('-m', 'ensurepip', '--upgrade', check=True)
run_source('-m', 'pip', 'install', 'torch==2.6.0+cu124',
           '--index-url', 'https://download.pytorch.org/whl/cu124', check=True)
run_source('-m', 'pip', 'install', '-r', 'requirements-gpu.txt', check=True)
run_source('-c', "from kv_cache_scaling.probe import _import_runtime; "
           "r=_import_runtime(); assert not r.torch.cuda.is_initialized(); "
           "print(r.torch.__version__, r.torch.version.cuda, r.transformers.__version__)", check=True)"""),
    code("""# Cell 3 — staging, per-probe capsule export, and external-save acknowledgement.
from google.colab import files
EXPERIMENT_ID = receipt['experiment_id']
STATE = pathlib.Path(receipt['persistent_state_root'])
SESSION = pathlib.Path('/content/kvcache-paired-' + str(uuid.UUID(EXPERIMENT_ID)))
SESSION.mkdir(exist_ok=False)  # Refuse to restart an initialized cohort in this Colab session.
LOCAL = SESSION / 'local'
AUDIT = SESSION / 'audit'
for root in (LOCAL, STATE, AUDIT):
    root.mkdir(parents=True, exist_ok=True)
if (STATE / f'paired_boundary_oom_budget-{EXPERIMENT_ID}.json').exists():
    raise RuntimeError('This experiment was already initialized; inspect its external checkpoints')
AUDIT.joinpath('prediction.json').write_bytes((REPO / receipt['prediction_path']).read_bytes())
AUDIT.joinpath('receipt.json').write_bytes(receipt_path.read_bytes())
prediction = json.loads(AUDIT.joinpath('prediction.json').read_text())
AUDIT.joinpath('context-inventory.json').write_bytes((REPO / prediction['context_inventory_path']).read_bytes())


def capsule_files(root, prefix):
    # Include only this registered cohort; never repackage older run data.
    runs = root / 'runs'
    if runs.exists():
        for run_dir in sorted(runs.iterdir()):
            manifest = run_dir / 'manifest.json'
            if manifest.is_file() and json.loads(manifest.read_text()).get('experiment_id') == EXPERIMENT_ID:
                for path in sorted(run_dir.rglob('*')):
                    if path.is_file() and path.name != '.lock' and not path.name.startswith('.pending-'):
                        yield f'{prefix}/{path.relative_to(root).as_posix()}', path.read_bytes()
    for path in sorted(root.iterdir()):
        if path.is_file() and EXPERIMENT_ID in path.name and not path.name.startswith('.pending-'):
            yield f'{prefix}/{path.relative_to(root).as_posix()}', path.read_bytes()
    if prefix == 'audit':
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.name != '.lock' and not path.name.startswith('.pending-'):
                yield f'{prefix}/{path.relative_to(root).as_posix()}', path.read_bytes()

def checkpoint_and_wait(label, *, require_ack=True):
    members = dict(capsule_files(STATE, 'state'))
    members.update(dict(capsule_files(LOCAL, 'local')))
    members.update(dict(capsule_files(AUDIT, 'audit')))
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    capsule = SESSION / f'bisect-{label}.zip'
    with zipfile.ZipFile(capsule, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items(): z.writestr(name, data)
        z.writestr('MANIFEST.json', json.dumps(manifest, sort_keys=True, indent=2))
    sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    print(f'Checkpoint: {capsule}; SHA-256={sha}; members={len(members)}')
    files.download(str(capsule))
    if require_ack:
        ack = input('Verify the SHA-256 of your local download and enter that digest: ').strip().lower()
        if ack != sha:
            raise RuntimeError('Checkpoint save not acknowledged; stopping before next probe')
        with (AUDIT / 'checkpoint-acknowledgements.jsonl').open('a') as log:
            log.write(json.dumps({'capsule': capsule.name, 'sha256': sha}) + '\\n')
            log.flush()
            os.fsync(log.fileno())
    return capsule, sha

print('If this runtime disconnects, stop the cohort; do not resume it in a new Colab runtime.')"""),
    md("""## Default allocator arm

Each script call below launches one probe only. The checkpoint download and SHA acknowledgement complete before the loop can dispatch another probe. If any step fails, stop and preserve the capsule; do not retry that probe."""),
    code("""# Run each arm at most once. A failure or missing acknowledgement halts the notebook.
def run_arm(allocator, run_id, default_run_id=None):
    (SESSION / (allocator + '.started')).touch(exist_ok=False)
    args = ['scripts/bisect_boundary.py', '--receipt', RECEIPT, '--run-id', run_id,
            '--local-root', str(LOCAL), '--allocator', allocator]
    if default_run_id:
        args += ['--default-run-id', default_run_id]
    # Three anchors, one upper confirmation, five midpoints and finalization.
    for step in range(1, 11):
        proc = run_source(*args, text=True, capture_output=True)
        print(proc.stdout)
        label = f'{allocator}-{step:02d}'
        if proc.returncode:
            (AUDIT / (label + '-error.txt')).write_text(proc.stderr)
            checkpoint_and_wait(label + '-failed', require_ack=False)
            raise RuntimeError(proc.stderr or f'{allocator} step {step} failed')
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if result.get('status') not in ('STEP_COMPLETE', 'COMPLETE'):
            raise RuntimeError(f'Unexpected coordinator output: {result}')
        checkpoint_and_wait(label)
        if result['status'] == 'COMPLETE':
            return result
    raise RuntimeError('Coordinator exceeded the registered number of steps')

DEFAULT_RUN_ID = str(uuid.uuid4())
default_result = run_arm('default', DEFAULT_RUN_ID)"""),
    code("""# Both arms must use the same still-connected Colab runtime and physical GPU.
current_uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                       text=True).strip()
if current_uuid.lower() != SESSION_GPU_UUID.lower():
    raise RuntimeError('GPU UUID changed; stop the cohort')
if default_result['status'] != 'COMPLETE':
    raise RuntimeError('Default arm and its external checkpoint must be complete')
EXPANDABLE_RUN_ID = str(uuid.uuid4())
expandable_result = run_arm('expandable', EXPANDABLE_RUN_ID, DEFAULT_RUN_ID)"""),
    code("""# Cell 6 — final endpoints and allocator delta.
assert default_result['gap'] <= 512 and expandable_result['gap'] <= 512
delta_tokens = expandable_result['last_success'] - default_result['last_success']
delta_percent = 100 * delta_tokens / default_result['last_success']
summary = {
    'gpu_uuid': SESSION_GPU_UUID,
    'experiment_id': EXPERIMENT_ID,
    'run_ids': {'default': DEFAULT_RUN_ID, 'expandable': EXPANDABLE_RUN_ID},
    'default': {k: default_result[k] for k in ('last_success','first_cuda_oom','gap')},
    'expandable': {k: expandable_result[k] for k in ('last_success','first_cuda_oom','gap')},
    'expandable_delta_tokens': delta_tokens,
    'expandable_delta_percent': delta_percent,
}
print(json.dumps(summary, indent=2))
summary_path = AUDIT / 'bisection_summary.json'
summary_path.write_text(json.dumps(summary, indent=2) + '\\n')
checkpoint_and_wait('final-complete')
print('Save the final capsule as artifacts/experiments/<date>-paired-boundary/checkpoint.zip')"""),
]

notebook = {
    "cells": cells,
    "metadata": {"language_info": {"name": "python", "version": "3.11"},
                 "accelerator": "GPU",
                 "colab": {"name": "BOUNDARY_COLAB.ipynb", "provenance": []}},
    "nbformat": 4,
    "nbformat_minor": 2,
}
out = ROOT / "notebooks/BOUNDARY_COLAB.ipynb"
out.write_text(json.dumps(notebook, indent=2))
print(f"Wrote {out} ({len(cells)} cells)")
