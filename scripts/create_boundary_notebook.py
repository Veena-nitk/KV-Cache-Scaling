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

**Do not share your Colab account or credentials. You run these cells.** This notebook only dispatches after a TA-approved amendment and a completed, pushed receipt. It runs the default arm first, then expandable segments on the same physical GPU UUID. Each invocation performs one probe, exports a hashed capsule, downloads it, and waits for you to confirm the external save before continuing.

If the receipt is still a template, the registered UUID differs from the allocated T4, or the runtime is not Python 3.11 / PyTorch 2.6 + CUDA 12.4 / Transformers 4.51.3, stop without running any probe."""),
    code("""# Cell 1 — runtime and one-GPU identity; no model is loaded here.
import json, hashlib, os, pathlib, subprocess, sys, uuid, zipfile
from datetime import datetime, timezone

REPO = pathlib.Path('/content/KV-Cache-Scaling')
REPO_URL = 'https://github.com/DevanshSharma351/KV-Cache-Scaling.git'
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', 'main', REPO_URL, str(REPO)], check=True)
os.chdir(REPO)
assert sys.version_info[:2] == (3, 11), f'Requires Python 3.11, got {sys.version}'
assert subprocess.run(['git', 'status', '--porcelain'], text=True, capture_output=True,
                      check=True).stdout.strip() == '', 'Use a clean clone of the reviewed receipt commit'
os.environ['PYTHONPATH'] = str(REPO / 'src')
print(sys.version)
print(subprocess.run(['nvidia-smi', '-L'], text=True, capture_output=True, check=True).stdout)
gpu_row = subprocess.run(
    ['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version', '--format=csv,noheader'],
    text=True, capture_output=True, check=True).stdout.strip()
assert gpu_row.count('GPU-') == 1 and ('Tesla T4' in gpu_row or ', T4,' in gpu_row), gpu_row
SESSION_GPU_UUID = gpu_row.split(',')[0].strip()
print('Physical GPU:', gpu_row)

# Install the pinned CUDA 12.4 stack before importing torch.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.6.0',
                '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-gpu.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)"""),
    code("""# Cell 2 — require the final receipt and exact registered GPU before probes.
RECEIPT = 'prereg/bisect-2026-10-28/receipt.json'
receipt_path = REPO / RECEIPT
assert receipt_path.is_file(), 'Receipt is missing; do not use receipt.template.json'
receipt = json.loads(receipt_path.read_text())
assert receipt.get('status') == 'COMPLETE_PREREGISTRATION_RECEIPT', 'Draft/template receipt cannot authorize probes'
prediction = json.loads((REPO / receipt['prediction_path']).read_text())
context = json.loads((REPO / prediction['context_inventory_path']).read_text())
registered_uuid = context['gpu_identity_csv'].split(',')[0].strip()
assert SESSION_GPU_UUID.lower() == registered_uuid.lower(), (
    f'Colab UUID differs from the pushed context inventory: {SESSION_GPU_UUID} != {registered_uuid}. Stop and preregister this allocation.')
gate = subprocess.run([sys.executable, '-m', 'kv_cache_scaling', 'check-receipt',
                       '--receipt', RECEIPT, '--root', str(REPO)],
                      cwd=REPO, text=True, capture_output=True)
print(gate.stdout)
assert gate.returncode == 0, gate.stderr

# Confirm importing the pinned runtime does not initialize CUDA in that child.
probe = subprocess.run([sys.executable, '-c',
    "from kv_cache_scaling.probe import _import_runtime; r=_import_runtime(); "
    "print(r.torch.__version__,r.torch.version.cuda,r.transformers.__version__,r.torch.cuda.is_initialized()); "
    "assert not r.torch.cuda.is_initialized()"], cwd=REPO, text=True, capture_output=True)
print(probe.stdout)
assert probe.returncode == 0, probe.stderr"""),
    code("""# Cell 3 — staging, per-probe capsule export, and external-save acknowledgement.
from google.colab import files
LOCAL = pathlib.Path('/content/kvcache-bisect-local')
STATE = pathlib.Path(receipt['persistent_state_root'])
AUDIT = pathlib.Path('/content/kvcache-bisect-audit')
for root in (LOCAL, STATE, AUDIT): root.mkdir(parents=True, exist_ok=True)
AUDIT.joinpath('prediction.json').write_bytes((REPO / receipt['prediction_path']).read_bytes())
AUDIT.joinpath('receipt.json').write_bytes(receipt_path.read_bytes())
EXPERIMENT_ID = receipt['experiment_id']

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

def checkpoint_and_wait(label):
    members = dict(capsule_files(STATE, 'state'))
    members.update(dict(capsule_files(LOCAL, 'local')))
    members.update(dict(capsule_files(AUDIT, 'audit')))
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    capsule = pathlib.Path(f'/content/bisect-{label}.zip')
    with zipfile.ZipFile(capsule, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items(): z.writestr(name, data)
        z.writestr('MANIFEST.json', json.dumps(manifest, sort_keys=True, indent=2))
    sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    print(f'Checkpoint: {capsule}; SHA-256={sha}; members={len(members)}')
    files.download(str(capsule))
    ack = input('After saving this download outside Colab, enter the displayed SHA-256 to allow the next probe: ').strip()
    if ack != sha: raise RuntimeError('Checkpoint save not acknowledged; stopping before next probe')
    return capsule, sha

print('If this runtime disconnects, stop the cohort; do not resume it in a new Colab runtime.')"""),
    md("""## Default allocator arm

Each script call below launches one probe only. The checkpoint download and SHA acknowledgement complete before the loop can dispatch another probe. If any step fails, stop and preserve the capsule; do not retry that probe."""),
    code("""# Cell 4 — default arm; one probe and checkpoint per loop iteration.
def existing_run_id(allocator):
    found = []
    for manifest_path in STATE.glob('runs/*/manifest.json'):
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('experiment_id') == EXPERIMENT_ID
                and manifest.get('execution_scope') == 'paired_allocator_bisection'
                and manifest.get('allocator') == allocator):
            found.append(manifest_path.parent.name)
    if len(found) > 1: raise RuntimeError(f'Multiple {allocator} runs found; inspect capsule')
    return found[0] if found else None

DEFAULT_RUN_ID = existing_run_id('default') or str(uuid.uuid4())
step = 0
while True:
    step += 1
    proc = subprocess.run([sys.executable, 'scripts/bisect_boundary.py',
        '--receipt', RECEIPT, '--run-id', DEFAULT_RUN_ID, '--local-root', str(LOCAL),
        '--allocator', 'default'], cwd=REPO, text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0: raise RuntimeError(proc.stderr or f'default step {step} failed; stop and inspect capsule')
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    checkpoint_and_wait(f'default-{step:02d}')
    if result['status'] == 'COMPLETE':
        default_result = result
        break
    assert result['status'] == 'STEP_COMPLETE', result"""),
    md("""## Expandable allocator arm

The runner checks the completed default bracket and the bound GPU identity before allowing this arm."""),
    code("""# Cell 5 — expandable arm; must be in the same Colab session and GPU UUID.
current_uuid = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
    text=True, capture_output=True, check=True).stdout.strip()
assert current_uuid.lower() == SESSION_GPU_UUID.lower(), 'GPU UUID changed; stop the cohort'
EXPANDABLE_RUN_ID = existing_run_id('expandable') or str(uuid.uuid4())
step = 0
while True:
    step += 1
    proc = subprocess.run([sys.executable, 'scripts/bisect_boundary.py',
        '--receipt', RECEIPT, '--run-id', EXPANDABLE_RUN_ID, '--local-root', str(LOCAL),
        '--allocator', 'expandable', '--default-run-id', DEFAULT_RUN_ID],
        cwd=REPO, text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0: raise RuntimeError(proc.stderr or f'expandable step {step} failed; stop and inspect capsule')
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    checkpoint_and_wait(f'expandable-{step:02d}')
    if result['status'] == 'COMPLETE':
        expandable_result = result
        break
    assert result['status'] == 'STEP_COMPLETE', result"""),
    code("""# Cell 6 — final endpoints and allocator delta.
assert default_result['gap'] <= 512 and expandable_result['gap'] <= 512
delta_tokens = expandable_result['last_success'] - default_result['last_success']
delta_percent = 100 * delta_tokens / default_result['last_success']
summary = {
    'gpu_uuid': SESSION_GPU_UUID,
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
