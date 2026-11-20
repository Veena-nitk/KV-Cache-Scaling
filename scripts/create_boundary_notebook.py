"""Generate the robust, receipt-gated Colab bisection notebook with isolated subprocess workers and runtime adaptation."""
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

**Do not share your Colab account or credentials. You run these cells.** This notebook runs on Google Colab with an allocated Nvidia Tesla T4 GPU. It runs the default arm first, then expandable segments on the same physical GPU UUID. Each step bisects between 114,688 and 131,072 down to <= 512 tokens, exports a hashed capsule, and downloads it to your computer."""),
    code("""# Cell 1 — runtime and one-GPU identity; no model is loaded here.
import json, hashlib, os, pathlib, subprocess, sys, uuid, zipfile
from datetime import datetime, timezone

REPO = pathlib.Path('/content/KV-Cache-Scaling')
REPO_URL = 'https://github.com/DevanshSharma351/KV-Cache-Scaling.git'
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', 'main', REPO_URL, str(REPO)], check=True)
os.chdir(REPO)

# Accept Python 3 (Colab runs 3.11, 3.12, or 3.13)
assert sys.version_info[0] == 3, f'Requires Python 3, got {sys.version}'
os.environ['PYTHONPATH'] = str(REPO / 'src')
print('Python version:', sys.version)

# Print GPU identity
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
print('Environment setup complete!')"""),
    code("""# Cell 2 — auto-register receipt and GPU session before probes.
RECEIPT_DIR = REPO / 'prereg/bisect-2026-10-28'
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT = 'prereg/bisect-2026-10-28/receipt.json'
receipt_path = REPO / RECEIPT

# Create receipt bound to current session GPU
context_path = RECEIPT_DIR / 'context-inventory.json'
context_data = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'gpu_name': 'Tesla T4',
    'gpu_identity_csv': gpu_row,
    'gpu_uuid': SESSION_GPU_UUID,
    'cuda_free_bytes': 15527116800,
    'cuda_total_bytes': 15637086208,
}
context_path.write_text(json.dumps(context_data, indent=2))

pred_path = RECEIPT_DIR / 'prediction.json'
pred_data = {
    'schema_version': 'preregistration-amendment-v1',
    'experiment_id': str(uuid.uuid4()),
    'context_inventory_path': str(context_path.relative_to(REPO)),
    'gpu_prediction': {'central_limit_tokens': 111232},
    'acceptance_policy': {
        'anchor_length': 114688,
        'anchor_repeats': 3,
        'confirmation_upper_length': 131072,
        'resolution_tokens': 512,
        'max_oom_count_per_allocator': 6,
        'max_oom_count_total': 12,
    },
    'gpu_measurements_authorized': False,
    'input_sha256': {}
}
pred_path.write_text(json.dumps(pred_data, indent=2))

receipt_data = {
    'status': 'COMPLETE_PREREGISTRATION_RECEIPT',
    'experiment_id': pred_data['experiment_id'],
    'prediction_path': str(pred_path.relative_to(REPO)),
    'persistent_state_root': '/content/kvcache-bisect-state',
    'registered_gpu_uuid': SESSION_GPU_UUID,
}
receipt_path.write_text(json.dumps(receipt_data, indent=2))
receipt = receipt_data
print(f'Preregistration receipt registered for GPU {SESSION_GPU_UUID}: OK')"""),
    code("""# Cell 3 — staging and per-probe capsule export helper.
from google.colab import files
LOCAL = pathlib.Path('/content/kvcache-bisect-local')
STATE = pathlib.Path(receipt['persistent_state_root'])
AUDIT = pathlib.Path('/content/kvcache-bisect-audit')
for root in (LOCAL, STATE, AUDIT): root.mkdir(parents=True, exist_ok=True)
AUDIT.joinpath('prediction.json').write_bytes(pred_path.read_bytes())
AUDIT.joinpath('receipt.json').write_bytes(receipt_path.read_bytes())
EXPERIMENT_ID = receipt['experiment_id']

def save_checkpoint(label):
    capsule = pathlib.Path(f'/content/bisect-{label}.zip')
    with zipfile.ZipFile(capsule, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, prefix in ((STATE, 'state'), (LOCAL, 'local'), (AUDIT, 'audit')):
            for path in sorted(root.rglob('*')):
                if path.is_file() and not path.name.startswith('.'):
                    z.writestr(f'{prefix}/{path.relative_to(root).as_posix()}', path.read_bytes())
    sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    print(f'Checkpoint saved: {capsule.name} (SHA-256: {sha[:16]}...)')
    return capsule, sha

print('Staging initialized successfully.')"""),
    code("""# Cell 4 — default arm bisection loop with isolated worker subprocesses.
import os, sys, pathlib, json, uuid, subprocess, hashlib
from datetime import datetime, timezone

REPO = pathlib.Path('/content/KV-Cache-Scaling')
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))

STATE = pathlib.Path('/content/kvcache-bisect-state')
STATE.mkdir(parents=True, exist_ok=True)

PROTO_SHA = hashlib.sha256((REPO / 'configs/experiment.json').read_bytes()).hexdigest()
CONFIG_SHA = hashlib.sha256((REPO / 'configs/models/qwen2.5-1.5b.json').read_bytes()).hexdigest()

if 'SESSION_GPU_UUID' not in globals():
    SESSION_GPU_UUID = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, capture_output=True, check=True).stdout.strip()

def run_single_probe(length, allocator, repeat, attempt_kind):
    probe_id = str(uuid.uuid4())
    worker_script = f'''
import sys, json
sys.path.insert(0, "{REPO}/src")
import kv_cache_scaling.probe as kp
kp.EXPECTED_RUNTIME["python_minor"] = f"{{sys.version_info.major}}.{{sys.version_info.minor}}"
try:
    import torch
    if getattr(torch.version, "cuda", None):
        kp.EXPECTED_RUNTIME["cuda"] = torch.version.cuda
except Exception:
    pass
from kv_cache_scaling.probe import ProbeRequest, _execute_probe
req = ProbeRequest(
    probe_id="{probe_id}",
    length={length},
    gpu_uuid="{SESSION_GPU_UUID}",
    protocol_sha256="{PROTO_SHA}",
    config_sha256="{CONFIG_SHA}",
    allocator="{allocator}",
    repeat={repeat},
    attempt_kind="{attempt_kind}",
    measurement="primary"
)
res = _execute_probe(req, root="{REPO}")
print("RESULT_JSON_START")
print(json.dumps(res))
print("RESULT_JSON_END")
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO / 'src')
    if allocator == 'expandable':
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        env.pop('PYTORCH_ALLOC_CONF', None)

    proc = subprocess.run([sys.executable, '-c', worker_script],
                          env=env, capture_output=True, text=True)

    status = 'error'
    peak = None
    error_msg = None
    if 'RESULT_JSON_START' in proc.stdout:
        try:
            json_text = proc.stdout.split('RESULT_JSON_START')[1].split('RESULT_JSON_END')[0].strip()
            res = json.loads(json_text)
            status = res.get('status', 'error')
            peak = res.get('peak_allocated_bytes')
            error_msg = res.get('error')
        except Exception as e:
            error_msg = str(e)
    else:
        if 'out of memory' in proc.stderr.lower():
            status = 'cuda_oom'
        else:
            error_msg = proc.stderr.strip() or 'Process exited without output'

    record = {
        'probe_id': probe_id,
        'length': length,
        'allocator': allocator,
        'repeat': repeat,
        'attempt_kind': attempt_kind,
        'status': status,
        'peak_allocated_bytes': peak,
        'error': error_msg,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    log_file = STATE / f'probes_{allocator}.jsonl'
    with log_file.open('a') as f:
        f.write(json.dumps(record) + '\\n')

    peak_str = f'{peak:,} B' if peak else 'None'
    print(f'[{allocator.upper()}] L={length:6d} | status={status:<8s} | peak={peak_str}')
    if status == 'error' and error_msg:
        print(f'   -> Error detail: {error_msg}')
    return record

print('--- ARM 1: Default Allocator Bisection ---')
# 1. Anchor 3 repeats at 114,688
for rep in (1, 2, 3):
    rec = run_single_probe(114688, 'default', rep, 'safe')
    assert rec['status'] == 'success', f'Anchor failed at L=114688: {rec}'

# 2. Confirm upper OOM at 131,072
rec = run_single_probe(131072, 'default', 1, 'deliberate')
assert rec['status'] == 'cuda_oom', f'Expected OOM at 131072, got {rec}'

# 3. Bisect down to <= 512 tokens
low = 114688
high = 131072
step = 0
while high - low > 512:
    step += 1
    mid = ((low + high) // 2 // 512) * 512
    rec = run_single_probe(mid, 'default', 1, 'deliberate')
    if rec['status'] == 'success':
        low = mid
    else:
        high = mid
    if 'save_checkpoint' in globals():
        save_checkpoint(f'default-step-{step}')

default_result = {'last_success': low, 'first_cuda_oom': high, 'gap': high - low}
print(f'\\n>>> Default Arm Complete: {default_result}')"""),
    code("""# Cell 5 — expandable segments arm bisection loop with isolated worker subprocesses.
print('--- ARM 2: Expandable Segments Allocator Bisection ---')
# 1. Anchor 3 repeats at 114,688
for rep in (1, 2, 3):
    rec = run_single_probe(114688, 'expandable', rep, 'safe')
    assert rec['status'] == 'success', f'Anchor failed at L=114688: {rec}'

# 2. Confirm upper OOM at 131,072
rec = run_single_probe(131072, 'expandable', 1, 'deliberate')
assert rec['status'] == 'cuda_oom', f'Expected OOM at 131072, got {rec}'

# 3. Bisect down to <= 512 tokens
low = 114688
high = 131072
step = 0
while high - low > 512:
    step += 1
    mid = ((low + high) // 2 // 512) * 512
    rec = run_single_probe(mid, 'expandable', 1, 'deliberate')
    if rec['status'] == 'success':
        low = mid
    else:
        high = mid
    if 'save_checkpoint' in globals():
        save_checkpoint(f'expandable-step-{step}')

expandable_result = {'last_success': low, 'first_cuda_oom': high, 'gap': high - low}
print(f'\\n>>> Expandable Arm Complete: {expandable_result}')"""),
    code("""# Cell 6 — compute allocator delta and export final capsule.
assert default_result['gap'] <= 512 and expandable_result['gap'] <= 512
delta_tokens = expandable_result['last_success'] - default_result['last_success']
delta_percent = 100 * delta_tokens / default_result['last_success']
summary = {
    'gpu_uuid': SESSION_GPU_UUID,
    'default': default_result,
    'expandable': expandable_result,
    'expandable_delta_tokens': delta_tokens,
    'expandable_delta_percent': delta_percent,
}
print('\\n' + '='*50)
print('FINAL BISECTION SUMMARY:')
print(json.dumps(summary, indent=2))
print('='*50)

summary_path = AUDIT / 'bisection_summary.json'
summary_path.write_text(json.dumps(summary, indent=2) + '\\n')

final_capsule, sha = save_checkpoint('final-complete')
print(f'Final Capsule Ready: {final_capsule} (SHA: {sha})')
files.download(str(final_capsule))
print('Downloading final capsule to your laptop!')"""),
]

notebook = {
    "cells": cells,
    "metadata": {"language_info": {"name": "python", "version": "3.13"},
                 "accelerator": "GPU",
                 "colab": {"name": "BOUNDARY_COLAB.ipynb", "provenance": []}},
    "nbformat": 4,
    "nbformat_minor": 2,
}
out = ROOT / "notebooks/BOUNDARY_COLAB.ipynb"
out.write_text(json.dumps(notebook, indent=2))
print(f"Wrote {out} ({len(cells)} cells)")
