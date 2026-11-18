"""Fresh-process worker entry for the approved, receipt-gated bisection cohort."""
import argparse
import json
import os
from pathlib import Path
import sys

from .probe import ProbeRequest, _execute_probe, write_probe_result
from .bisect_gate import load_verified_bisection, verify_bisection_launch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request = ProbeRequest.from_dict(json.loads(Path(args.request).read_text()))
    try:
        verified = load_verified_bisection(args.root, args.receipt)
        verify_bisection_launch(verified, request, args.run_id)
    except Exception as error:
        result = {
            "schema_version": "primary-probe-v1",
            "probe_id": request.probe_id,
            "status": "authorization_error",
            "stage": "receipt",
            "length": request.length,
            "error": {"type": type(error).__name__, "message": str(error)},
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        write_probe_result(args.output, result)
        return 1
    try:
        result = _execute_probe(request, root=args.root)
    except Exception as error:
        result = {
            "schema_version": "primary-probe-v1",
            "probe_id": request.probe_id,
            "status": "error",
            "stage": "worker",
            "length": request.length,
            "error": {"type": type(error).__name__, "message": str(error)},
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    write_probe_result(args.output, result)
    return 0 if result["status"] == "success" else (2 if result["status"] == "cuda_oom" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
