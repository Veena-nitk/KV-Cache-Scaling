# Boundary bisection Colab procedure

The [boundary notebook](BOUNDARY_COLAB.ipynb) is the live paired experiment. The [presentation notebook](FINAL_PRESENTATION.ipynb) is a CPU replay of saved evidence.

## Before opening the notebook

1. Obtain and record TA approval for the prospective [paired amendment](../prereg/bisect-amendment-2026-10-28/AMENDMENT_DRAFT.md).
2. In the still-connected T4 session, capture a real context-only inventory using the pinned Python 3.11, PyTorch 2.6.0+cu124, and Transformers 4.51.3 stack. Use a fresh experiment ID. Freeze the inventory, approved policy, approval reference (`ta_approval_reference`), and code hashes in a formal prediction and commit and push it before any model probe. The older draft and submitted probe JSONL cannot serve as this record.
3. Create the later completed receipt on the same GitHub branch. It must bind the same experiment ID, real prediction SHA, context inventory, code, and matching approval reference. The receipt template is a draft. Set `KV_CACHE_PAIRED_BRANCH` and `KV_CACHE_PAIRED_RECEIPT` in the Colab environment if the pushed branch or receipt path differs from the notebook defaults.
4. Use one T4 allocation for both arms. Provide `python3.11` or `uv` on the Colab PATH, or set `KV_CACHE_PAIRED_PYTHON` to an existing dedicated Python 3.11 interpreter. The notebook installs the pinned GPU packages there after the receipt and GPU UUID checks.

## Run order

Open `notebooks/BOUNDARY_COLAB.ipynb` in your Colab session using a Tesla T4. Run its cells in order. It first checks the exact GPU and completed receipt. If those checks pass, it:

1. Runs the three 114,688-token anchor repeats for `default`.
2. Confirms a typed `cuda_oom` at 131,072, then launches one midpoint at a time until the endpoint gap is at most 512.
3. Exports and downloads a hash-manifested capsule after each probe. Verify the SHA-256 of the downloaded file on your local filesystem and enter that digest; the loop will not dispatch the next point without the matching SHA acknowledgement.
4. Repeats the same steps for `expandable_segments:True`, checking the same registered GPU UUID and the completed default arm.
5. Prints both endpoint pairs and the expandable delta, then downloads a final capsule.

Use the same still-connected Colab runtime for both allocator arms. If the runtime disconnects, the GPU UUID changes, an anchor fails, the 131,072 confirmation succeeds, a worker errors/times out, or a checkpoint is missing, stop the cohort. Do not resume in a new runtime under the same receipt or retry a probe. Save every downloaded checkpoint.

The runner uses the completed receipt through `verify_receipt`, checks the amendment's pinned acceptance policy, and authorizes each worker from its durable journal entry. A per-experiment budget ledger is stored alongside the journal and preserved in every capsule. The cap is six deliberate launches per allocator, with no more than six typed OOM outcomes per allocator and twelve total. It never calls the old one-use ledger or modifies it.

## Save the final capsule

Save each intermediate download under `artifacts/experiments/2026-11-18-paired-boundary/checkpoints/` with its original `bisect-*.zip` filename. Save the final download at `artifacts/experiments/2026-11-18-paired-boundary/checkpoint.zip`, with a new README containing both run IDs, exact GPU UUID, final zip SHA-256, and TA approval reference. Keep all intermediate checkpoints. Never overwrite `artifacts/experiments/2026-10-21-boundary/checkpoint.zip`.

After verifying the capsule and its member hashes, provide it to Codex or say **RESULTS READY**. The offline follow-up will verify one UUID, recalculate both boundary pairs and the allocator delta, and compare the committed and fitted models with the new endpoints.
