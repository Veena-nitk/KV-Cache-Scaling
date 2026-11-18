# Boundary bisection Colab procedure

**No Colab account sharing is needed. You run [BOUNDARY_COLAB.ipynb](BOUNDARY_COLAB.ipynb) in your own session.** This is an execution guide, not evidence that any GPU probe ran.

## Before opening the notebook

1. Obtain TA approval for `prereg/bisect-amendment-2026-10-28/AMENDMENT_DRAFT.md`.
2. Freeze the current coordinator and worker code, regenerate the amendment prediction hashes, and create a completed receipt for a **new experiment ID**. The receipt template is not runnable.
3. The existing context inventory records a specific T4 UUID from 2026-10-19. Colab may assign a different GPU. The notebook checks that the current GPU UUID equals the receipt's frozen inventory; if not, stop and preregister a fresh context-only inventory before any model probe.
4. Push the approved amendment prediction and later receipt in the required order. Ask before each push. Do not edit old preregistration packages, receipts, or capsules.

## Run order

Open `notebooks/BOUNDARY_COLAB.ipynb` in your Colab session using a Tesla T4. Run its cells in order. It first checks the exact GPU and completed receipt. If those checks pass, it:

1. Runs the three 114,688-token anchor repeats for `default`.
2. Confirms a typed `cuda_oom` at 131,072, then launches one midpoint at a time until the endpoint gap is at most 512.
3. Exports and downloads a hash-manifested capsule after each probe. Enter the printed capsule SHA only after saving the download outside Colab; the loop will not dispatch the next point without the matching SHA acknowledgement.
4. Repeats the same steps for `expandable_segments:True`, checking the same registered GPU UUID and the completed default arm.
5. Prints both endpoint pairs and the expandable delta, then downloads a final capsule.

Use the same still-connected Colab runtime for both allocator arms. If the runtime disconnects, the GPU UUID changes, an anchor fails, the 131,072 confirmation succeeds, a worker errors/times out, or a checkpoint is missing, stop the cohort. Do not resume in a new runtime under the same receipt or retry a probe. Save every downloaded checkpoint.

The runner uses the completed receipt through `verify_receipt`, checks the amendment's pinned acceptance policy, and authorizes each worker from its durable journal entry. A per-experiment budget ledger is stored alongside the journal and preserved in every capsule. The cap is six deliberate launches per allocator, with no more than six typed OOM outcomes per allocator and twelve total. It never calls the old one-use ledger or modifies it.

## Save the final capsule

Save each intermediate download under `artifacts/experiments/2026-11-18-paired-boundary/checkpoints/` with its original `bisect-*.zip` filename. Save the final download at `artifacts/experiments/2026-11-18-paired-boundary/checkpoint.zip`, with a new README containing both run IDs, exact GPU UUID, final zip SHA-256, and TA approval reference. Keep all intermediate checkpoints. Never overwrite `artifacts/experiments/2026-10-21-boundary/checkpoint.zip`.

After verifying the capsule and its member hashes, provide it to Codex or say **RESULTS READY**. The offline follow-up will verify one UUID, recalculate both boundary pairs and the allocator delta, and update the prediction-error tables separately for the committed and fitted models.
