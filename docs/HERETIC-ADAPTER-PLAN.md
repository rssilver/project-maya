# Plan: running a Heretic LoRA adapter on Maya's GLM engine

Working notes for branch `windows-ram-overflow` (fork `rssilver/project-maya`, remote `fork`). Status: planned, not
started. Context summary at the end.

## The adapter

`D:\Maya-models\ablation\glm-5.3-heretic-lora-v3.gguf` (MorinoNushi, heretic-gguf; 68 MB, F16, `adapter.lora.alpha`
= 1, rank 1, so the scale is 1). It covers layers 15-43, three kinds of tensor:

| Tensor | Layers | lora_a (per row of input) | lora_b (output, 4096) |
|---|---|---|---|
| `blk.N.attn_output.weight` | 25 | 1 x 8192 (DSA layers) or 1 x 16384 (KDA layers) | 4096 |
| `blk.N.ffn_down_shexp.weight` | 23 | 1 x 2048 (shared expert) | 4096 |
| `blk.N.ffn_down_exps.weight` | 19 | 288 x 1 x 2048 (one per expert) | 288 x 4096 (one per expert) |

Each one adds `delta = B (A . x)` to a matrix's output, where `x` is that matrix's **input**. Within a layer every
`B` points the same way (cosine >= 0.9997), but the strengths differ per component and **per routed expert**
(median 0.016, max 1.2). So the delta cannot be applied to a layer's summed output: it needs each matrix's own input.

## Where each input lives (decode)

| Matrix | Its input in the one-token decode | Format |
|---|---|---|
| attention output, DSA layers | `F->attn_q` (`fast_dsa`, glm_fast_path.cu ~5707) | q8_1, `n_head * v_head` |
| attention output, KDA layers | `F->gated_q` (glm_fast_path.cu ~6031) | q8_1, `DI` |
| shared expert down | `F->sh_hq` (written by the route kernel) | q8_1, `n_ff_sh` = 2048 |
| routed expert down, GPU entries | `F->hq`, 8 entries x 2048, entry `i` = plan entry `i` | q8_1 |
| routed expert down, CPU lane | `F->cpu_hq` after `lane_experts` (fast_cpu_experts ~2932) | native host q8 (`kc::kNativeHBytes` each) |

A plan entry with `plan_ptr == 0` is not computed on the GPU (CPU lane, CPU-only overflow, disk skip, draft miss).
`plan_id` / `plan_w` give its expert and routing weight.

## Changes

1. **Loader** (new `src/core/glm_lora.cuh`, like `glm_ablate.cuh`): `MAYA_GLM_LORA=<file.gguf>` (+ `_SCALE`). Read the
   GGUF with the engine's GGUF reader; per layer keep A and B in F16 on the GPU (~68 MB of VRAM) and in F32 on the
   host for the CPU lane (~134 MB of RAM). Refuse a file whose shapes do not match the pack (log it, run the stock model).
2. **Attention** (both layer kinds): after the output projection's `mv`, a small kernel `a = A . dequant(q8_1 input)`,
   then `F->mixer += scale * a * B`. One block; ~16K multiply-adds.
3. **Shared expert**: after the route kernel writes `sh_hq`, `a_sh = A_sh . dequant(sh_hq)`; after `fast_moe`,
   `F->ffn += scale * a_sh * B_sh`.
4. **Routed experts on the GPU**: after `moe_gate_up` fills `hq`, for each entry `i` with `plan_ptr[i] != 0`:
   `a_i = A[plan_id[i]] . dequant(hq[i])`; after `moe_down`, `F->ffn += scale * sum_i plan_w[i] * a_i * B[plan_id[i]]`.
   Check first that `hq` is not already multiplied by the routing weight, and that the CUDA-graph capture of KDA layers
   (`kda_graphs`) still replays it (it reads `plan_*` from device memory, so it should).
5. **Routed experts on the CPU lane**: in the host's CPU-lane answer (glm_fast_path.cu ~4810-4880, `fast_cpu_experts`),
   after `lane_experts` has each expert's `h` in `cpu_hq`, add `w_j * (A_e . h_j) * B_e` to the answer row. Expert ids
   come from `rq->ids[i]` for the `cpu_mask` bits. This covers the RAM overflow tier too.
6. **Prompt path**: first version, turn the batched prompt path off while the adapter is on (the switch at
   glm_prefill.cu:1050, `STRATA_GLM_NO_PREFILL`), so prompts run token by token through the hooked decode. That is exact
   but slow (prompts at decode speed). Later, hook `glm_prefill.cu`'s batched experts the same way.
7. **Not covered, logged at start**: batched rows (`--parallel` above 1, glm_rows.cu), drafter verify windows
   (glm_window.cu), deferred CPU answers (turn `MAYA_GLM_DEFER_CPU` off), the MTP draft block.
8. **Docs** in docs/SETTINGS.md; a commit on `windows-ram-overflow`.

## Checking it

- **Exactness**: dump one layer's `F->ffn` and `F->mixer` (`fast_dump`) for the same token with and without the adapter.
  The difference must equal `B * (A . x)` computed in Python from the dumped inputs (gguf-py reads the adapter).
- **No change when off**: greedy answers identical to the current build.
- **Effect and quality**: the eval harness `eval_maya.py` (it was in the old scratchpad; rewrite if gone): 16 benign
  borderline prompts, 8 answer checks, tok/s, run for Maya-S stock, + alesha-pro vector, + Heretic adapter. The benign
  set showed 0/16 refusals even for stock Maya-S, so add prompts the stock model actually refuses, or judge answers.
- **Speed**: `--bench`. Expect a few % on decode; prompts slow (step 6) until the batched path is hooked.

## Context summary (2026-11-22)

- **PC**: RTX 5090 32 GB (also drives the display, ~1.9 GB), Ryzen 9 7950X3D, 128 GB DDR5-5600, Windows 11, US 120 V
  circuit on a 10 A breaker (the PC alone can trip it; GPU kept at its 400 W floor; Balanced plan, core parking off).
- **Windows pinning cap**: the driver pins only ~half the RAM. Fixed with `MAYA_GLM_RAM_OVERFLOW=1` (commit 38a422ec):
  leftover experts in pageable RAM, flagged `gf::kCpuOnly`, computed by the CPU lane.
- **Speeds** (standard bench): Maya-M-Derisked 15.2 -> 24.2 tok/s; Maya-S 31.6 tok/s; Maya-S + alesha-pro vector
  ~31 tok/s. Short answers show ~50 tok/s on the dashboard (few tokens, warm experts), not representative.
- **Ablation**: alesha-pro's per-layer vector ported as `MAYA_GLM_ABLATE` (+ `_LAYERS=45`) in `src/core/glm_ablate.cuh`
  (commit 8bc95386). It is on in `maya-maya-s-v2-iq2_xxs.json`.
- **Models on D:\Maya-models**: Maya-M-Derisked, Maya-S-v2-IQ2_XXS, GLM-5.3-Flash-DFlash2, `ablation\` (vector +
  Heretic LoRA). Maya-S stores attention and shared experts in Q6_K, routed down weights in IQ2_S/IQ3_XXS (merging the
  adapter into the files would need Maya's unpublished imatrix: hence the runtime approach).
- **Repo**: local checkout is v1.1.49; upstream (`origin`, mw00/project-maya) is v1.1.73+. Untracked
  `maya-bench-2026-11-10.txt` is the old Nov 10 benchmark.
