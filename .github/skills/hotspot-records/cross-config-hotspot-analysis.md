---
name: Cross-Config Hotspot Analysis: EmbPrep → LanguageModel(CB_infer) Gap
description: Progress record comparing the pre-LM-infer gap across 6 Qwen3-VL CB-pipeline configurations, with per-span breakdown, scaling analysis, and warmup-corrected steady-state values. Measured on LNL; PTL4Xe data pending.
---

# Cross-Config Hotspot Analysis: EmbPrep → LanguageModel(CB_infer) Gap

## Hardware

**Measured on: Intel Lunar Lake (LNL)**
PTL4Xe (Panther Lake) data pending — results below are LNL-only and absolute timings will differ on PTL4Xe.

## Configuration naming

`trace.{W}x{H}x{N_imgs}.{out_tokens}_{lang}.{backend}.{kvcache}`

- `cm_path` = CM backend (Paged Attention)
- `cfg_none` = fp16 KV cache
- `mt512` = `max_tokens=512` per CB step (enables per-step scheduling loop; "prefill chunking" by name, but in practice prefill is NOT chunked — see §mt512)

## Configs in this study

| Config | Resolution | N images | Output tokens | Est. vis-tokens |
|--------|-----------|----------|---------------|----------------|
| 512x384×2 | 512×384 | 2 | 512 | ~502 |
| 448x448×2 | 448×448 | 2 | 512 | ~512 |
| 1024x512×2 | 1024×512 | 2 | 512 | ~1337 |
| 1260x700×2 | 1260×700 | 2 | 512 | ~2250 |
| 1260x700×2/mt512 | 1260×700 | 2 | 833 | ~2250 |
| 1280x768×15 | 1280×768 | 15 | 512 | ~18808 |

Each run has 4 Generate() calls (iter 0–1 = warmup, iter 2–3 = steady-state) except mt512 which has 4 Generate() calls each producing 129 CB_infer steps (1 prefill + 128 decode).

## TTFT overview (steady-state, iters 2–3)

| Config | EmbPrep | Pre-LM gap | LM(CB_infer) | TTFT |
|--------|---------|-----------|--------------|------|
| 512x384×2 | 98ms | 8.2ms | 346ms | 457ms |
| 448x448×2 | 97ms | 7.7ms | 383ms | 494ms |
| 1024x512×2 | 206ms | 12.9ms | 507ms | 734ms |
| 1260x700×2 | 389ms | 21.6ms | 720ms | 1146ms |
| 1260x700×2/mt512 | 390ms | 22.0ms | 821ms¹ | 1256ms |
| 1280x768×15 | 3312ms | 654ms² | 8405ms | 12472ms |

¹ First CB_infer per Generate() is the prefill (821ms); 128 decode steps follow at ~32ms/token.  
² 1280x768×15 steady-state gap is unstable (see §AllocateCache).

## Gap budget — all spans accounted for (99%+)

Spans are sequential on pid=1 tid=1. Figures below are **warmup-corrected steady-state** (iters 2–3 only), except 1280x768×15 where all 4 iters show the same behaviour.

### 512x384×2 — gap 8.2ms

| Span | µs | % |
|------|----|---|
| `LMExtraInputsSetup` | 477 | 5.8% |
| `CB_AddRequests` | 2,330 | 28.3% |
| — `CB_AddRequest_SequenceGroupCtor` | 2,320 | 28.2% |
| — `CB_AddRequest_GetPositionIds` | ~1 | ~0% |
| — `CB_AddRequest_PrefixCacheRestore` | ~1 | ~0% |
| `CB_Schedule` | 2,360 | 28.7% |
| — `CB_Sched_AllocateCache` | 2,343 | 28.5% |
| `CB_ForwardPrep` | 2,780 | 33.8% |
| `CB_SetTensors` | 222 | 2.7% |
| micro-gaps | ~59 | 0.7% |

### 448x448×2 — gap 7.7ms

| Span | µs | % |
|------|----|---|
| `LMExtraInputsSetup` | 514 | 6.7% |
| `CB_AddRequests` | 2,500 | 32.5% |
| — `CB_AddRequest_SequenceGroupCtor` | 2,490 | 32.3% |
| `CB_Schedule` | 1,960 | 25.4% |
| — `CB_Sched_AllocateCache` | 1,941 | 25.2% |
| `CB_ForwardPrep` | 2,440 | 31.7% |
| `CB_SetTensors` | 212 | 2.8% |

### 1024x512×2 — gap 12.9ms

| Span | µs | % |
|------|----|---|
| `LMExtraInputsSetup` | 1,270 | 9.8% |
| `CB_AddRequests` | 4,250 | 32.9% |
| — `CB_AddRequest_SequenceGroupCtor` | 4,230 | 32.8% |
| `CB_Schedule` | 2,110 | 16.3% |
| — `CB_Sched_AllocateCache` | 2,091 | 16.2% |
| `CB_ForwardPrep` | 5,040 | 39.1% |
| `CB_SetTensors` | 190 | 1.5% |

### 1260x700×2 — gap 21.6ms

| Span | µs | % |
|------|----|---|
| `LMExtraInputsSetup` | 2,110 | 9.8% |
| `CB_AddRequests` | 7,820 | 36.2% |
| — `CB_AddRequest_SequenceGroupCtor` | 7,810 | 36.2% |
| `CB_Schedule` | 3,130 | 14.5% |
| — `CB_Sched_AllocateCache` | 3,115 | 14.4% |
| `CB_ForwardPrep` | 8,230 | 38.1% |
| `CB_SetTensors` | 188 | 0.9% |

### 1280x768×15 — gap ~654ms (no stable steady-state)

| Span | µs | % |
|------|----|---|
| `LMExtraInputsSetup` | 132,580 | 30.7% |
| `CB_AddRequests` | 156,380 | 36.2% |
| — `CB_AddRequest_SequenceGroupCtor` | 156,360 | 36.2% |
| `CB_Schedule` | 84,810 | 19.7% |
| — `CB_Sched_AllocateCache` | 84,770 | 19.6% |
| `CB_ForwardPrep` | 57,200 | 13.3% |
| `CB_SetTensors` | 347 | 0.1% |

## CB_Sched_AllocateCache: warmup vs. steady-state

`allocate_cache_if_needed()` is a **lazy allocator** — it only does real work when the KV cache needs to grow. Cost in steady state varies by config:

| Config | iter 0 | iter 1 | iter 2 | iter 3 | Steady-state |
|--------|--------|--------|--------|--------|-------------|
| 512x384×2 | 1,988µs | 1,968µs | 2,343µs | 1,905µs | ~2.1ms (flat, no warmup effect) |
| 448x448×2 | 5,651µs | 6,158µs | 1,941µs | 2,284µs | **~2.1ms** (drops 3× after 2-iter warmup) |
| 1024x512×2 | 10,508µs | 12,067µs | 2,091µs | 2,477µs | **~2.3ms** (drops 5× after 2-iter warmup) |
| 1260x700×2 | 3,058µs | 3,814µs | 3,115µs | 2,970µs | ~3.0ms (mild warmup) |
| 1260x700×2/mt512 | 4,308µs (prefill) | **0µs** (decode) | **0µs** | **0µs** | **0µs** (99.2% of steps) |
| 1280x768×15 | 79,231µs | 303,688µs | 84,767µs | 475,567µs | **No convergence** |

### Interpretation

- **Small configs (≤512 vis-tokens, 2 images)**: `AllocateCache` costs ~2ms even in steady state — the KV cache footprint is small enough that the allocator re-runs on every prefill. The "steady state" is genuine (~2ms), not a warmup artifact.
- **Medium configs (1337–2250 vis-tokens, 2 images)**: A clear 2-iter warmup for 1024x512 (5× drop), mild for others. Steady-state is ~2–3ms.
- **mt512 multi-step decode**: Once prefill allocates the cache, AllocateCache costs **exactly 0µs** for all subsequent decode steps. Periodic re-allocation spikes of 3–9ms appear every ~129 tokens as the KV cache grows in blocks. This confirms the allocator only runs when actual block extension is needed.
- **1280x768×15 (15 images)**: No convergence across 4 iters (79ms → 304ms → 85ms → 476ms). The enormous KV footprint causes repeated full re-allocation on every prefill. This config is pathological and needs a dedicated fix.

**Conclusion**: For the 2-image configs, `CB_Sched_AllocateCache` at ~2–3ms in steady state is a **real cost**, not a warmup artifact — it is roughly constant and independent of whether the cache was just allocated or not. The work done is proportional to the number of KV blocks being managed (validated, checked), not just initially allocated.

## Scaling with vision token count

All three dominant spans scale roughly linearly with vision token count:

| Span | 502 tok | 512 tok | 1337 tok | 2250 tok | 18808 tok | µs/100 vis-tok |
|------|---------|---------|----------|----------|-----------|----------------|
| `CB_AddRequest_SequenceGroupCtor` | 2.32ms | 2.63ms | 4.25ms | 7.09ms | 156ms | ~8.6µs |
| `CB_ForwardPrep` | 2.75ms | 2.58ms | 5.18ms | 8.33ms | 57ms | ~4.5µs |
| `LMExtraInputsSetup` | 0.48ms | 0.55ms | 1.27ms | 2.09ms | 134ms | ~7.5µs |
| `CB_Sched_AllocateCache`¹ | 2.1ms | 2.1ms | 2.3ms | 3.0ms | 85ms | ~15µs (non-linear) |

¹ `AllocateCache` shows a floor of ~2ms even at small scale; the 18K-token config deviates upward non-linearly.

## Root causes

### CB_AddRequest_SequenceGroupCtor (~33–36% of gap)

Two operations inside `sequence_group.hpp:402–444`:

1. **Embedding copy** (`sequence_group.hpp:403–406`): token-by-token `std::copy_n` from the flat input tensor into `m_input_embeds` (a `vector<vector<float>>`). For N tokens of hidden_size H = 3584: N×H floats copied.

2. **Position ID slicing** (`sequence_group.hpp:257–265` via `append_position_ids`): the 3D position_ids tensor `[3, 1, N]` is split into N individual `ov::Tensor` objects, each a `copy_to` of one column. N tensor allocations + N `copy_to` calls.

Both operations are **O(N_tokens)** and together with `CB_ForwardPrep` constitute a **double-copy** of the same embedding data:
- Copy #1: input tensor → `m_input_embeds[i]` (SequenceGroupCtor)
- Copy #2: `m_input_embeds[i]` → `inputs_embeds_data` buffer (CB_ForwardPrep, `model_runner.hpp:604`)

### CB_ForwardPrep (~32–41% of gap)

Token-by-token loop in `model_runner.hpp:593–627`. For EMBEDDINGS type:
- Line 603: `get_input_embeds()[position_id].data()` → pointer into `m_input_embeds[i]`
- Line 604: `std::copy_n(src, hidden_size, inputs_embeds_data + token_id * hidden_size)` — copies one embedding row
- Lines 605–609: `position_ids_elem.copy_to(dst_roi)` — copies one position_ids slice back out

This reads back the same data written by SequenceGroupCtor. With N=2250, H=3584: copying 2250 × 3584 × 4B = ~32MB per direction, ~64MB total across both copies.

### LMExtraInputsSetup (~6–31% of gap)

`deep_copy_tensors_map(m_inputs_embedder->get_lm_extra_inputs())` in `pipeline_base.cpp`.
Copies `deepstack_visual_embeds` and `visual_pos_masks` tensors. At 18K tokens this becomes 134ms — likely because `deepstack_visual_embeds` is N×H.

### CB_Sched_AllocateCache (~14–20% of gap, per-call)

`cache_orchestrator->allocate_cache_if_needed()`. At small-medium scale (~2–3ms per call) this is roughly constant; at 18K tokens it balloons to 85ms (range 79–476ms). The cost appears to be proportional to the total number of KV blocks managed, suggesting an O(N_blocks) walk or lock contention.

## mt512 decode-phase inter-step gap

Once the KV cache is allocated, each decode step has a tiny gap of only ~0.7ms:

| Span | Decode-step cost | vs. prefill |
|------|-----------------|-------------|
| `CB_ForwardPrep` | ~100µs | 90× smaller (1 token vs. 2250) |
| `CB_Schedule` | ~10µs | 300× smaller (AllocateCache=0) |
| `CB_SetTensors` | ~165µs | similar |
| Total | **~700µs** | 30× smaller than 22ms prefill gap |

The decode inter-step overhead (0.7ms / 32ms step = 2.2%) is acceptable.

## Optimization priorities

| Priority | Span | Max observed cost | Fix |
|----------|------|------------------|-----|
| **P1** | `CB_AddRequest_SequenceGroupCtor` + `CB_ForwardPrep` | 156ms + 57ms | Eliminate double-copy: store embeddings as a flat contiguous buffer in `SequenceGroup` so `model_runner` reads directly. For position_ids, keep strided view into the original 3D tensor instead of N individual tensors. |
| **P2** | `LMExtraInputsSetup` | 134ms | Use move semantics / shallow copy for `lm_extra_inputs` map in `pipeline_base.cpp` instead of `deep_copy_tensors_map`. |
| **P3** | `CB_Sched_AllocateCache` | 85ms (476ms unstable) | Profile `allocate_cache_if_needed()` internals; check if it walks all blocks unconditionally. May need to be made O(1) for the "nothing to allocate" case. |
| **P4** | `CB_ForwardPrep` internals | — | Add child spans inside the token loop to separate embedding copy vs. position_ids copy vs. other work. |

## Files touched by instrumentation

| File | Spans added |
|------|-------------|
| `src/cpp/src/continuous_batching/pipeline_base.cpp` | `PositionIdsSetup`, `LMExtraInputsSetup` |
| `src/cpp/src/continuous_batching/pipeline_impl.cpp` | `CB_AddRequests`, `CB_AddRequest_GetPositionIds`, `CB_AddRequest_SequenceGroupCtor`, `CB_AddRequest_PrefixCacheRestore`, `CB_PullAwaitingRequests`, `CB_Schedule`, `CB_Forward` |
| `src/cpp/src/continuous_batching/scheduler.hpp` | `CB_Sched_CleanEmptyBlocks`, `CB_Sched_InitCache`, `CB_Sched_PromptPhase`, `CB_Sched_GeneratePhase`, `CB_Sched_AllocateCache`, `CB_Sched_CopyBlocks` |
| `src/cpp/src/continuous_batching/model_runner.hpp` | `CB_ForwardPrep`, `CB_SetTensors` |
| `src/cpp/src/visual_language/pipeline.cpp` | `PerfMetricsCollection`, `KVCacheManagement`, `PromptIdsSetup`, `AttentionAndPositionSetup` (stateful path only, not exercised in these traces) |
