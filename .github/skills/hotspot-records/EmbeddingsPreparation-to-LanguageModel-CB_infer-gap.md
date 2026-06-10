---
name: VLM CB Pipeline: EmbeddingsPreparation → LanguageModel(CB_infer) Gap
description: Progress record for the ~8ms hotspot between EmbeddingsPreparation and LanguageModel(CB_infer) in the Qwen3-VL CB pipeline. Covers trace instrumentation rounds, breakdown of CB_Schedule / CB_ForwardPrep / DARK1 costs, and next steps.
---

# VLM CB Pipeline: EmbeddingsPreparation → LanguageModel(CB_infer) Gap

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL 7B |
| Input | 448×448×2 images, 512 output tokens, Chinese prompt |
| Backend | CM (Paged Attention, `cm_path`) |
| KV cache | fp16 (`cfg_none`) |
| Trace file | `trace.448x448x2.512_cn.cm_path.cfg_none/merged_trace.json` |

## Observed gap

A ~8ms dead zone appears between `VisionEmbeddingsMerger(Qwen3VL)` end and the first
`LanguageModel(CB_infer)` start on every steady-state inference. Zero GPU kernels and
zero CPU trace events fall inside it before instrumentation.

| Call | Gap |
|------|-----|
| Warmup #1 | ~41 ms (JIT compilation) |
| Warmup #2 | ~27 ms |
| Steady #3 | **~8.4 ms** |
| Steady #4 | **~8.2 ms** |

## Final breakdown (steady-state average, ~8.5 ms total)

```
LMExtraInputsSetup.end
  │
  ├── 2,525 µs  (29.7%)  DARK1 — untraced CB generate() entry / add_request() / SequenceGroup ctor
  │
  ├──     0 µs  ( 0.0%)  CB_PullAwaitingRequests  (timestamp marker only)
  │
  ├── 2,897 µs  (34.1%)  CB_Schedule — m_scheduler->schedule(): PA KV-cache block allocation
  │                       (no child spans yet; opaque)
  │
  └── CB_Forward
        ├── 2,855 µs  (33.6%)  CB_ForwardPrep — per-token embedding copy loop
        │                       ~4K tokens × 3584 floats × 4B ≈ 57 MB memcpy @ ~20 GB/s ≈ 2.9 ms
        │                       (proportional to sequence length; real throughput bottleneck)
        ├──   206 µs  ( 2.4%)  CB_SetTensors — set_tensor() calls on InferRequest
        └──   ~1 µs            → LanguageModel(CB_infer)
```

## Root causes

| Rank | Segment | Cost | Root cause |
|------|---------|------|------------|
| 1 | **CB_Schedule** | ~2.9 ms | PA block allocation for a large prefill (~4K tokens). Proportional to number of KV blocks needed. |
| 2 | **CB_ForwardPrep** | ~2.9 ms | Sequential per-token copy of embedding vectors into a contiguous input buffer (`inputs_embeds_data`). ~57 MB of float memcpy on a single thread. |
| 3 | **DARK1** | ~2.5 ms | Gap between `lm_extra_inputs_list` being filled and `_pull_awaiting_requests()` firing. Lives inside `ContinuousBatchingImpl::generate()` → `add_request()` → `SequenceGroup` construction (copies position_ids list) + optional prefix-cache restore. |
| 4 | **CB_SetTensors** | ~0.2 ms | `m_request.set_tensor()` calls. Acceptable. |

## Instrumentation added

### Round 1 — post-EmbeddingsPreparation spans

Files: `pipeline_base.cpp`, `pipeline.cpp` (stateful path, not exercised here)

| Span | File | What it covers |
|------|------|----------------|
| `PositionIdsSetup` | `pipeline_base.cpp` | `get_position_ids()` call |
| `LMExtraInputsSetup` | `pipeline_base.cpp` | `get_lm_extra_inputs()` / `deep_copy_tensors_map(...)` |
| `PerfMetricsCollection` | `pipeline.cpp` | metric struct fill (stateful only) |
| `KVCacheManagement` | `pipeline.cpp` | `trim_kv_cache`, `reset_state` (stateful only) |
| `PromptIdsSetup` | `pipeline.cpp` | `SequenceGroup` ctor (stateful only) |
| `AttentionAndPositionSetup` | `pipeline.cpp` | attention mask alloc + `get_position_ids` (stateful only) |

Result: only `PositionIdsSetup` (~1 µs) and `LMExtraInputsSetup` (~518 µs) appeared.
Remaining dark zone: **~6.8 ms** after `LMExtraInputsSetup`.

### Round 2 — CB step loop spans

Files: `pipeline_impl.cpp` (`#include "chrome_trace.hpp"` added), `model_runner.hpp`

| Span | File | What it covers |
|------|------|----------------|
| `CB_PullAwaitingRequests` | `pipeline_impl.cpp:step()` | `_pull_awaiting_requests()` |
| `CB_Schedule` | `pipeline_impl.cpp:step()` | `m_scheduler->schedule(m_requests)` |
| `CB_Forward` | `pipeline_impl.cpp:step()` | entire `m_model_runner->forward()` call |
| `CB_ForwardPrep` | `model_runner.hpp:forward()` | tensor alloc + per-token embedding copy loop |
| `CB_SetTensors` | `model_runner.hpp:forward()` | all `m_request.set_tensor(...)` calls |

`CB_ForwardPrep` uses `ScopedTrace::end()` explicitly to end before `CB_SetTensors` starts,
since both live in the same scope.

## Source locations

| Span | File | Lines |
|------|------|-------|
| `LMExtraInputsSetup` | `pipeline_base.cpp` | ~424–434 |
| DARK1 start | `pipeline_base.cpp` | after `lm_extra_inputs_list.push_back(...)` |
| DARK1 end / `CB_PullAwaitingRequests` | `pipeline_impl.cpp` | ~355 |
| `CB_Schedule` | `pipeline_impl.cpp` | ~362–381 |
| `CB_ForwardPrep` start | `model_runner.hpp` | ~399 (after tensor count loop) |
| `CB_ForwardPrep` end | `model_runner.hpp` | ~686 (after per-token copy loop) |
| `CB_SetTensors` start | `model_runner.hpp` | ~691 |
| `CB_SetTensors` end | `model_runner.hpp` | ~771 |
| `LanguageModel(CB_infer)` | `model_runner.hpp` | ~774 |

## Next investigation steps

1. **DARK1**: Add `ScopedTrace` around `add_request()` loop in `pipeline_impl.cpp:533–555`
   and around `SequenceGroup` ctor to confirm where the 2.5 ms goes.

2. **CB_Schedule**: Add child spans inside `Scheduler::schedule()` to identify whether
   cost is in block allocation, compaction, or eviction logic.

3. **CB_ForwardPrep fix candidates**:
   - Replace element-wise `std::copy_n` with `std::memcpy` / SIMD copy
   - Store embeddings in a pre-allocated contiguous buffer inside `SequenceGroup` so
     `model_runner` can use it directly (zero-copy path)
   - Parallelize over sequences with a thread pool for multi-sequence batches
