---
name: openvino-genai-chrome-trace
description: Profile OpenVINO GenAI VLM pipeline execution with Chrome Trace Event Format. Use when debugging TTFT, model execution timing, or performance bottlenecks in VLM inference (Qwen3-VL, Qwen2-VL, etc.). Also use when the user asks about tracing, profiling, or visualizing model execution timelines in openvino.genai.
---

# OpenVINO GenAI VLM Chrome Tracing

## Activation

### Method 1: Explicit trace file path

```bash
OPENVINO_CHROME_TRACE=/absolute/path/to/output_trace.json ./your_vlm_app
```

Use an absolute path. Scripts like `benchmark_vlm.py` may `cd` internally, making relative paths unreliable.

## Viewing the Trace

Open the output JSON in `chrome://tracing` (Chrome) or https://ui.perfetto.dev/

## Trace Events

All events are on the same thread (`tid=1`) so they nest automatically in Perfetto.

Events come from two sources:
- **ScopedTrace** (real-time wall-clock): fires as code runs, records actual start + duration
- **Reconstructed** (after-the-fact from metrics): placed retroactively using stored durations and timepoints

### ScopedTrace events

| Event | Pipeline Path | Source |
|-------|--------------|--------|
| `EncodeImages` | Both | `pipeline.cpp`, `pipeline_base.cpp` |
| `EncodeVideos` | Both | `pipeline.cpp`, `pipeline_base.cpp` |
| `EmbeddingsPreparation` | Both | `pipeline.cpp`, `pipeline_base.cpp` |
| `VisionEncoder(ov_preprocess)` | Both | `qwen2vl/classes.cpp` |
| `VisionEncoder(cpp_preprocess)` | Both | `qwen2vl/classes.cpp` |
| `Tokenizer(encode)` | Both | `inputs_embedder.cpp` |
| `TextEmbeddings` | Both | `embedding_model.cpp` |
| `VisionEmbeddingsMerger` | Both | `qwen2vl/classes.cpp` |
| `VisionEmbeddingsMerger(Qwen3VL)` | Both | `qwen3_vl/classes.cpp` |
| `VisionEmbeddingsPos(patched)` | Both | `qwen3_vl/classes.cpp` |
| `VisionEmbeddingsPos(original)` | Both | `qwen3_vl/classes.cpp` |
| `LanguageModel(prefill)` | Stateful only | `lm_encoding.cpp` |
| `LanguageModel(CB_infer)` | CB only | `model_runner.hpp` |

### Reconstructed events

| Event | Duration source | Position source | Source file |
|-------|----------------|-----------------|-------------|
| `Generate` | `stop_time - start_time` (real timepoints) | `[start_time, stop_time]` | `pipeline.cpp`, `continuous_batching_adapter.hpp` |
| `TTFT` | `m_times_to_first_token[0]` | `start_time + duration` | same |
| `TPOT_phase(avg=X.XXms)` | `stop_time - m_new_token_times[0]` (real timepoints) | `[m_new_token_times[0], stop_time]` | same |

Note: `EmbeddingsPreparationTime` is **not** a reconstructed trace event — `EmbeddingsPreparation` is a real ScopedTrace on both paths. The INFO log still prints `EmbeddingsPreparationTime: X ms` as a duration summary only.

## Trace Hierarchy (as seen in Perfetto)

```
Generate                                         [reconstructed]
├── TTFT                                         [reconstructed]
│   ├── EncodeImages                             [ScopedTrace]
│   │   └── VisionEncoder(ov_preprocess) × N    [ScopedTrace] → openvino_vision_embeddings_model.xml
│   ├── EmbeddingsPreparation                    [ScopedTrace]
│   │   ├── Tokenizer(encode)                    [ScopedTrace] → openvino_tokenizer.xml
│   │   ├── TextEmbeddings                       [ScopedTrace] → openvino_text_embeddings_model.xml
│   │   ├── VisionEmbeddingsMerger(Qwen3VL)      [ScopedTrace] → openvino_vision_embeddings_merger_model.xml
│   │   └── VisionEmbeddingsPos(patched)         [ScopedTrace] → openvino_vision_embeddings_pos_model.xml
│   └── LanguageModel(prefill / CB_infer)        [ScopedTrace] → openvino_language_model.xml
└── TPOT_phase(avg=X.XXms)                       [reconstructed]
      ├── TextEmbeddings (per token)             [ScopedTrace]
      └── LanguageModel (per token)              [ScopedTrace]
```

`EncodeImages` wraps the per-image loop; `VisionEncoder` fires once per tile inside it. For Qwen3-VL dynamic resolution a single image may produce multiple tiles (e.g. 8).

## Pipeline Paths

Two execution paths, each with a string-prompt and ChatHistory sub-path:

- **Stateful** (no `scheduler_config`): LM inference via `lm_encoding.cpp`
  - String prompt: `pipeline.cpp:308`
  - ChatHistory: `pipeline.cpp:501`
- **ContinuousBatching (CB)** (with `scheduler_config`, e.g. `--cb_config`): LM inference via `model_runner.hpp`. Used by `benchmark_vlm.py`.
  - String prompt non-chat: `pipeline_base.cpp` else-branch
  - String prompt chat: `pipeline_base.cpp` if-branch
  - ChatHistory: `pipeline_base.cpp` ChatHistory overload
  - Adapter finalize: `continuous_batching_adapter.hpp:finalize_decoded_results`

## How TTFT Breakdown Metrics Are Measured

Both metrics are consistent across paths and aligned with ScopedTrace events:

| Metric | Definition | Aligned ScopedTrace |
|--------|-----------|---------------------|
| `prepare_embeddings_durations` | `get_inputs_embeds` call only: Tokenizer + TextEmbeddings + Merger + Pos. Does NOT include EncodeImages, EncodeVideos, normalize_prompt, apply_chat_template, or get_position_ids. | `EmbeddingsPreparation` ScopedTrace (both paths) |
| `lm_prefill_durations` | LM forward during prefill only. Does NOT include scheduler, sampler, or append_embeddings. | `LanguageModel(CB_infer)` / `LanguageModel(prefill)` ScopedTrace |

`lm_prefill_durations` sources:
- Stateful: `m_token_infer_durations[0]` = just `m_llm.infer()` (`lm_encoding.cpp`)
- CB: `m_inference_durations[0]` = sum of `m_model_runner->forward()` across all prefill steps (`continuous_batching_adapter.hpp`). Accumulated via a `first_token_produced` flag in `pipeline_impl.cpp` — every step until and including the one that produces the first token is counted. This handles both multi-chunk prefill and the common single-step case (when `max_num_batched_tokens` is large, e.g. `sys.maxsize` in `benchmark_vlm.py`, the entire prompt is processed in one step that simultaneously produces the first token).

### Timeline (both paths)

```
generate_start_time
│
├── EncodeImages / EncodeVideos / normalize_prompt   (IN TTFT, NOT in any breakdown metric)
│
├── EmbeddingsPreparation ScopedTrace ─────────────  prepare_embeddings_durations START
│     ├── Tokenizer(encode)
│     ├── TextEmbeddings
│     ├── VisionEmbeddingsMerger
│     └── VisionEmbeddingsPos
├── ─────────────────────────────────────────────── prepare_embeddings_durations END
│
├── [get_position_ids, set_tensor, add_request, scheduler]
│
├── LanguageModel ScopedTrace ──────────────────────  lm_prefill_durations
│     └── m_llm.infer() / m_model_runner->forward()
│
├── [sampler, append_embeddings — NOT in any breakdown metric]
│
└── m_new_token_times[0] ───────────────────────────  TTFT END
```

`EmbPrep + LM_prefill < TTFT` on both paths. The gap = EncodeImages + normalize_prompt + setup + scheduler + sampler + append_embeddings.

## Per-Model Metrics (No Trace Required)

Per-model TTFT breakdown is available in `perf_metrics` without enabling tracing. Populated on all paths including CB chat (all four execution sub-paths set `mm.reset(); mm.collecting = true` before `EmbeddingsPreparation`).

```python
perf_metrics = res.perf_metrics
print(f"Vision encoder:  {perf_metrics.get_vision_encoder_duration().mean:.2f} ms")
print(f"Tokenizer:       {perf_metrics.get_tokenizer_duration().mean:.2f} ms")
print(f"Text embeddings: {perf_metrics.get_text_embeddings_duration().mean:.2f} ms")
print(f"Merger:          {perf_metrics.get_vision_embeddings_merger_duration().mean:.2f} ms")
print(f"Position:        {perf_metrics.get_vision_embeddings_pos_duration().mean:.2f} ms")
print(f"LM prefill:      {perf_metrics.get_lm_prefill_duration().mean:.2f} ms")
```

Accumulate across multiple `generate()` calls:
```python
total = res1.perf_metrics + res2.perf_metrics
```

## How `ModelMetrics` Accumulation Works

`ScopedTrace` destructor routes its measured duration into a thread-local `ModelMetrics` accumulator by name prefix (`chrome_trace.hpp`):

| Name prefix | Accumulator field |
|-------------|------------------|
| `VisionEncoder*` | `vision_encoder_us` |
| `Tokenizer*` | `tokenizer_us` |
| `TextEmbeddings` | `text_embeddings_us` |
| `VisionEmbeddingsMerger*` | `vision_embeddings_merger_us` |
| `VisionEmbeddingsPos*` | `vision_embeddings_pos_us` |

This only fires when `get_model_metrics().collecting == true`. The gate is set differently per path:

| Path | `mm.collecting = true` placed before | `VisionEncoder` captured? |
|------|--------------------------------------|--------------------------|
| Stateful (both sub-paths) | `EncodeImages` | Yes |
| CB non-chat | `EncodeImages` | Yes |
| CB chat (string prompt) | `EmbeddingsPreparation` | No |
| CB ChatHistory | `EmbeddingsPreparation` | No |

For CB chat paths, `VisionEncoder` fires before `mm.collecting` is armed, so `vision_encoder_durations` is always zero. The other fields (tokenizer, text_embeddings, merger, pos) are captured on all paths.

Note: `EmbeddingsPreparation` itself matches no prefix, so it does not accumulate — it only emits the ScopedTrace event.

## Hidden Cost: `apply_chat_template` (~24ms on LNL)

In the **non-chat CB path** with `apply_chat_template=True` (default), a ~24ms CPU gap appears at the start of `EmbeddingsPreparation` before the first `Tokenizer(encode)` event. Zero GPU kernels fire during this gap.

Call chain inside `get_inputs_embeds`:
1. `qwen3_vl/classes.cpp` — `get_encoded_input_ids(unified_prompt, metrics)`
2. → `inputs_embedder.cpp` — `apply_chat_template_tokenize(prompt, metrics)`
3. → `inputs_embedder.cpp` — **`m_tokenizer.apply_chat_template()`** ← ~24ms Jinja2 render cost
4. → `inputs_embedder.cpp` — `Tokenizer(encode)` starts after template is rendered

The `unified_prompt` string is huge — `normalize_prompt` inserts hundreds of `<|image_pad|>` tokens per tile, producing a string with thousands of characters. The minja Jinja2 engine must build an `nlohmann::json` object and render it. ~24ms on Core Ultra 7 268V, ~10-15ms on desktop.

**Workaround:** pre-template outside `generate()` to move the cost outside the timed path:

```python
config = ov_genai.GenerationConfig()
config.max_new_tokens = 512
config.apply_chat_template = False  # skip re-templating inside generate()

tokenizer = pipe.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], add_generation_prompt=True)

res = pipe.generate(prompt, images=images, generation_config=config)
```

The chat path does not have this issue — `apply_chat_template` is called before `EmbeddingsPreparation` starts.

## INFO Log Output

When `OPENVINO_CHROME_TRACE` is set, each ScopedTrace also prints to the logger on destruction:
```
[INFO] [TRACE] [HH:MM:SS.mmm] EventName: X.XXX ms
```
Aggregate metrics (Generate, TTFT, EmbeddingsPreparationTime, TPOT_phase) are also logged at the end of each `generate()` call. No log output is produced when `OPENVINO_CHROME_TRACE` is not set.

## Warmup Effect

The first inference on each model is ~100x slower (e.g. ~300ms vs ~2-4ms for VisionEncoder) due to GPU JIT kernel compilation. Subsequent calls use cached kernels.

## GPU Kernel vs Model XML

`openvino_vision_embeddings_model.xml` does NOT contain explicit Resize ops, yet CLIntercept shows `resample_bfyx_cubic_opt` kernels. The `ov_preprocess` path patches an `ov::op::v8::If` node at compile time. The GPU plugin decomposes it into two sub-networks:
- **net_id 2**: Image preprocessing (u8→f32 + bicubic resize + normalize + tile) — source of `resample_bfyx_cubic_opt`
- **net_id 3**: The actual ViT encoder from the XML

The `cpp_preprocess` path does resizing in C++, so no resize kernels appear in GPU traces.

If op condition `cond_img_vid` (scalar f32): `1.0` = image (then_body: resize+normalize+tile), `0.0` = video (else_body: two frames resize+normalize+concat). Source: `qwen2vl/classes.cpp`.

## Source Files

All in `src/cpp/src/`:
- `chrome_trace.hpp` — `ChromeTrace` singleton, `ScopedTrace` RAII, `ModelMetrics` accumulator
- `visual_language/pipeline.cpp` — Stateful pipeline: ScopedTraces + aggregate metric emission
- `visual_language/continuous_batching_adapter.hpp` — CB adapter: `finalize_decoded_results`, aggregate metric emission
- `continuous_batching/pipeline_base.cpp` — CB path: EncodeImages/Videos, `EmbeddingsPreparation`, per-model metrics
- `continuous_batching/model_runner.hpp` — CB path: `LanguageModel(CB_infer)` ScopedTrace
- `lm_encoding.cpp` — Stateful path: `LanguageModel(prefill)` ScopedTrace
- `visual_language/qwen2vl/classes.cpp` — VisionEncoder, Merger, If op preprocessing
- `visual_language/qwen3_vl/classes.cpp` — Qwen3VL Merger, VisionEmbeddingsPos
- `visual_language/embedding_model.cpp` — TextEmbeddings
- `visual_language/inputs_embedder.cpp` — Tokenizer(encode)
- `visual_language/perf_metrics.hpp` — `VLMRawPerfMetrics` / `VLMPerfMetrics` definitions
- `visual_language/perf_metrics.cpp` — `evaluate_statistics`
- `python/py_vlm_pipeline.cpp` — Python bindings for VLM metrics

## Adding New Trace Points

```cpp
#include "chrome_trace.hpp"

// Outside namespace ov::genai — use qualified name:
{
    ov::genai::ScopedTrace trace("MyModelName");
    my_infer_request.infer();
}

// Inside namespace ov::genai:
{
    ScopedTrace trace("MyModelName");
    my_infer_request.infer();
}
```

The name prefix determines which `ModelMetrics` accumulator field the duration routes to (see `chrome_trace.hpp`).

## Build

```bash
source ~/openvino.venv/bin/activate
cd ~/openvino.genai
bash build.sh
```
