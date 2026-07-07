# OpenVINO GenAI Performance Tracer

Profile VLM pipeline execution at two levels and merge them into a single timeline:

1. **GenAI-level trace** — high-level pipeline events (which model ran and for how long)
2. **CLIntercept GPU trace** — low-level GPU kernel dispatches (gemm, resample, eltwise, etc.)

## Quick Start

### Step 1: Capture GenAI Trace

Set `OPENVINO_CHROME_TRACE` to an **absolute path** before running your app:

```bash
OPENVINO_CHROME_TRACE=/tmp/genai_trace.json \
  python benchmark_vlm.py -m $MODEL_DIR -d GPU --cb_config '...'
```

### Step 2: Capture CLIntercept GPU Kernel Trace

Use `cliloader` with `ChromePerformanceTiming` enabled:

```bash
CLI_ChromePerformanceTiming=1 \
CLI_InOrderQueue=1 \
  cliloader --dump-dir /tmp/cli_dump/ \
  python benchmark_vlm.py -m $MODEL_DIR -d GPU --cb_config '...'
```

This produces `/tmp/cli_dump/clintercept_trace.json`.

**Important:** Both traces must come from the **same process invocation**. Run your app once with both `OPENVINO_CHROME_TRACE` and `cliloader` enabled simultaneously:

```bash
OPENVINO_CHROME_TRACE=/tmp/genai_trace.json \
CLI_ChromePerformanceTiming=1 \
CLI_InOrderQueue=1 \
  cliloader --dump-dir /tmp/cli_dump/ \
  python benchmark_vlm.py -m $MODEL_DIR -d GPU --cb_config '...'
```

### Step 3: Merge Traces

```bash
python tools/perf_tracer/merge_traces.py \
  /tmp/genai_trace.json \
  /tmp/cli_dump/clintercept_trace.json \
  --normalize \
  -o /tmp/merged_trace.json
```

### Step 4: View in Perfetto

Open the merged JSON in https://ui.perfetto.dev/ or `chrome://tracing`.

You will see two process rows:
- **OpenVINO GenAI Pipeline** — nested model execution events
- **GPU Kernels (CLIntercept)** — individual kernel dispatches on GPU queues

Events are time-aligned so you can directly correlate which GPU kernels belong to which model call.

## CLIntercept Trace Events

GPU kernel events appear on separate thread lanes per OpenCL queue. Common kernels:

| Kernel Pattern | Model Stage |
|----------------|-------------|
| `resample_bfyx_cubic_opt` | Vision encoder (image resize) |
| `gen_conv`, `conv_*` | Vision encoder (patch embedding) |
| `gemm_kernel` | All model stages (matmul) |
| `sdpa_opt`, `pa_sdpa_opt` | Language model (attention) |
| `rms_gpu_*` | Language model (RMSNorm) |

## merge_traces.py Options

```
usage: merge_traces.py [-h] [-o OUTPUT] [--normalize] genai_trace cli_trace

positional arguments:
  genai_trace    Path to GenAI chrome trace JSON
  cli_trace      Path to CLIntercept trace JSON

options:
  -o, --output   Output path (default: merged_trace.json next to genai trace)
  --normalize    Normalize timestamps to start from 0 (recommended)
```

## Timestamp Alignment

Both traces use the same `steady_clock` time base:
- GenAI trace stores absolute microseconds from `steady_clock::time_since_epoch()`
- CLIntercept stores a `clintercept_start_time` metadata event (absolute steady_clock us) and all kernel timestamps are offsets from that value

The merge script converts CLIntercept relative timestamps to absolute, then optionally normalizes both to start from 0.

## Notes

- `CLI_InOrderQueue=1` is recommended for CLIntercept to get accurate per-kernel timing without queue overlap.
- For very long runs, the merged file can be large (>100MB). Consider limiting `--infer_count` when capturing traces.
- For GenAI trace event types, hierarchy, per-model metrics, and pipeline path details, see [SKILL.md](SKILL.md).
