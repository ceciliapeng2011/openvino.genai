#!/usr/bin/env python3
"""
Merge OpenVINO GenAI chrome trace and CLIntercept GPU kernel trace into a
single Perfetto-compatible JSON file.

GenAI trace: high-level pipeline events (VisionEncoder, TextEmbeddings, etc.)
CLIntercept trace: low-level GPU kernel dispatches (gemm, eltwise, resample, etc.)

Both use steady_clock time base:
  - GenAI: absolute microseconds from steady_clock epoch
  - CLIntercept: microseconds relative to clintercept_start_time (also steady_clock epoch)

The merged file uses a common time base so events align in Perfetto UI.

Usage:
    python merge_traces.py <genai_trace.json> <clintercept_trace.json> [-o merged.json]
    python merge_traces.py <genai_trace.json> <clintercept_trace.json> --normalize
"""

import argparse
import json
import sys


def load_json_events(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_cli_start_time(cli_events):
    for ev in cli_events:
        if ev.get("name") == "clintercept_start_time":
            return ev["args"]["start_time"]
    return None


def merge_traces(genai_path, cli_path, output_path, normalize=False):
    genai_events = load_json_events(genai_path)
    cli_events = load_json_events(cli_path)

    cli_start_us = extract_cli_start_time(cli_events)
    if cli_start_us is None:
        print("ERROR: clintercept_start_time not found in CLIntercept trace.", file=sys.stderr)
        sys.exit(1)

    # GenAI events already have absolute steady_clock timestamps in "ts" field.
    # CLIntercept events have timestamps relative to cli_start_us.
    # Convert CLIntercept events to absolute steady_clock timestamps.

    # Find the earliest timestamp across both traces for normalization
    genai_min_ts = None
    for ev in genai_events:
        if "ts" in ev and isinstance(ev["ts"], (int, float)):
            if genai_min_ts is None or ev["ts"] < genai_min_ts:
                genai_min_ts = ev["ts"]

    cli_first_ts = None
    for ev in cli_events:
        if ev.get("ph") == "X" and "ts" in ev:
            abs_ts = cli_start_us + ev["ts"]
            if cli_first_ts is None or abs_ts < cli_first_ts:
                cli_first_ts = abs_ts

    global_min_ts = min(
        genai_min_ts if genai_min_ts is not None else float('inf'),
        cli_first_ts if cli_first_ts is not None else float('inf')
    )
    if normalize:
        offset = global_min_ts
    else:
        offset = 0

    merged = []

    # Process metadata: separate processes for GenAI (pid=1) and GPU kernels (pid=2)
    merged.append({
        "ph": "M", "name": "process_name",
        "pid": 1, "tid": 0,
        "args": {"name": "OpenVINO GenAI Pipeline"}
    })
    merged.append({
        "ph": "M", "name": "process_sort_index",
        "pid": 1, "tid": 0,
        "args": {"sort_index": 0}
    })
    merged.append({
        "ph": "M", "name": "process_name",
        "pid": 2, "tid": 0,
        "args": {"name": "GPU Kernels (CLIntercept)"}
    })
    merged.append({
        "ph": "M", "name": "process_sort_index",
        "pid": 2, "tid": 0,
        "args": {"sort_index": 1}
    })

    # Add GenAI events under pid=1
    for ev in genai_events:
        out = dict(ev)
        out["pid"] = 1
        if "ts" in out and isinstance(out["ts"], (int, float)):
            out["ts"] = out["ts"] - offset
        merged.append(out)

    # Track CLIntercept thread names
    cli_thread_names = {}
    for ev in cli_events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            tid = ev.get("tid")
            tname = ev.get("args", {}).get("name", "")
            cli_thread_names[tid] = tname

    # Add CLIntercept thread metadata under pid=2
    for tid, tname in cli_thread_names.items():
        merged.append({
            "ph": "M", "name": "thread_name",
            "pid": 2, "tid": tid,
            "args": {"name": tname}
        })

    # Add CLIntercept kernel events under pid=2, converting timestamps to absolute
    for ev in cli_events:
        if ev.get("ph") not in ("X", "B", "E"):
            continue
        out = dict(ev)
        out["pid"] = 2
        if "ts" in out:
            out["ts"] = cli_start_us + out["ts"] - offset
        merged.append(out)

    with open(output_path, 'w') as f:
        json.dump(merged, f)

    n_genai = len(genai_events)
    n_cli = sum(1 for ev in cli_events if ev.get("ph") in ("X", "B", "E"))
    print(f"Merged {n_genai} GenAI events + {n_cli} GPU kernel events -> {output_path}")
    print(f"  GenAI time range: {genai_min_ts - offset:.0f} us (first event)")
    if cli_first_ts:
        print(f"  GPU time range:   {cli_first_ts - offset:.0f} us (first kernel)")
    if normalize:
        print(f"  Normalized: all timestamps shifted by -{offset:.0f} us")
    print(f"  Open in: chrome://tracing or https://ui.perfetto.dev/")


def main():
    parser = argparse.ArgumentParser(
        description="Merge OpenVINO GenAI trace + CLIntercept trace into unified Perfetto JSON")
    parser.add_argument("genai_trace", help="Path to GenAI chrome trace JSON")
    parser.add_argument("cli_trace", help="Path to CLIntercept trace JSON")
    parser.add_argument("-o", "--output", default=None,
                        help="Output merged trace path (default: merged_trace.json in same dir as genai trace)")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize timestamps to start from 0 (easier to read in Perfetto)")
    args = parser.parse_args()

    if args.output is None:
        import os
        d = os.path.dirname(args.genai_trace) or "."
        args.output = os.path.join(d, "merged_trace.json")

    merge_traces(args.genai_trace, args.cli_trace, args.output, args.normalize)


if __name__ == "__main__":
    main()
