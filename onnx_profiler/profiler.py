import json
from collections import defaultdict

import numpy as np


def aggregate_profile(profile_path, num_runs, num_warmup):
    with open(profile_path, "r") as f:
        profile_data = json.load(f)

    events = profile_data if isinstance(profile_data, list) else profile_data.get("traceEvents", [])

    node_durations = defaultdict(list)
    node_op_type = {}
    node_provider = {}
    all_trace_events = []

    model_run_count = 0
    raw_trace = []
    for ev in events:
        if "dur" not in ev or "ts" not in ev:
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "unknown")
        op = ""
        provider = ""
        if "args" in ev:
            op = ev["args"].get("op_name", "")
            provider = ev["args"].get("provider", "")
        raw_trace.append({
            "name": name,
            "cat": cat,
            "op": op,
            "provider": provider,
            "ts": ev["ts"],
            "dur": ev["dur"],
            "tid": ev.get("tid", 0),
            "pid": ev.get("pid", 0),
        })
        if cat == "Node":
            if op:
                node_op_type[name] = op
            if provider:
                node_provider[name] = provider

    model_runs = sorted(
        [e for e in raw_trace if e["name"] == "model_run"],
        key=lambda e: e["ts"],
    )
    # num_runs=0 の場合 (JSON 直接指定): 全 model_run を計測扱い
    actual_num_runs = num_runs if num_runs > 0 else max(len(model_runs) - num_warmup, 0)

    model_run_labels = {}
    for i, mr in enumerate(model_runs):
        if i < num_warmup:
            model_run_labels[mr["ts"]] = f"model_run (Warmup {i+1}/{num_warmup})"
        else:
            run_idx = i - num_warmup + 1
            model_run_labels[mr["ts"]] = f"model_run (Run {run_idx}/{actual_num_runs})"

    for e in raw_trace:
        if e["name"] == "SequentialExecutor::Execute":
            continue
        if e["name"] == "model_run" and e["ts"] in model_run_labels:
            e["label"] = model_run_labels[e["ts"]]
        all_trace_events.append(e)

    profiled_runs = model_runs[num_warmup:]
    profiled_ranges = [(mr["ts"], mr["ts"] + mr["dur"]) for mr in profiled_runs]

    for e in raw_trace:
        if e["cat"] != "Node":
            continue
        for run_start, run_end in profiled_ranges:
            if e["ts"] >= run_start and e["ts"] + e["dur"] <= run_end:
                node_durations[e["name"]].append(e["dur"])
                break

    rows = []
    for name, durs in node_durations.items():
        avg = np.mean(durs)
        op_type = node_op_type.get(name, "")
        provider = node_provider.get(name, "")
        rows.append({
            "name": name,
            "op_type": op_type,
            "provider": provider,
            "avg_us": float(avg),
            "min_us": float(np.min(durs)),
            "max_us": float(np.max(durs)),
            "std_us": float(np.std(durs)),
            "count": len(durs),
        })

    rows.sort(key=lambda r: r["avg_us"], reverse=True)
    total_avg = sum(r["avg_us"] for r in rows)
    for r in rows:
        r["pct"] = r["avg_us"] / total_avg * 100 if total_avg > 0 else 0

    op_type_totals = defaultdict(float)
    for r in rows:
        op_type_totals[r["op_type"] or "(unknown)"] += r["avg_us"]
    op_type_rows = sorted(op_type_totals.items(), key=lambda x: x[1], reverse=True)
    op_type_summary = [
        {"op_type": op, "avg_us": dur, "pct": dur / total_avg * 100 if total_avg > 0 else 0}
        for op, dur in op_type_rows
    ]

    # EP (Execution Provider) 別サマリー
    ep_totals = defaultdict(float)
    ep_counts = defaultdict(int)
    for r in rows:
        ep = r["provider"] or "(unknown)"
        ep_totals[ep] += r["avg_us"]
        ep_counts[ep] += 1
    ep_summary = sorted(
        [{"provider": ep, "avg_us": dur, "pct": dur / total_avg * 100 if total_avg > 0 else 0,
          "node_count": ep_counts[ep]}
         for ep, dur in ep_totals.items()],
        key=lambda x: x["avg_us"], reverse=True,
    )

    avg_run_us = float(np.mean([r["dur"] for r in profiled_runs])) if profiled_runs else 0.0

    node_trace = [e for e in all_trace_events if e["cat"] == "Node"]
    per_run_op = []
    for ri, mr in enumerate(profiled_runs):
        run_start = mr["ts"]
        run_end = mr["ts"] + mr["dur"]
        op_dur = defaultdict(float)
        for n in node_trace:
            if n["ts"] >= run_start and n["ts"] + n["dur"] <= run_end:
                op_dur[n["op"] or "(unknown)"] += n["dur"]
        sorted_ops = sorted(op_dur.items(), key=lambda x: x[1], reverse=True)
        per_run_op.append({
            "run_index": ri + 1,
            "run_dur_us": mr["dur"],
            "ops": [{"op_type": op, "dur_us": dur} for op, dur in sorted_ops],
        })

    return {
        "total_avg_us": total_avg,
        "avg_run_us": avg_run_us,
        "num_runs": actual_num_runs,
        "num_nodes": len(rows),
        "nodes": rows,
        "op_type_summary": op_type_summary,
        "ep_summary": ep_summary,
        "trace_events": all_trace_events,
        "per_run_op": per_run_op,
    }


def build_flamegraph_tree(agg):
    """ノード名の階層パスから flamegraph 用ツリーを構築."""
    tree = {"name": "root", "value": 0, "children": {}}

    for r in agg["nodes"]:
        parts = r["name"].strip("/").split("/")
        node = tree
        for part in parts:
            if part not in node["children"]:
                node["children"][part] = {"name": part, "value": 0, "children": {}}
            node = node["children"][part]
        node["value"] += r["avg_us"]

    def finalize(node):
        children = list(node["children"].values())
        total = node["value"] + sum(finalize(c) for c in children)
        node["value"] = total
        node["children"] = sorted(children, key=lambda c: c["value"], reverse=True)
        return total

    finalize(tree)
    return tree
