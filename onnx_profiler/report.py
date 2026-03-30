import html as html_module
import json
import os
from datetime import datetime

import onnxruntime as ort

from .constants import OP_COLOR_PALETTE

def generate_html_report(agg, model_path, input_info, output_path, num_warmup,
                         flame_data=None):
    e = html_module.escape
    model_name = e(os.path.basename(model_path))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_run_ms = agg["avg_run_us"] / 1000

    if num_warmup > 0:
        runs_value = f"{num_warmup}+{agg['num_runs']}"
        runs_unit = "warmup + profiled"
    else:
        runs_value = str(agg['num_runs'])
        runs_unit = "runs (from JSON)"

    # Op Type → 色 の統一マッピング (時間降順 = op_type_summary の順)
    op_color_map = {}
    for i, o in enumerate(agg["op_type_summary"]):
        op_color_map[o["op_type"]] = OP_COLOR_PALETTE[i % len(OP_COLOR_PALETTE)]
    op_color_map_json = json.dumps(op_color_map, ensure_ascii=False)

    top_nodes = agg["nodes"][:20]
    chart_labels = json.dumps([n["name"][-50:] for n in top_nodes], ensure_ascii=False)
    chart_values = json.dumps([round(n["avg_us"] / 1000, 2) for n in top_nodes])
    # 棒グラフ: 各ノードの op_type の色を使う
    bar_colors = json.dumps([op_color_map.get(n["op_type"], "#94a3b8") for n in top_nodes])

    trace_json = json.dumps(agg["trace_events"], ensure_ascii=False)
    per_run_op_json = json.dumps(agg["per_run_op"], ensure_ascii=False)
    flame_json = json.dumps(flame_data, ensure_ascii=False) if flame_data else "null"

    input_rows_html = ""
    for info in input_info:
        input_rows_html += f"<tr><td>{e(info['name'])}</td><td>{info['shape']}</td><td>{e(info['dtype'])}</td></tr>\n"

    # EP 別サマリー HTML
    ep_summary = agg.get("ep_summary", [])
    has_ep = len(ep_summary) > 0 and not (len(ep_summary) == 1 and ep_summary[0]["provider"] == "(unknown)")
    ep_summary_html = ""
    if has_ep:
        for ep in ep_summary:
            short_name = ep["provider"].replace("ExecutionProvider", "")
            ep_summary_html += (
                f"<div class='ep-item'>"
                f"<span class='ep-badge ep-{short_name.lower()}'>{e(short_name)}</span>"
                f"<span class='ep-detail'>{ep['node_count']} nodes &mdash; "
                f"{ep['avg_us']/1000:,.2f} ms ({ep['pct']:.1f}%)</span>"
                f"</div>\n"
            )

    node_rows_html = ""
    cumulative_pct = 0.0
    for i, r in enumerate(agg["nodes"]):
        cumulative_pct += r["pct"]
        name_disp = e(r["name"])
        op_disp = e(r["op_type"])
        provider_disp = r.get("provider", "")
        ep_short = provider_disp.replace("ExecutionProvider", "") if provider_disp else ""
        bar_color = op_color_map.get(r["op_type"], "#94a3b8")
        ep_td = f"<td><span class='ep-badge ep-{ep_short.lower()}'>{e(ep_short)}</span></td>" if has_ep else ""
        node_rows_html += (
            f"<tr data-ep='{e(ep_short)}'>"
            f"<td class='rank'>{i+1}</td>"
            f"<td class='node-name' title='{name_disp}'>{name_disp}</td>"
            f"<td>{op_disp}</td>"
            f"{ep_td}"
            f"<td class='num'>{r['avg_us']/1000:,.2f}</td>"
            f"<td class='num'>{r['min_us']/1000:,.2f}</td>"
            f"<td class='num'>{r['max_us']/1000:,.2f}</td>"
            f"<td class='num'>{r['std_us']/1000:,.2f}</td>"
            f"<td class='num'>{r['pct']:.2f}%</td>"
            f"<td class='num'>{cumulative_pct:.1f}%</td>"
            f"<td><div class='bar' style='width:{min(r['pct'] * 3, 100):.1f}%;background:{bar_color}'></div></td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ONNX Runtime Profile: {model_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3-flame-graph@4/dist/d3-flamegraph.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/d3-flame-graph@4/dist/d3-flamegraph.css">
<style>
  :root {{
    --bg: #f8fafc; --surface: #ffffff; --surface2: #f1f5f9;
    --text: #1e293b; --text2: #64748b; --accent: #3b82f6;
    --accent2: #8b5cf6; --green: #10b981; --orange: #f59e0b;
    --red: #ef4444; --border: #e2e8f0;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
  .container {{ max-width:1400px; margin:0 auto; padding:24px; }}
  h1 {{ font-size:1.6rem; font-weight:700; margin-bottom:4px; color:#0f172a; }}
  .subtitle {{ color:var(--text2); font-size:0.85rem; margin-bottom:24px; }}

  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:28px; }}
  .card {{ background:var(--surface); border-radius:12px; padding:20px; border:1px solid var(--border); box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
  .card .label {{ color:var(--text2); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; }}
  .card .value {{ font-size:1.8rem; font-weight:700; color:var(--accent); margin-top:4px; }}
  .card .unit {{ font-size:0.8rem; color:var(--text2); }}

  .section {{ background:var(--surface); border-radius:12px; padding:24px; margin-bottom:24px; border:1px solid var(--border); box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
  .section h2 {{ font-size:1.1rem; font-weight:600; margin-bottom:16px; display:flex; align-items:center; gap:8px; color:#0f172a; }}
  .section h2 .badge {{ background:var(--surface2); color:var(--text2); font-size:0.7rem; padding:2px 8px; border-radius:99px; font-weight:500; }}



  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  th {{ text-align:left; padding:10px 12px; border-bottom:2px solid var(--border); color:var(--text2);
       font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em;
       position:sticky; top:0; background:var(--surface); z-index:1; }}
  td {{ padding:8px 12px; border-bottom:1px solid var(--surface2); }}
  tr:hover {{ background:#f8fafc; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:'Cascadia Code','Fira Code',monospace; }}
  .rank {{ text-align:center; color:var(--text2); width:40px; }}
  .node-name {{ max-width:380px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
                font-family:'Cascadia Code','Fira Code',monospace; font-size:0.78rem; }}
  .bar {{ height:16px; border-radius:4px; background:linear-gradient(90deg,var(--accent),var(--accent2)); min-width:2px; opacity:0.8; }}
  .op-bar {{ background:linear-gradient(90deg,var(--green),var(--accent)); }}
  .table-wrap {{ max-height:600px; overflow-y:auto; border-radius:8px; }}
  .input-table td, .input-table th {{ padding:8px 16px; }}
  .filter-bar {{ display:flex; gap:12px; margin-bottom:12px; align-items:center; flex-wrap:wrap; }}
  .filter-bar input {{ background:var(--surface2); border:1px solid var(--border); color:var(--text);
                       padding:6px 12px; border-radius:6px; font-size:0.82rem; width:300px; }}
  .filter-bar input::placeholder {{ color:var(--text2); }}
  .filter-bar select {{ background:var(--surface2); border:1px solid var(--border); color:var(--text);
                        padding:6px 12px; border-radius:6px; font-size:0.82rem; }}
  .footer {{ text-align:center; color:var(--text2); font-size:0.75rem; padding:20px 0; }}

  /* ── EP badges ── */
  .ep-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.7rem;
               font-weight:600; white-space:nowrap; }}
  .ep-cuda {{ background:#7c3aed; color:#fff; }}
  .ep-cpu  {{ background:#64748b; color:#fff; }}
  .ep-tensorrt {{ background:#76b900; color:#fff; }}
  .ep-dml  {{ background:#0078d4; color:#fff; }}
  .ep-badge:not(.ep-cuda):not(.ep-cpu):not(.ep-tensorrt):not(.ep-dml) {{ background:#94a3b8; color:#fff; }}

  .ep-summary {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:28px; }}
  .ep-item {{ display:flex; align-items:center; gap:8px; background:var(--surface); border-radius:10px;
              padding:10px 16px; border:1px solid var(--border); box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
  .ep-detail {{ font-size:0.82rem; color:var(--text2); }}

  /* ── Timeline (chrome://tracing style) ── */
  .tl-outer {{ position:relative; overflow:hidden; border:1px solid #cbd5e1; border-radius:8px; background:#fff; }}

  /* Overview / minimap */
  .tl-overview {{ height:48px; background:#f1f5f9; border-bottom:1px solid #cbd5e1; position:relative; cursor:pointer; }}
  .tl-overview canvas {{ display:block; width:100%; height:100%; }}
  .tl-ov-sel {{ position:absolute; top:0; height:100%; background:rgba(59,130,246,0.12);
                border-left:2px solid var(--accent); border-right:2px solid var(--accent);
                pointer-events:none; }}

  /* Time axis */
  .tl-axis {{ height:28px; background:#f8fafc; border-bottom:1px solid #e2e8f0; position:relative; overflow:hidden; }}

  /* Main timeline area */
  .tl-body-wrap {{ display:flex; }}
  .tl-labels {{ flex:0 0 180px; background:#1e293b; overflow-y:hidden; }}
  .tl-label-row {{ color:#e2e8f0; font-size:0.72rem; padding:0 10px; display:flex; align-items:center;
                   border-bottom:1px solid #334155; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .tl-label-row.header {{ background:#0f172a; font-weight:700; font-size:0.76rem; color:#94a3b8; }}
  .tl-canvas-wrap {{ flex:1; overflow:hidden; position:relative; cursor:grab; }}
  .tl-canvas-wrap:active {{ cursor:grabbing; }}
  #tlCanvas {{ display:block; }}

  #tlTooltip {{
    display:none; position:fixed; background:#fff; border:1px solid #cbd5e1;
    border-radius:8px; padding:10px 14px; font-size:0.78rem; box-shadow:0 4px 16px rgba(0,0,0,0.12);
    pointer-events:none; z-index:200; max-width:480px; line-height:1.6;
  }}
  #tlTooltip .tt-name {{ font-weight:600; color:#0f172a; word-break:break-all; }}
  #tlTooltip .tt-cat  {{ display:inline-block; padding:1px 6px; border-radius:4px; font-size:0.7rem; font-weight:600; color:#fff; }}
  #tlTooltip .tt-dur  {{ color:var(--text2); margin-top:2px; }}
</style>
</head>
<body>
<div class="container">
  <h1>ONNX Runtime Profile Report</h1>
  <div class="subtitle">{model_name} &mdash; {timestamp}</div>

  <div class="cards">
    <div class="card">
      <div class="label">Avg Inference ({agg['num_runs']} runs)</div>
      <div class="value">{avg_run_ms:,.2f} <span class="unit">ms</span></div>
    </div>
    <div class="card">
      <div class="label">Runs</div>
      <div class="value">{runs_value}</div>
      <div class="unit">{runs_unit}</div>
    </div>
    <div class="card">
      <div class="label">Nodes</div>
      <div class="value">{agg['num_nodes']}</div>
    </div>
    <div class="card">
      <div class="label">Op Types</div>
      <div class="value">{len(agg['op_type_summary'])}</div>
    </div>
  </div>

  {"<div class='ep-summary'>" + ep_summary_html + "</div>" if has_ep else ""}

  <div class="section">
    <h2>Input Tensors</h2>
    <table class="input-table">
      <tr><th>Name</th><th>Shape</th><th>DType</th></tr>
      {input_rows_html}
    </table>
  </div>

  <!-- ════ Timeline ════ -->
  <div class="section" style="padding:16px;">
    <h2>Execution Timeline</h2>
    <div class="tl-outer" id="tlOuter">
      <div class="tl-overview" id="tlOverview">
        <canvas id="tlOvCanvas"></canvas>
        <div class="tl-ov-sel" id="tlOvSel"></div>
      </div>
      <div class="tl-axis" id="tlAxis"></div>
      <div class="tl-body-wrap" id="tlBodyWrap">
        <div class="tl-labels" id="tlLabels"></div>
        <div class="tl-canvas-wrap" id="tlCanvasWrap">
          <canvas id="tlCanvas"></canvas>
        </div>
      </div>
    </div>
  </div>
  <div id="tlTooltip"></div>

  <!-- Run selector + Flamegraph + Per-Run pie -->
  <div class="section">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
      <h2 style="margin:0">Profiling Details</h2>
      <select id="runSelect" style="background:var(--surface2); border:1px solid var(--border); color:var(--text); padding:6px 12px; border-radius:6px; font-size:0.85rem;"></select>
    </div>
    <div style="display:grid; grid-template-columns:1fr 320px; gap:20px; align-items:start;">
      <div>
        <h3 style="font-size:0.9rem; color:var(--text2); margin-bottom:8px;">Flamegraph</h3>
        <div id="flamegraph"></div>
      </div>
      <div style="background:var(--surface2); border-radius:10px; padding:16px;">
        <h3 id="pieTitle" style="font-size:0.9rem; color:var(--text2); margin-bottom:8px;"></h3>
        <canvas id="perRunPie"></canvas>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Top 20 Nodes by Avg Duration</h2>
    <canvas id="barChart"></canvas>
  </div>

  <div class="section">
    <h2>All Nodes <span class="badge">{agg['num_nodes']} nodes</span></h2>
    <div class="filter-bar">
      <input type="text" id="nodeFilter" placeholder="Filter by node name or op type..." oninput="filterNodes()">
      <select id="opTypeSelect" onchange="filterNodes()">
        <option value="">All Op Types</option>
      </select>
      {"<select id='epSelect' onchange='filterNodes()'><option value=''>All EPs</option></select>" if has_ep else ""}
    </div>
    <div class="table-wrap">
      <table id="nodeTable">
        <thead>
          <tr><th>#</th><th>Node Name</th><th>Op Type</th>{"<th>EP</th>" if has_ep else ""}<th>Avg (ms)</th><th>Min</th><th>Max</th><th>Std</th><th>%</th><th>Cumulative</th><th style="width:15%"></th></tr>
        </thead>
        <tbody>
          {node_rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <div class="footer">Generated by profile_onnx.py &mdash; ONNX Runtime {e(ort.__version__)}</div>
</div>

<script>
/* ══════════════════════════════════════
   Chart.js
   ══════════════════════════════════════ */
const OP_COLOR_MAP = {op_color_map_json};
new Chart(document.getElementById('barChart'), {{
  type:'bar',
  data:{{ labels:{chart_labels}, datasets:[{{ data:{chart_values}, backgroundColor:{bar_colors}, borderRadius:4 }}] }},
  options:{{ indexAxis:'y', responsive:true,
    plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>c.parsed.x.toLocaleString()+' ms'}}}} }},
    scales:{{ x:{{grid:{{color:'#e2e8f0'}},ticks:{{color:'#64748b'}}}}, y:{{grid:{{display:false}},ticks:{{color:'#1e293b',font:{{size:10,family:"'Cascadia Code',monospace"}}}}}} }}
  }}
}});


/* ── Node table filter ── */
const HAS_EP = {'true' if has_ep else 'false'};
const opTypesSet = new Set();
const epSet = new Set();
document.querySelectorAll('#nodeTable tbody tr').forEach(tr=>{{
  const op=tr.children[2].textContent; if(op) opTypesSet.add(op);
  if (HAS_EP) {{ const ep=tr.dataset.ep; if(ep) epSet.add(ep); }}
}});
const selEl=document.getElementById('opTypeSelect');
[...opTypesSet].sort().forEach(op=>{{ const o=document.createElement('option'); o.value=op; o.textContent=op; selEl.appendChild(o); }});
if (HAS_EP) {{
  const epEl=document.getElementById('epSelect');
  [...epSet].sort().forEach(ep=>{{ const o=document.createElement('option'); o.value=ep; o.textContent=ep; epEl.appendChild(o); }});
}}
function filterNodes(){{
  const q=document.getElementById('nodeFilter').value.toLowerCase();
  const opF=document.getElementById('opTypeSelect').value;
  const epF=HAS_EP ? document.getElementById('epSelect').value : '';
  document.querySelectorAll('#nodeTable tbody tr').forEach(tr=>{{
    const n=tr.children[1].textContent.toLowerCase(), op=tr.children[2].textContent;
    const ep=tr.dataset.ep||'';
    tr.style.display=(!q||n.includes(q)||op.toLowerCase().includes(q))&&(!opF||op===opF)&&(!epF||ep===epF)?'':'none';
  }});
}}

/* ══════════════════════════════════════
   Timeline  (chrome://tracing style)
   ══════════════════════════════════════ */
(function(){{
  const RAW = {trace_json};
  if(!RAW.length) return;

  /* ── Colors (shared OP_COLOR_MAP from Python) ── */
  const CAT_COLORS = {{
    'model_loading_uri':'#ec4899','model_loading_from':'#ec4899',
    'session_initialization':'#0891b2',
    'model_run':'#b45309',
  }};
  function colorFor(ev) {{
    if (ev.cat === 'Session') return CAT_COLORS[ev.name] || '#64748b';
    return OP_COLOR_MAP[ev.op] || '#94a3b8';
  }}

  /* ── Globals ── */
  const globalMin = Math.min(...RAW.map(e=>e.ts));
  const globalMax = Math.max(...RAW.map(e=>e.ts+e.dur));

  /* ══ Row assignment ══
     Row 0: Process header (dark)
     Row 1: Thread header (dark)
     Row 2: model_loading* + session_initialization + model_run  (Session, same row)
     Row 3+: Node events — EP が複数ある場合は EP ごとにヘッダ行 + レーン群
  */
  const HEADER_ROWS = 2;           // Process + Thread
  const SESSION_TOP_ROW = 2;       // loading / init / model_run

  const sessionEvents = RAW.filter(e => e.cat === 'Session');
  const nodeEvents    = RAW.filter(e => e.cat === 'Node');

  // Assign Session events to fixed row
  const placed = [];  // {{ row, ev }}
  for (const ev of sessionEvents) {{
    placed.push({{ row: SESSION_TOP_ROW, ev }});
  }}

  // EP 別にノードをグルーピング
  const epGroups = {{}};
  for (const ev of nodeEvents) {{
    const ep = ev.provider ? ev.provider.replace('ExecutionProvider','') : '';
    if (!epGroups[ep]) epGroups[ep] = [];
    epGroups[ep].push(ev);
  }}
  const epNames = Object.keys(epGroups).sort((a,b) => {{
    // CUDA / GPU 系を先、CPU を後、空を最後
    if (!a) return 1; if (!b) return -1;
    if (a === 'CPU') return 1; if (b === 'CPU') return -1;
    return a.localeCompare(b);
  }});
  const hasMultiEP = epNames.length > 1 || (epNames.length === 1 && epNames[0] !== '');

  // EP ごとにレーンパッキング (ヘッダ行なし、ラベルで EP 名を表示)
  let nextRow = 3;  // Session row の次から
  const rowEpLabel = {{}};  // row -> EP name (各 EP グループの先頭レーンのみ)

  if (hasMultiEP) {{
    for (const ep of epNames) {{
      const groupStartRow = nextRow;
      const evs = epGroups[ep];
      evs.sort((a,b) => a.ts - b.ts || b.dur - a.dur);
      const laneEnds = [];
      for (const ev of evs) {{
        let lane = -1;
        for (let l = 0; l < laneEnds.length; l++) {{
          if (laneEnds[l] <= ev.ts) {{ lane = l; break; }}
        }}
        if (lane === -1) {{ lane = laneEnds.length; laneEnds.push(0); }}
        laneEnds[lane] = ev.ts + ev.dur;
        placed.push({{ row: nextRow + lane, ev }});
      }}
      rowEpLabel[groupStartRow] = ep || 'Other';
      nextRow += laneEnds.length;
    }}
  }} else {{
    // EP 情報なし or 単一 EP: 従来通りフラットにパック
    nodeEvents.sort((a,b) => a.ts - b.ts || b.dur - a.dur);
    const laneEnds = [];
    for (const ev of nodeEvents) {{
      let lane = -1;
      for (let l = 0; l < laneEnds.length; l++) {{
        if (laneEnds[l] <= ev.ts) {{ lane = l; break; }}
      }}
      if (lane === -1) {{ lane = laneEnds.length; laneEnds.push(0); }}
      laneEnds[lane] = ev.ts + ev.dur;
      placed.push({{ row: nextRow + lane, ev }});
    }}
    nextRow += laneEnds.length;
  }}

  const totalRows = nextRow;

  /* ── Sizing ── */
  const ROW_H = 24;
  const dpr = window.devicePixelRatio || 1;

  /* ── Row labels ── */
  const labelsEl = document.getElementById('tlLabels');
  labelsEl.innerHTML = '';
  for (let r = 0; r < totalRows; r++) {{
    const div = document.createElement('div');
    div.className = 'tl-label-row' + (r < HEADER_ROWS ? ' header' : '');
    div.style.height = ROW_H + 'px';
    if (r === 0) div.textContent = 'Process ' + (RAW[0].pid || '');
    else if (r === 1) div.textContent = 'Thread ' + (RAW[0].tid || '');
    else if (rowEpLabel[r]) div.textContent = rowEpLabel[r];
    else div.textContent = '';
    labelsEl.appendChild(div);
  }}

  const bodyH = totalRows * ROW_H;
  const bodyWrap = document.getElementById('tlBodyWrap');
  bodyWrap.style.height = Math.min(bodyH, 480) + 'px';
  bodyWrap.style.overflowY = bodyH > 480 ? 'auto' : 'hidden';
  labelsEl.style.minHeight = bodyH + 'px';

  const canvasWrap = document.getElementById('tlCanvasWrap');
  canvasWrap.style.minHeight = bodyH + 'px';
  bodyWrap.addEventListener('scroll', () => {{
    labelsEl.style.transform = `translateY(${{-bodyWrap.scrollTop}}px)`;
  }});

  /* ── View state ── */
  let viewStart = globalMin;
  let viewEnd = globalMax;

  /* ── Main canvas ── */
  const canvas = document.getElementById('tlCanvas');
  const ctx = canvas.getContext('2d');

  function resizeCanvas() {{
    const w = canvasWrap.clientWidth;
    canvas.width = w * dpr;
    canvas.height = bodyH * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = bodyH + 'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }}

  /* nice tick step */
  function niceStep(span) {{
    const raw = span / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const res = raw / mag;
    if (res<=1.5) return mag;
    if (res<=3.5) return 2*mag;
    if (res<=7.5) return 5*mag;
    return 10*mag;
  }}

  function fmtTime(us, span) {{
    if (span > 2000000) return (us/1e6).toFixed(2)+'s';
    if (span > 200000) return (us/1e3).toFixed(0)+'ms';
    if (span > 2000) return (us/1e3).toFixed(1)+'ms';
    return us.toFixed(0)+'us';
  }}

  /* ── Axis ── */
  const axisEl = document.getElementById('tlAxis');
  function drawAxis() {{
    const w = canvasWrap.clientWidth;
    const span = viewEnd - viewStart;
    const step = niceStep(span);
    const start = Math.ceil(viewStart / step) * step;
    let html = '';
    for (let t = start; t <= viewEnd; t += step) {{
      const x = 180 + (t - viewStart) / span * w;
      html += `<div style="position:absolute;left:${{x}}px;top:0;height:100%;border-left:1px solid #cbd5e1;font-size:11px;color:#64748b;padding:6px 0 0 4px;white-space:nowrap;">${{fmtTime(t - globalMin, span)}}</div>`;
    }}
    axisEl.innerHTML = html;
  }}

  /* ── Draw ── */
  function draw() {{
    resizeCanvas();
    const W = canvas.width / dpr;
    const span = viewEnd - viewStart;
    ctx.clearRect(0,0,W,bodyH);

    // BG stripes
    for (let i = 0; i < totalRows; i++) {{
      ctx.fillStyle = i % 2 === 0 ? '#ffffff' : '#f8fafc';
      ctx.fillRect(0, i*ROW_H, W, ROW_H);
    }}
    // Header BG
    ctx.fillStyle = '#1e293b'; ctx.fillRect(0, 0, W, ROW_H);
    ctx.fillStyle = '#334155'; ctx.fillRect(0, ROW_H, W, ROW_H);

    // Grid lines
    const step = niceStep(span);
    const start = Math.ceil(viewStart / step) * step;
    ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 1;
    for (let t = start; t <= viewEnd; t += step) {{
      const x = (t - viewStart) / span * W;
      ctx.beginPath(); ctx.moveTo(x, HEADER_ROWS*ROW_H); ctx.lineTo(x, bodyH); ctx.stroke();
    }}

    // Events
    for (const {{row, ev}} of placed) {{
      const x1 = (ev.ts - viewStart) / span * W;
      const x2 = (ev.ts + ev.dur - viewStart) / span * W;
      if (x2 < 0 || x1 > W) continue;
      const cx1 = Math.max(x1, 0);
      const cx2 = Math.min(x2, W);
      const bw = Math.max(cx2 - cx1, 0.5);
      const y = row * ROW_H;

      ctx.fillStyle = colorFor(ev);
      if (bw < 2) {{
        ctx.fillRect(cx1, y+2, bw, ROW_H-4);
      }} else {{
        ctx.beginPath();
        ctx.roundRect(cx1, y+2, bw, ROW_H-4, Math.min(3, bw/2));
        ctx.fill();
      }}

      if (bw > 30) {{
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${{Math.min(11, ROW_H-8)}}px system-ui, sans-serif`;
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.save();
        ctx.beginPath(); ctx.rect(cx1+3, y+2, bw-6, ROW_H-4); ctx.clip();
        let lbl = ev.label || (ev.cat === 'Session' ? ev.name : (ev.op || ev.name.split('/').pop()));
        ctx.fillText(lbl, cx1+4, y + ROW_H/2);
        ctx.restore();
      }}
    }}

    drawAxis();
    drawOverviewSel();
  }}

  /* ── Overview / Minimap ── */
  const ovCanvas = document.getElementById('tlOvCanvas');
  const ovCtx = ovCanvas.getContext('2d');
  const ovEl = document.getElementById('tlOverview');
  function drawOverview() {{
    const w = ovEl.clientWidth;
    const h = ovEl.clientHeight;
    ovCanvas.width = w * dpr; ovCanvas.height = h * dpr;
    ovCanvas.style.width = w + 'px'; ovCanvas.style.height = h + 'px';
    ovCtx.setTransform(dpr,0,0,dpr,0,0);
    ovCtx.clearRect(0,0,w,h);
    const span = globalMax - globalMin;
    const buckets = new Float32Array(w);
    for (const {{ev}} of placed) {{
      if (ev.cat !== 'Node') continue;
      const px1 = Math.floor((ev.ts - globalMin) / span * w);
      const px2 = Math.ceil((ev.ts + ev.dur - globalMin) / span * w);
      for (let px = Math.max(0,px1); px < Math.min(w,px2); px++)
        buckets[px] += ev.dur / Math.max(1, px2-px1);
    }}
    const maxB = Math.max(...buckets) || 1;
    for (let px = 0; px < w; px++) {{
      if (buckets[px] <= 0) continue;
      const barH = (buckets[px] / maxB) * (h - 4);
      ovCtx.fillStyle = '#22c55e';
      ovCtx.globalAlpha = 0.4 + 0.6 * (buckets[px] / maxB);
      ovCtx.fillRect(px, h - barH - 2, 1, barH);
    }}
    ovCtx.globalAlpha = 1;
  }}

  const ovSel = document.getElementById('tlOvSel');
  function drawOverviewSel() {{
    const w = ovEl.clientWidth;
    const span = globalMax - globalMin;
    const l = (viewStart - globalMin) / span * w;
    const r = (viewEnd - globalMin) / span * w;
    ovSel.style.left = l + 'px';
    ovSel.style.width = Math.max(r - l, 4) + 'px';
  }}

  ovEl.addEventListener('click', (e) => {{
    const rect = ovEl.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    const center = globalMin + frac * (globalMax - globalMin);
    const halfSpan = (viewEnd - viewStart) / 2;
    viewStart = center - halfSpan; viewEnd = center + halfSpan;
    draw();
  }});

  /* ── Tooltip ── */
  const tooltip = document.getElementById('tlTooltip');
  function findEventAt(mx, my) {{
    const W = canvas.width / dpr;
    const span = viewEnd - viewStart;
    // Search in reverse so top-drawn (later in array = Node) wins over Session
    for (let i = placed.length - 1; i >= 0; i--) {{
      const {{row, ev}} = placed[i];
      const x1 = (ev.ts - viewStart) / span * W;
      const x2 = (ev.ts + ev.dur - viewStart) / span * W;
      const y = row * ROW_H;
      if (mx >= Math.max(x1,0) && mx <= Math.min(x2,W) && my >= y+2 && my <= y+ROW_H-2) return ev;
    }}
    return null;
  }}
  canvas.addEventListener('mousemove', (e) => {{
    if (dragging) {{ tooltip.style.display='none'; return; }}
    const rect = canvas.getBoundingClientRect();
    const ev = findEventAt(e.clientX - rect.left, e.clientY - rect.top);
    if (ev) {{
      const c = colorFor(ev);
      const durStr = ev.dur >= 1000 ? (ev.dur/1000).toFixed(2)+' ms' : ev.dur+' us';
      const ttName = ev.label || ev.name;
      const epShort = ev.provider ? ev.provider.replace('ExecutionProvider','') : '';
      const epBadge = epShort ? ` <span class="ep-badge ep-${{epShort.toLowerCase()}}">${{epShort}}</span>` : '';
      tooltip.innerHTML = `<div class="tt-name">${{ttName}}</div>`
        + `<span class="tt-cat" style="background:${{c}}">${{ev.cat}}</span>`
        + (ev.op ? ` <span style="color:${{c}};font-weight:600">${{ev.op}}</span>` : '')
        + epBadge
        + `<div class="tt-dur">${{durStr}}</div>`;
      tooltip.style.display = 'block';
      let tx = e.clientX + 14, ty = e.clientY + 14;
      if (tx + 400 > window.innerWidth) tx = e.clientX - 420;
      if (ty + 80 > window.innerHeight) ty = e.clientY - 90;
      tooltip.style.left = tx + 'px'; tooltip.style.top = ty + 'px';
      canvas.style.cursor = 'pointer';
    }} else {{
      tooltip.style.display = 'none';
      canvas.style.cursor = 'grab';
    }}
  }});
  canvas.addEventListener('mouseleave', () => {{ tooltip.style.display='none'; }});

  /* ── Pan ── */
  let dragging = false, dragX = 0, dragVS = 0;
  canvas.addEventListener('mousedown', (e) => {{
    dragging = true; dragX = e.clientX; dragVS = viewStart;
    canvas.style.cursor = 'grabbing';
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    const W = canvas.width / dpr;
    const span = viewEnd - viewStart;
    viewStart = dragVS - (e.clientX - dragX) / W * span;
    viewEnd = viewStart + span;
    draw();
  }});
  window.addEventListener('mouseup', () => {{ dragging = false; canvas.style.cursor = 'grab'; }});

  /* ── Zoom (wheel) ── */
  canvasWrap.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / (canvas.width / dpr);
    const pivot = viewStart + frac * (viewEnd - viewStart);
    const factor = e.deltaY > 0 ? 1.3 : 1/1.3;
    const newSpan = (viewEnd - viewStart) * factor;
    viewStart = pivot - frac * newSpan;
    viewEnd = viewStart + newSpan;
    draw();
  }}, {{ passive: false }});

  /* ── Init ── */
  drawOverview();
  draw();
  window.addEventListener('resize', () => {{ drawOverview(); draw(); }});
}})();

/* ══════════════════════════════════════
   Run selector + Per-Run pie (single)
   ══════════════════════════════════════ */
(function(){{
  const perRun = {per_run_op_json};
  if (!perRun.length) return;

  const sel = document.getElementById('runSelect');
  perRun.forEach((run, i) => {{
    const o = document.createElement('option');
    o.value = i;
    const ms = (run.run_dur_us / 1000).toFixed(2);
    o.textContent = 'Run ' + run.run_index + ' (' + ms + ' ms)';
    sel.appendChild(o);
  }});

  let pieChart = null;
  function drawPie(idx) {{
    const run = perRun[idx];
    const ms = (run.run_dur_us / 1000).toFixed(2);
    document.getElementById('pieTitle').textContent = 'Run ' + run.run_index + ' (' + ms + ' ms)';

    const top = run.ops.slice(0, 12);
    const rest = run.ops.slice(12);
    const labels = top.map(o => o.op_type);
    const values = top.map(o => o.dur_us);
    const colors = top.map(o => OP_COLOR_MAP[o.op_type] || '#94a3b8');
    if (rest.length) {{
      labels.push('Others');
      values.push(rest.reduce((s, o) => s + o.dur_us, 0));
      colors.push('#cbd5e1');
    }}

    if (pieChart) pieChart.destroy();
    pieChart = new Chart(document.getElementById('perRunPie'), {{
      type: 'doughnut',
      data: {{ labels, datasets: [{{ data: values, backgroundColor: colors, borderWidth: 2, borderColor: '#fff', hoverOffset: 6 }}] }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ color: '#1e293b', font: {{ size: 10 }}, padding: 5, boxWidth: 12 }} }},
          tooltip: {{ callbacks: {{ label: ctx => {{
            const v = ctx.parsed;
            const total = ctx.dataset.data.reduce((a,b)=>a+b,0);
            const pct = (v / total * 100).toFixed(1);
            return ctx.label + ': ' + (v/1000).toFixed(2) + ' ms (' + pct + '%)';
          }} }} }}
        }}
      }}
    }});
  }}

  sel.addEventListener('change', () => drawPie(parseInt(sel.value)));
  drawPie(0);
}})();

/* ══════════════════════════════════════
   Flamegraph (d3-flame-graph)
   ══════════════════════════════════════ */
(function(){{
  const data = {flame_json};
  if (!data) return;
  const container = document.getElementById('flamegraph');
  const w = container.clientWidth || 1200;
  const tip = d3.select('body').append('div')
    .style('position','absolute').style('background','#fff').style('border','1px solid #cbd5e1')
    .style('border-radius','6px').style('padding','8px 12px').style('font-size','12px')
    .style('box-shadow','0 2px 8px rgba(0,0,0,0.1)').style('pointer-events','none')
    .style('display','none').style('z-index','300');
  const chart = flamegraph().width(w).cellHeight(22)
    .selfValue(false)
    .setColorMapper(function(d) {{
      if (!d.parent) return '#e2e8f0';
      const root = d3.select('#flamegraph').datum();
      const pct = d.data.value / (root.value || 1);
      return d3.interpolateYlOrRd(Math.min(pct * 8, 1));
    }});
  d3.select('#flamegraph').datum(data).call(chart);
  // Manual tooltip
  d3.select('#flamegraph').selectAll('.d3-flame-graph-label').each(function() {{
    const el = d3.select(this.parentNode);
    el.on('mouseover', function(ev) {{
      const d = d3.select(this).datum();
      if (!d || !d.data) return;
      const ms = (d.data.value / 1000).toFixed(2);
      const root = d3.select('#flamegraph').datum();
      const pct = (d.data.value / (root.value || 1) * 100).toFixed(1);
      tip.html('<b>' + d.data.name + '</b><br>' + ms + ' ms (' + pct + '%)')
        .style('display','block').style('left',(ev.pageX+12)+'px').style('top',(ev.pageY+12)+'px');
    }}).on('mousemove', function(ev) {{
      tip.style('left',(ev.pageX+12)+'px').style('top',(ev.pageY+12)+'px');
    }}).on('mouseout', function() {{ tip.style('display','none'); }});
  }});
}})();


</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
