"""ONNX モデルの入力を自動推定し、ランダム入力で推論を実行してプロファイル JSON と HTML レポートを保存する."""

import argparse
import os
import sys

import onnxruntime as ort

from onnx_profiler.constants import NUM_WARMUP, NUM_RUNS
from onnx_profiler.inputs import parse_input_shapes, parse_input_values, generate_random_inputs
from onnx_profiler.profiler import aggregate_profile, build_flamegraph_tree
from onnx_profiler.report import generate_html_report


def print_summary(agg):
    print(f"\n  上位10ノード:")
    print(f"  {'#':<4} {'ノード名':<46} {'Op':<14} {'Avg(ms)':>10} {'%':>7}")
    print(f"  {'-'*85}")
    for i, r in enumerate(agg["nodes"][:10]):
        name = r["name"]
        if len(name) > 44:
            name = name[:41] + "..."
        print(f"  {i+1:<4} {name:<46} {r['op_type']:<14} {r['avg_us']/1000:>10,.2f} {r['pct']:>6.1f}%")

    print(f"\n  Op Type 別:")
    print(f"  {'Op':<20} {'Avg(ms)':>12} {'%':>7}")
    print(f"  {'-'*42}")
    for o in agg["op_type_summary"][:10]:
        print(f"  {o['op_type']:<20} {o['avg_us']/1000:>12,.2f} {o['pct']:>6.1f}%")


def generate_report(profile_file, output_path, num_runs, num_warmup, model_path=None, input_info=None):
    agg = aggregate_profile(profile_file, num_runs, num_warmup)

    print(f"\n{'='*76}")
    if num_warmup > 0:
        print(f" 平均推論時間 ({num_runs}回): {agg['avg_run_us']/1000:.2f} ms  |  ノード数: {agg['num_nodes']}")
    else:
        n = len(agg["per_run_op"])
        print(f" 平均推論時間 ({n}回): {agg['avg_run_us']/1000:.2f} ms  |  ノード数: {agg['num_nodes']}")
    print(f"{'='*76}")
    print_summary(agg)

    flame_data = build_flamegraph_tree(agg)
    generate_html_report(agg, model_path or profile_file, input_info or [], output_path, num_warmup,
                         flame_data=flame_data)
    print(f"\nHTML レポート: {output_path}")


def cmd_run(args):
    """モデルを実行してプロファイル + HTML 生成."""
    if not os.path.exists(args.model):
        print(f"エラー: モデルファイルが見つかりません: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    shape_overrides = parse_input_shapes(args.input_shape)
    value_overrides = parse_input_values(args.input_value)

    print(f"モデル読み込み中: {args.model}")
    inputs, input_info = generate_random_inputs(args.model, args.dynamic_size, shape_overrides, value_overrides)

    print(f"\n入力テンソル情報:")
    for info in input_info:
        print(f"  {info['name']}: shape={info['shape']}, dtype={info['dtype']}")

    print(f"\nプロファイリング実行中 (ウォームアップ {args.num_warmup} 回 + 計測 {args.num_runs} 回)...")
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = os.path.join(os.path.abspath(args.output_dir), "ort_profile")

    session = ort.InferenceSession(args.model, sess_options=opts)
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(args.num_warmup):
        session.run(output_names, inputs)

    for i in range(args.num_runs):
        session.run(output_names, inputs)
        print(f"  run {i+1}/{args.num_runs} done")

    profile_file = session.end_profiling()
    print(f"\nプロファイル JSON: {profile_file}")

    profile_basename = os.path.splitext(os.path.basename(profile_file))[0]
    html_path = os.path.join(os.path.abspath(args.output_dir), profile_basename + ".html")
    generate_report(profile_file, html_path, args.num_runs, args.num_warmup,
                    model_path=args.model, input_info=input_info)


def cmd_report(args):
    """既存のプロファイル JSON から HTML レポートを生成."""
    if not os.path.exists(args.json):
        print(f"エラー: JSON ファイルが見つかりません: {args.json}", file=sys.stderr)
        sys.exit(1)

    print(f"プロファイル JSON 読み込み中: {args.json}")
    profile_basename = os.path.splitext(os.path.basename(args.json))[0]

    if args.output:
        html_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(html_path)) or ".", exist_ok=True)
    else:
        html_path = os.path.join(os.path.dirname(os.path.abspath(args.json)), profile_basename + ".html")

    # ウォームアップ不明なので全 model_run を計測扱い
    generate_report(args.json, html_path, num_runs=0, num_warmup=0)


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime Profiler")
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # run: モデル実行 + プロファイル
    p_run = subparsers.add_parser("run", help="モデルを実行してプロファイル + HTML 生成")
    p_run.add_argument("model", help="ONNX モデルファイルのパス")
    p_run.add_argument("-d", "--dynamic-size", type=int, default=1,
                       help="動的次元に使用するサイズ (デフォルト: 1)")
    p_run.add_argument("-s", "--input-shape", action="append", metavar="NAME:D0,D1,...",
                       help="入力のシェイプを明示指定 (例: -s images:1,3,480,640)")
    p_run.add_argument("-v", "--input-value", action="append", metavar="NAME=V0,V1,...",
                       help="入力の固定値を指定 (例: -v sr=16000)")
    p_run.add_argument("-o", "--output-dir", default=".",
                       help="出力ディレクトリ (デフォルト: カレントディレクトリ)")
    p_run.add_argument("-n", "--num-runs", type=int, default=NUM_RUNS,
                       help=f"計測実行回数 (デフォルト: {NUM_RUNS})")
    p_run.add_argument("-w", "--num-warmup", type=int, default=NUM_WARMUP,
                       help=f"ウォームアップ回数 (デフォルト: {NUM_WARMUP})")

    # report: 既存 JSON から HTML 生成
    p_report = subparsers.add_parser("report", help="既存のプロファイル JSON から HTML レポートを生成")
    p_report.add_argument("json", help="プロファイル JSON ファイルのパス")
    p_report.add_argument("-o", "--output", help="出力 HTML パス (デフォルト: JSON と同名)")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
