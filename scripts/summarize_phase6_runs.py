#!/usr/bin/env python3
"""
Summarize a set of Phase 6 evaluation runs into a single comparison table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 6 runs by run-id or prefix.")
    parser.add_argument("--run-id", nargs="*", default=None, help="Explicit evaluation run ids")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix to match evaluation run directories")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save the summary table")
    return parser.parse_args()


def discover_latest_run() -> list[Path]:
    candidates = sorted(
        [p for p in EVAL_ROOT.iterdir() if p.is_dir() and (p / "phase6_results.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Phase 6 runs with phase6_results.json found under {EVAL_ROOT}")
    return [candidates[0]]


def discover_runs(args: argparse.Namespace) -> list[Path]:
    if args.run_id:
        return [EVAL_ROOT / run_id for run_id in args.run_id]
    if args.prefix:
        return sorted([p for p in EVAL_ROOT.iterdir() if p.is_dir() and p.name.startswith(args.prefix)])
    return discover_latest_run()


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    df = pd.DataFrame(fold_results)
    if df.empty:
        return {}
    summary = {
        "annualized_return": df["annualized_return"].mean() if "annualized_return" in df else None,
        "annualized_volatility": df["annualized_volatility"].mean() if "annualized_volatility" in df else None,
        "sharpe_ratio": df["sharpe_ratio"].mean() if "sharpe_ratio" in df else None,
        "max_drawdown": df["max_drawdown"].mean() if "max_drawdown" in df else None,
        "information_ratio": df["information_ratio"].mean() if "information_ratio" in df else None,
    }
    return summary


def build_strategy_table(payload: dict) -> pd.DataFrame:
    comparison = payload.get("comparison", {})
    strategies = comparison.get("strategies")
    if isinstance(strategies, list) and strategies:
        rows = []
        for strategy in strategies:
            rows.append(
                {
                    "name": strategy.get("name"),
                    "type": strategy.get("type"),
                    "annualized_return": strategy.get("annualized_return"),
                    "annualized_volatility": strategy.get("annualized_volatility", strategy.get("volatility")),
                    "sharpe_ratio": strategy.get("sharpe_ratio"),
                    "max_drawdown": strategy.get("max_drawdown"),
                    "information_ratio": strategy.get("information_ratio"),
                }
            )
        return pd.DataFrame(rows)

    rows = []
    agents = payload.get("agents", {})
    for agent_name, agent_payload in agents.items():
        summary = aggregate_fold_results(agent_payload.get("fold_results", []))
        rows.append({"name": agent_name.upper(), "type": "rl_agent", **summary})
    return pd.DataFrame(rows)


def resolve_agent_summary(payload: dict, agent_name: str) -> dict:
    comparison = payload.get("comparison", {})

    strategies = comparison.get("strategies")
    if isinstance(strategies, list):
        for strategy in strategies:
            strategy_name = str(strategy.get("name", "")).strip().lower()
            strategy_type = str(strategy.get("type", "")).strip().lower()
            if strategy_type == "rl_agent" or strategy_name == agent_name.lower():
                return {
                    "annualized_return": strategy.get("annualized_return"),
                    "annualized_volatility": strategy.get("annualized_volatility", strategy.get("volatility")),
                    "sharpe_ratio": strategy.get("sharpe_ratio"),
                    "max_drawdown": strategy.get("max_drawdown"),
                    "information_ratio": strategy.get("information_ratio"),
                }

    named_summary = comparison.get(agent_name)
    if isinstance(named_summary, dict):
        return {
            "annualized_return": named_summary.get("annualized_return"),
            "annualized_volatility": named_summary.get("annualized_volatility", named_summary.get("volatility")),
            "sharpe_ratio": named_summary.get("sharpe_ratio"),
            "max_drawdown": named_summary.get("max_drawdown"),
            "information_ratio": named_summary.get("information_ratio"),
        }

    agents = payload.get("agents", {})
    agent_payload = agents.get(agent_name, {})
    return aggregate_fold_results(agent_payload.get("fold_results", []))


def annualized_to_period_return(annualized_return: float | None, test_start: str, test_end: str) -> float | None:
    if annualized_return is None or pd.isna(annualized_return):
        return None
    start = pd.to_datetime(test_start)
    end = pd.to_datetime(test_end)
    n_days = max((end - start).days, 1)
    return float((1.0 + float(annualized_return)) ** (n_days / 252.0) - 1.0)


def save_equity_plot(run_dir: Path, payload: dict, agent_name: str) -> Path | None:
    agents = payload.get("agents", {})
    agent_payload = agents.get(agent_name, {})
    fold_results = agent_payload.get("fold_results", [])
    if not fold_results:
        return None

    eq_rows: list[dict] = []
    agent_equity = 1.0
    benchmark_equity = 1.0

    for fold in fold_results:
        agent_period_return = annualized_to_period_return(
            fold.get("annualized_return"),
            fold.get("test_start"),
            fold.get("test_end"),
        )
        benchmark_ann_return = None
        if fold.get("annualized_return") is not None and fold.get("excess_return") is not None:
            benchmark_ann_return = float(fold.get("annualized_return")) - float(fold.get("excess_return"))
        benchmark_period_return = annualized_to_period_return(
            benchmark_ann_return,
            fold.get("test_start"),
            fold.get("test_end"),
        )

        if agent_period_return is None or benchmark_period_return is None:
            continue

        agent_equity *= 1.0 + agent_period_return
        benchmark_equity *= 1.0 + benchmark_period_return
        fold_end = pd.to_datetime(fold.get("test_end"))

        eq_rows.append({"date": fold_end, "strategy": agent_name.upper(), "equity": agent_equity})
        eq_rows.append({"date": fold_end, "strategy": "Benchmark", "equity": benchmark_equity})

    if not eq_rows:
        return None

    eq_df = pd.DataFrame(eq_rows).sort_values(["date", "strategy"])
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=eq_df, x="date", y="equity", hue="strategy", linewidth=2.5, ax=ax)
    ax.set_title(f"Phase 6 Walk-Forward Equity by Fold: {agent_name.upper()} vs Benchmark")
    ax.set_xlabel("Fold test end")
    ax.set_ylabel("Equity (compounded from fold summaries)")
    ax.legend(title="")
    fig.tight_layout()

    plot_path = run_dir / "phase6_equity_vs_benchmark.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    rows = []
    run_dirs = discover_runs(args)
    if not args.run_id and not args.prefix:
        print(f"[info] No --run-id/--prefix provided. Using latest Phase 6 run: {run_dirs[0].name}")

    for run_dir in run_dirs:
        results_path = run_dir / "phase6_results.json"
        if not results_path.exists():
            continue
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        agents = payload.get("agents", {})
        agent_name = next(iter(agents.keys()), "unknown")
        agent_summary = resolve_agent_summary(payload, agent_name)
        strategy_table = build_strategy_table(payload)
        benchmark_row = strategy_table[strategy_table["type"] == "benchmark"].head(1)
        baseline_rows = strategy_table[strategy_table["type"] == "baseline"].copy()
        best_baseline = (
            baseline_rows.sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False]).head(1)
            if not baseline_rows.empty
            else pd.DataFrame()
        )
        plot_path = save_equity_plot(run_dir, payload, agent_name)

        benchmark_return = benchmark_row["annualized_return"].iloc[0] if not benchmark_row.empty else None
        benchmark_sharpe = benchmark_row["sharpe_ratio"].iloc[0] if not benchmark_row.empty else None
        best_baseline_name = best_baseline["name"].iloc[0] if not best_baseline.empty else None
        best_baseline_return = best_baseline["annualized_return"].iloc[0] if not best_baseline.empty else None
        best_baseline_sharpe = best_baseline["sharpe_ratio"].iloc[0] if not best_baseline.empty else None

        rows.append(
            {
                "run_id": run_dir.name,
                "agent": agent_name,
                "annualized_return": agent_summary.get("annualized_return"),
                "annualized_volatility": agent_summary.get("annualized_volatility"),
                "sharpe_ratio": agent_summary.get("sharpe_ratio"),
                "max_drawdown": agent_summary.get("max_drawdown"),
                "information_ratio": agent_summary.get("information_ratio"),
                "benchmark_return": benchmark_return,
                "benchmark_sharpe": benchmark_sharpe,
                "best_baseline": best_baseline_name,
                "best_baseline_return": best_baseline_return,
                "best_baseline_sharpe": best_baseline_sharpe,
                "return_vs_benchmark": (
                    agent_summary.get("annualized_return") - benchmark_return
                    if agent_summary.get("annualized_return") is not None and benchmark_return is not None
                    else None
                ),
                "sharpe_vs_benchmark": (
                    agent_summary.get("sharpe_ratio") - benchmark_sharpe
                    if agent_summary.get("sharpe_ratio") is not None and benchmark_sharpe is not None
                    else None
                ),
                "return_vs_best_baseline": (
                    agent_summary.get("annualized_return") - best_baseline_return
                    if agent_summary.get("annualized_return") is not None and best_baseline_return is not None
                    else None
                ),
                "sharpe_vs_best_baseline": (
                    agent_summary.get("sharpe_ratio") - best_baseline_sharpe
                    if agent_summary.get("sharpe_ratio") is not None and best_baseline_sharpe is not None
                    else None
                ),
                "equity_plot": str(plot_path) if plot_path else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No valid phase6_results.json files were found for the requested runs.")
    df = df.sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False])
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
