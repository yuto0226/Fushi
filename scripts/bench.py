"""
benchmark_algorithms.py
=======================
測量不同搜尋演算法的四項關鍵指標：

  1. EBF (node)      — Δnodes[d] / Δnodes[d-1]，越低代表剪枝效果越好
  2. EBF (wall-clock)— Δtime[d]  / Δtime[d-1]，最直觀的實際時間成長率
  3. NPS             — nodes per second，吞吐量指標
  4. TT Hit Rate     — 轉置表命中率（僅適用於有 TT 的演算法）

輸出圖表（scripts/plots/）
--------------------------
  fig_perf_overview.png   2×2 總覽：EBF / NPS / Wall-clock EBF / TT Hit Rate（跨局面平均 ± 1σ）
  fig_node_reduction.png  各深度節點數 & 時間縮減比率（AlphaBeta / AlphaBeta+TT）
  fig_depth_heatmap.png   Depth × Position 搜尋時間熱點圖（兩演算法並排）

執行方式
--------
  cd /home/yuto/learn/chess-engine
  uv run python scripts/benchmark_algorithms.py

選用參數（環境變數）
--------------------
  MAX_DEPTH   搜尋最大深度 (預設 7)
  OUT_DIR     圖表輸出目錄 (預設 scripts/plots/)
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import chess

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from Fushi.evaluate import ShannonEvaluator  # noqa: E402
from Fushi.search import AlphaBetaSearcher, TranspositionTable  # noqa: E402
from Fushi.search import SearchInfo  # noqa: E402

matplotlib.use("Agg")


MAX_DEPTH: int = int(os.environ.get("MAX_DEPTH", "25"))
OUT_DIR: Path = Path(os.environ.get("OUT_DIR", str(_REPO_ROOT / "scripts" / "plots")))


TEST_POSITIONS: dict[str, str] = {
    "Opening (Ruy Lopez)": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "Opening (Sicilian)": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "Middlegame (sharp)": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "Middlegame (open)": "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2NB4/PPP2PPP/R1BQK2R w KQ - 0 8",
    "Endgame (K+R vs K)": "8/8/8/8/8/3k4/8/3KR3 w - - 0 1",
    "Endgame (pawns)": "8/p7/1p6/2p5/3P4/4P3/5PP1/8 w - - 0 1",
    "Initial Position": chess.STARTING_FEN,
}


@dataclass
class DepthRecord:
    """Measurements for a single depth iteration."""

    depth: int
    # Node counts
    cumulative_nodes: int  # total nodes from start of search
    delta_nodes: int  # new nodes added this depth iteration
    # Time
    cumulative_time_ms: float  # elapsed time from start of search
    delta_time_ms: float  # time spent in this depth iteration
    # TT statistics (None when algorithm has no TT)
    tt_probes_delta: int | None  # probe calls this depth iteration
    tt_hits_delta: int | None  # cache hits this depth iteration
    tt_stores_delta: int | None  # store calls this depth iteration
    tt_hit_rate: float | None  # hit rate = hits / probes
    tt_exact_delta: int | None  # EXACT hits this depth iteration
    tt_lowerbound_delta: int | None  # LOWERBOUND hits this depth iteration
    tt_upperbound_delta: int | None  # UPPERBOUND hits this depth iteration
    # Computed metrics
    ebf: float | None  # geometric EBF = nodes^(1/d)
    ebf_time: float | None  # wall-clock EBF = delta_time[d] / delta_time[d-1]
    nps: float | None  # nodes per second = cumulative_nodes / (cumulative_time_ms/1000)


@dataclass
class AlgoResult:
    """一個演算法在一個局面上的完整測量結果。"""

    algo_name: str
    position_name: str
    records: list[DepthRecord] = field(default_factory=list)
    error: str | None = None


@dataclass
class AlgoSpec:
    """描述一個待測演算法的全部參數。

    新增演算法：只需在 ALGO_REGISTRY 加入一筆 AlgoSpec，其餘程式碼無需修改。

    Attributes:
        name:         顯示名稱，同時作為 AlgoResult.algo_name 的鍵值。
        color:        matplotlib 顏色字串（用於折線 / 長條圖）。
        time_limit_s: 每局面最長搜尋秒數，超時自動停止。
        factory:      接受 max_depth: int，回傳 (Searcher, TT | None)。
    """

    name: str
    color: str
    time_limit_s: float
    factory: Callable[[int], tuple[Any, TranspositionTable | None]]


def _ev() -> ShannonEvaluator:
    return ShannonEvaluator()


def _make_alphabeta(d: int) -> tuple[AlphaBetaSearcher, None]:
    return AlphaBetaSearcher(_ev(), depth=d, tt=None), None


def _make_alphabeta_tt(d: int) -> tuple[AlphaBetaSearcher, TranspositionTable]:
    tt = TranspositionTable(size_mb=32)
    return AlphaBetaSearcher(_ev(), depth=d, tt=tt), tt


# register bench algo
ALGO_REGISTRY: list[AlgoSpec] = [
    AlgoSpec(
        name="AlphaBeta",
        color="#2ecc71",
        time_limit_s=30.0,
        factory=_make_alphabeta,
    ),
    AlgoSpec(
        name="AlphaBeta+TT",
        color="#3498db",
        time_limit_s=30.0,
        factory=_make_alphabeta_tt,
    ),
]

_COLOR_MAP: dict[str, str] = {s.name: s.color for s in ALGO_REGISTRY}


def run_single(
    spec: AlgoSpec,
    position_name: str,
    fen: str,
    max_depth: int,
) -> AlgoResult:
    """
    對一個演算法在一個局面上執行搜尋，逐深度收集指標。
    超過 spec.time_limit_s 則提前停止。
    """
    result = AlgoResult(algo_name=spec.name, position_name=position_name)
    board = chess.Board(fen)

    try:
        searcher, tt = spec.factory(max_depth)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    prev_nodes: int = 0
    prev_time_ms: float = 0.0
    prev_tt_probes: int = 0
    prev_tt_hits: int = 0
    prev_tt_stores: int = 0
    prev_tt_exact: int = 0
    prev_tt_lowerbound: int = 0
    prev_tt_upperbound: int = 0

    deadline = time.monotonic() + spec.time_limit_s
    timed_out = False

    def on_info(info: SearchInfo) -> None:
        nonlocal prev_nodes, prev_time_ms
        nonlocal prev_tt_probes, prev_tt_hits, prev_tt_stores
        nonlocal prev_tt_exact, prev_tt_lowerbound, prev_tt_upperbound
        nonlocal timed_out

        delta_nodes = info.nodes - prev_nodes
        delta_time = info.time_ms - prev_time_ms

        # TT statistics
        tt_probes_d: int | None = None
        tt_hits_d: int | None = None
        tt_stores_d: int | None = None
        tt_hit_rate: float | None = None
        tt_exact_d: int | None = None
        tt_lowerbound_d: int | None = None
        tt_upperbound_d: int | None = None
        if tt is not None:
            tt_probes_d = tt.probes - prev_tt_probes
            tt_hits_d = tt.hits - prev_tt_hits
            tt_stores_d = tt.stores - prev_tt_stores
            tt_exact_d = tt.exact_hits - prev_tt_exact
            tt_lowerbound_d = tt.lowerbound_hits - prev_tt_lowerbound
            tt_upperbound_d = tt.upperbound_hits - prev_tt_upperbound
            tt_hit_rate = (
                tt_hits_d / tt_probes_d if tt_probes_d and tt_probes_d > 0 else 0.0
            )

        # geometric EBF = nodes^(1/d)
        ebf: float | None = None
        if info.nodes > 0 and info.depth > 0:
            ebf = info.nodes ** (1.0 / info.depth)

        # wall-clock EBF = delta_time[d] / delta_time[d-1]
        ebf_time: float | None = None
        if (
            hasattr(on_info, "_prev_delta_time")
            and on_info._prev_delta_time > 0  # type: ignore[attr-defined]
            and delta_time > 0
        ):
            ebf_time = delta_time / on_info._prev_delta_time  # type: ignore[attr-defined]
        on_info._prev_delta_time = float(delta_time)  # type: ignore[attr-defined]

        # NPS = cumulative_nodes / total_elapsed_seconds
        nps: float | None = None
        if info.time_ms > 0:
            nps = info.nodes / (info.time_ms / 1000.0)

        record = DepthRecord(
            depth=info.depth,
            cumulative_nodes=info.nodes,
            delta_nodes=delta_nodes,
            cumulative_time_ms=float(info.time_ms),
            delta_time_ms=float(delta_time),
            tt_probes_delta=tt_probes_d,
            tt_hits_delta=tt_hits_d,
            tt_stores_delta=tt_stores_d,
            tt_hit_rate=tt_hit_rate,
            tt_exact_delta=tt_exact_d,
            tt_lowerbound_delta=tt_lowerbound_d,
            tt_upperbound_delta=tt_upperbound_d,
            ebf=ebf,
            ebf_time=ebf_time,
            nps=nps,
        )
        result.records.append(record)

        # update prev
        prev_nodes = info.nodes
        prev_time_ms = float(info.time_ms)
        if tt is not None:
            prev_tt_probes = tt.probes
            prev_tt_hits = tt.hits
            prev_tt_stores = tt.stores
            prev_tt_exact = tt.exact_hits
            prev_tt_lowerbound = tt.lowerbound_hits
            prev_tt_upperbound = tt.upperbound_hits

    def stop_condition() -> bool:
        return time.monotonic() > deadline

    try:
        searcher.search(
            board,
            on_info=on_info,
            stop_condition=stop_condition,
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    return result


# ===========================================================================
# 圖表輔助：跨局面平均 ± 標準差
# ===========================================================================


def _avg_by_depth(
    results: list[AlgoResult],
    algo: str,
    extractor,
    max_depth: int,
) -> tuple[list[int], list[float], list[float]]:
    """
    對指定演算法，把所有局面在同一深度的指標值彙整，
    回傳 (depths, means, stds)。只回傳至少有 1 個局面有資料的深度。
    """
    from collections import defaultdict

    depth_vals: dict[int, list[float]] = defaultdict(list)
    for res in results:
        if res.algo_name != algo or res.error:
            continue
        for rec in res.records:
            v = extractor(rec)
            if v is not None and not math.isnan(v) and not math.isinf(v):
                depth_vals[rec.depth].append(v)

    depths, means, stds = [], [], []
    for d in range(1, max_depth + 1):
        vals = depth_vals.get(d, [])
        if not vals:
            continue
        depths.append(d)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    return depths, means, stds


def _plot_metric_band(
    ax,
    results: list[AlgoResult],
    algos: list[str],
    extractor,
    max_depth: int,
    ylabel: str,
    title: str,
    marker: str = "o",
    log_y: bool = False,
    pct_y: bool = False,
) -> None:
    """在 ax 上繪製多演算法的 avg ± 1σ band 折線。"""
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Depth", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    if log_y:
        ax.set_yscale("log")
    if pct_y:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.set_ylim(-0.02, 1.02)

    for algo in algos:
        depths, means, stds = _avg_by_depth(results, algo, extractor, max_depth)
        if not depths:
            continue
        color = _COLOR_MAP.get(algo)
        d_arr = np.array(depths, dtype=float)
        m_arr = np.array(means)
        s_arr = np.array(stds)
        ax.plot(
            d_arr,
            m_arr,
            marker=marker,
            markersize=5,
            label=algo,
            color=color,
            linewidth=2,
        )
        ax.fill_between(d_arr, m_arr - s_arr, m_arr + s_arr, color=color, alpha=0.15)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# 圖表 1 – Performance Overview（2×2 總覽，跨局面平均）
# ---------------------------------------------------------------------------


def plot_perf_overview(
    results: list[AlgoResult],
    out_dir: Path,
    max_depth: int,
) -> None:
    """
    2×2 子圖，全部指標跨局面平均（±1σ 陰影帶）：
      [0,0] Node-EBF      — 越低代表 α-β 剪枝越有效，理想值 ≈ √b
      [0,1] Wall-clock EBF— 實際時間成長率，最直觀的效能指標
      [1,0] NPS           — 每秒節點數，顯示吞吐量與 TT 開銷
      [1,1] TT Hit Rate   — 轉置表命中率（僅有 TT 的演算法）
    """
    algos = list(dict.fromkeys(r.algo_name for r in results))
    tt_algos = [
        a
        for a in algos
        if any(
            any(rec.tt_hit_rate is not None for rec in r.records)
            for r in results
            if r.algo_name == a
        )
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Search Algorithm Performance Overview\n(mean ± 1σ across all test positions)",
        fontsize=13,
        fontweight="bold",
    )

    _plot_metric_band(
        axes[0, 0],
        results,
        algos,
        extractor=lambda r: r.ebf,
        max_depth=max_depth,
        ylabel="Node-EBF  (Δnodes[d] / Δnodes[d−1])",
        title="① Node-EBF vs Depth",
        marker="o",
    )
    axes[0, 0].axhline(
        math.sqrt(30),
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"ideal α-β ≈ √30 ≈ {math.sqrt(30):.1f}",
    )
    axes[0, 0].legend(fontsize=7)

    _plot_metric_band(
        axes[0, 1],
        results,
        algos,
        extractor=lambda r: r.ebf_time,
        max_depth=max_depth,
        ylabel="Wall-clock EBF  (Δtime[d] / Δtime[d−1])",
        title="② Wall-clock EBF vs Depth",
        marker="s",
    )
    axes[0, 1].axhline(
        math.sqrt(30),
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label="ideal α-β ≈ √30",
    )
    axes[0, 1].legend(fontsize=7)

    _plot_metric_band(
        axes[1, 0],
        results,
        algos,
        extractor=lambda r: r.nps,
        max_depth=max_depth,
        ylabel="NPS  (nodes / second)",
        title="③ NPS vs Depth",
        marker="^",
        log_y=True,
    )

    _plot_metric_band(
        axes[1, 1],
        results,
        tt_algos,
        extractor=lambda r: r.tt_hit_rate,
        max_depth=max_depth,
        ylabel="TT Hit Rate",
        title="④ TT Hit Rate vs Depth",
        marker="D",
        pct_y=True,
    )

    fig.tight_layout()
    out_path = out_dir / "fig_perf_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Performance Overview → {out_path}")


# ---------------------------------------------------------------------------
# 圖表 2 – Node & Time Reduction（AlphaBeta 為基準的縮減比率）
# ---------------------------------------------------------------------------


def plot_node_reduction(
    results: list[AlgoResult],
    out_dir: Path,
    max_depth: int,
) -> None:
    """
    以「AlphaBeta（無 TT）」為分母，計算其他演算法在每個深度的
    節點數比率與時間比率（跨局面平均）。
    比率 < 1 → 比基準搜更少節點 / 花更少時間。
    """
    algos = list(dict.fromkeys(r.algo_name for r in results))
    baseline = "AlphaBeta"
    compare_algos = [a for a in algos if a != baseline]

    if baseline not in algos or not compare_algos:
        print("  ⚠  node_reduction 需要 'AlphaBeta' 作為基準，略過")
        return

    positions = list(dict.fromkeys(r.position_name for r in results))

    from collections import defaultdict

    def collect_ratios(field: str) -> dict[str, dict[int, list[float]]]:
        ratios: dict[str, dict[int, list[float]]] = {
            a: defaultdict(list) for a in compare_algos
        }
        for pos_name in positions:
            base_res = next(
                (
                    r
                    for r in results
                    if r.algo_name == baseline and r.position_name == pos_name
                ),
                None,
            )
            if base_res is None or base_res.error:
                continue
            base_map = {rec.depth: getattr(rec, field) for rec in base_res.records}
            for algo in compare_algos:
                cmp_res = next(
                    (
                        r
                        for r in results
                        if r.algo_name == algo and r.position_name == pos_name
                    ),
                    None,
                )
                if cmp_res is None or cmp_res.error:
                    continue
                for rec in cmp_res.records:
                    base_val = base_map.get(rec.depth)
                    cmp_val = getattr(rec, field)
                    if (
                        base_val
                        and base_val > 0
                        and cmp_val is not None
                        and cmp_val > 0
                    ):
                        ratios[algo][rec.depth].append(cmp_val / base_val)
        return ratios

    node_ratios = collect_ratios("delta_nodes")
    time_ratios = collect_ratios("delta_time_ms")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Node & Time Reduction vs Baseline ({baseline})\n"
        "ratio < 1  →  fewer nodes / less time than baseline",
        fontsize=12,
        fontweight="bold",
    )

    for ax, ratios, label, unit in [
        (axes[0], node_ratios, "Node count ratio", "Δnodes[algo] / Δnodes[base]"),
        (axes[1], time_ratios, "Search time ratio", "Δtime[algo]  / Δtime[base]"),
    ]:
        ax.set_xlabel("Depth", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axhline(
            1.0, color="gray", linestyle="--", linewidth=1, label=f"{baseline} (=1)"
        )
        ax.grid(True, alpha=0.3, linestyle=":")

        for algo in compare_algos:
            d_r = ratios[algo]
            depths = sorted(d_r)
            if not depths:
                continue
            means = [float(np.mean(d_r[d])) for d in depths]
            stds = [float(np.std(d_r[d])) if len(d_r[d]) > 1 else 0.0 for d in depths]
            color = _COLOR_MAP.get(algo)
            d_arr = np.array(depths, dtype=float)
            m_arr = np.array(means)
            s_arr = np.array(stds)
            ax.plot(
                d_arr,
                m_arr,
                marker="o",
                markersize=5,
                label=algo,
                color=color,
                linewidth=2,
            )
            ax.fill_between(
                d_arr, m_arr - s_arr, m_arr + s_arr, color=color, alpha=0.15
            )
            if means:
                ax.annotate(
                    f"{means[-1]:.2f}×",
                    (depths[-1], means[-1]),
                    textcoords="offset points",
                    xytext=(6, 0),
                    fontsize=8,
                    color=color,
                )
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "fig_node_reduction.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Node & Time Reduction → {out_path}")


# ---------------------------------------------------------------------------
# 圖表 3 – Depth × Position 熱點圖（演算法並排）
# ---------------------------------------------------------------------------


def plot_depth_heatmap(
    results: list[AlgoResult],
    out_dir: Path,
    max_depth: int,
) -> None:
    """
    x 軸 = 深度, y 軸 = 局面，顏色 = 本深度迭代耗時 (ms, log 色階)
    所有演算法並排於同一圖，共用 y 軸，色階個別獨立。
    """
    algos = list(dict.fromkeys(r.algo_name for r in results))
    positions = list(dict.fromkeys(r.position_name for r in results))
    depths = list(range(1, max_depth + 1))

    n_algos = len(algos)
    fig, axes = plt.subplots(
        1,
        n_algos,
        figsize=(8 * n_algos, max(5, 0.65 * len(positions) + 2)),
        squeeze=False,
    )
    fig.suptitle(
        "Search Time Heatmap  (Depth × Position, ms per depth iteration)",
        fontsize=13,
        fontweight="bold",
    )

    for aidx, algo in enumerate(algos):
        ax = axes[0][aidx]
        ax.set_title(algo, fontsize=11, fontweight="bold")

        matrix = np.full((len(positions), len(depths)), np.nan)
        for pidx, pos_name in enumerate(positions):
            res = next(
                (
                    r
                    for r in results
                    if r.algo_name == algo and r.position_name == pos_name
                ),
                None,
            )
            if res is None or res.error:
                continue
            for rec in res.records:
                didx = rec.depth - 1
                if 0 <= didx < len(depths):
                    matrix[pidx, didx] = max(rec.delta_time_ms, 0.1)

        valid = matrix[~np.isnan(matrix)]
        vmin = max(float(valid.min()), 0.1) if len(valid) else 0.1
        vmax = max(float(valid.max()), vmin + 1.0) if len(valid) else 1.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", norm=norm)

        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels(depths, fontsize=8)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(
            positions if aidx == 0 else [""] * len(positions), fontsize=7
        )
        ax.set_xlabel("Depth", fontsize=9)
        if aidx == 0:
            ax.set_ylabel("Position", fontsize=9)

        for pidx in range(len(positions)):
            for didx in range(len(depths)):
                val = matrix[pidx, didx]
                if not np.isnan(val):
                    text = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
                    brightness = (np.log10(val) - np.log10(vmin)) / max(
                        np.log10(vmax) - np.log10(vmin), 1e-9
                    )
                    fg = "white" if brightness > 0.58 else "black"
                    ax.text(
                        didx, pidx, text, ha="center", va="center", fontsize=6, color=fg
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("ms (log)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path = out_dir / "fig_depth_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Depth-Time Heatmap → {out_path}")


# ---------------------------------------------------------------------------
# 文字摘要
# ---------------------------------------------------------------------------


def print_summary(results: list[AlgoResult]) -> None:
    SEP = "─" * 130
    print(f"\n{SEP}")
    print(
        f"  {'Algorithm':<18} {'Position':<28} {'Depth':>5} {'Nodes':>11} "
        f"{'ms':>8} {'EBF':>6} {'W-EBF':>7} {'NPS':>12}"
        f"  {'Probes':>8} {'Hits':>8} {'Hit%':>6} {'Exact%':>7} {'LB%':>6} {'UB%':>6}"
    )
    print(SEP)
    for res in results:
        if res.error:
            print(
                f"  {'  ' + res.algo_name:<18} {res.position_name:<28}   ERROR: {res.error}"
            )
            continue
        if not res.records:
            print(f"  {'  ' + res.algo_name:<18} {res.position_name:<28}   (no data)")
            continue
        last = res.records[-1]
        ebf_s = f"{last.ebf:.2f}" if last.ebf is not None else "  N/A"
        webf_s = f"{last.ebf_time:.2f}" if last.ebf_time is not None else "  N/A"
        nps_s = f"{last.nps:>12,.0f}" if last.nps is not None else "         N/A"
        # TT columns (sum over all recorded depths for this position)
        if last.tt_probes_delta is not None:
            tot_probes = sum(r.tt_probes_delta or 0 for r in res.records)
            tot_hits = sum(r.tt_hits_delta or 0 for r in res.records)
            tot_exact = sum(r.tt_exact_delta or 0 for r in res.records)
            tot_lb = sum(r.tt_lowerbound_delta or 0 for r in res.records)
            tot_ub = sum(r.tt_upperbound_delta or 0 for r in res.records)
            hit_pct = tot_hits / tot_probes if tot_probes > 0 else 0.0
            exact_pct = tot_exact / tot_hits if tot_hits > 0 else 0.0
            lb_pct = tot_lb / tot_hits if tot_hits > 0 else 0.0
            ub_pct = tot_ub / tot_hits if tot_hits > 0 else 0.0
            tt_cols = (
                f"  {tot_probes:>8,} {tot_hits:>8,} {hit_pct:>6.1%}"
                f" {exact_pct:>7.1%} {lb_pct:>6.1%} {ub_pct:>6.1%}"
            )
        else:
            tt_cols = (
                f"  {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>7} {'N/A':>6} {'N/A':>6}"
            )
        print(
            f"  {res.algo_name:<18} {res.position_name:<28} "
            f"{last.depth:>5} {last.cumulative_nodes:>11,} "
            f"{last.cumulative_time_ms:>8.0f} {ebf_s:>6} {webf_s:>7} "
            f"{nps_s}{tt_cols}"
        )
    print(SEP + "\n")


def main() -> None:
    print(f"\n{'=' * 60}")
    print("  Chess Engine Algorithm Benchmark")
    print(f"  最大深度: {MAX_DEPTH}   輸出目錄: {OUT_DIR}")
    print(f"{'=' * 60}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[AlgoResult] = []
    for spec in ALGO_REGISTRY:
        for pos_name, fen in TEST_POSITIONS.items():
            print(f"▶ {spec.name} | {pos_name}", flush=True)
            res = run_single(spec, pos_name, fen, MAX_DEPTH)
            if res.records:
                last = res.records[-1]
                status = f"深度 {last.depth}  nodes={last.cumulative_nodes:,}"
            else:
                status = f"ERROR: {res.error}" if res.error else "(無資料)"
            print(f"  → {status}")
            all_results.append(res)

    print_summary(all_results)

    print("產生圖表…")
    plot_perf_overview(all_results, OUT_DIR, MAX_DEPTH)  # 總覽 2×2
    plot_node_reduction(all_results, OUT_DIR, MAX_DEPTH)  # 節點 & 時間縮減比率
    plot_depth_heatmap(all_results, OUT_DIR, MAX_DEPTH)  # Depth × Position 熱點圖

    print(f"\n全部圖表已儲存至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
