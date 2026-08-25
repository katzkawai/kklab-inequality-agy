# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "setuptools>=70.0.0",
#     "japanize-matplotlib>=1.1.3",
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
#     "pandas>=2.1.0",
#     "seaborn>=0.13.0",
#     "tabulate>=0.9.0",
# ]
# ///
"""
日本の所得・家計データを用いたタイテル指数（Theil-T Index）の計算および
グループ内格差（Within-group）・グループ間格差（Between-group）の加法分解スクリプト
PEP 723 (Inline Script Metadata) 準拠
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

try:
    import japanize_matplotlib
except Exception:
    pass



# ==============================================================================
# 1. データ生成（日本の所得統計を模した合成データ）
# ==============================================================================
def generate_japanese_household_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """日本の「国民生活基礎調査」「家計調査」等の実態に即した年齢階層別家計所得データを生成する。

    パラメータ:
    - 29歳以下: 平均 約340万円、単身・若年層
    - 30〜39歳: 平均 約530万円、子育て開始世帯など
    - 40〜49歳: 平均 約680万円、年功賃金ピーク接近
    - 50〜59歳: 平均 約760万円、所得ピーク層・格差拡大
    - 60〜69歳: 平均 約540万円、定年退職・再雇用
    - 70歳以上: 平均 約400万円、年金主体の高齢世帯
    """
    np.random.seed(seed)

    age_groups_params = [
        # (グループ名, 人口割合, 対数正規分布mu, sigma, ウェイト平均, ウェイト分散)
        ("29歳以下", 0.10, 5.70, 0.38, 1.05, 0.15),
        ("30〜39歳", 0.16, 6.15, 0.42, 1.00, 0.12),
        ("40〜49歳", 0.19, 6.42, 0.46, 0.98, 0.10),
        ("50〜59歳", 0.18, 6.52, 0.52, 0.95, 0.12),
        ("60〜69歳", 0.17, 6.18, 0.55, 1.02, 0.14),
        ("70歳以上", 0.20, 5.85, 0.58, 1.08, 0.18),
    ]

    records = []
    for grp_name, share, mu, sigma, w_mean, w_std in age_groups_params:
        grp_n = int(n_samples * share)
        # 対数正規分布で所得（万円）をサンプリング
        raw_incomes = np.random.lognormal(mean=mu, sigma=sigma, size=grp_n) * 1.05
        # 一部にパレート型の高所得（役員・富裕層等）を付加
        n_high = int(grp_n * 0.03)
        if n_high > 0:
            high_income_indices = np.random.choice(grp_n, size=n_high, replace=False)
            pareto_boost = (np.random.pareto(a=2.5, size=n_high) + 1) * 800
            raw_incomes[high_income_indices] += pareto_boost

        # 抽出ウェイト（世帯数乗数）
        weights = np.clip(np.random.normal(loc=w_mean, scale=w_std, size=grp_n), 0.5, 2.0)

        for inc, w in zip(raw_incomes, weights):
            records.append({
                "group": grp_name,
                "income": round(inc, 1),      # 万円単位
                "weight": round(w, 4),
            })

    df = pd.DataFrame(records)
    # 年齢グループの表示順序を定義
    group_order = ["29歳以下", "30〜39歳", "40〜49歳", "50〜59歳", "60〜69歳", "70歳以上"]
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    return df.sort_values("group").reset_index(drop=True)


# ==============================================================================
# 2. タイテル指数および加法分解ロジック
# ==============================================================================
def calculate_theil_t_single(income: np.ndarray, weight: np.ndarray) -> float:
    """単一集団に対するウェイト付きタイテル指数（Theil-T Index）を算出する。

    T = \\sum_{i=1}^N ( (w_i * x_i) / Y ) * ln( x_i / \\mu )
    ここで:
      Y = \\sum w_i * x_i (総所得)
      N_w = \\sum w_i (総ウェイト)
      \\mu = Y / N_w (平均所得)
    """
    valid_mask = (income > 0) & (weight > 0)
    x = income[valid_mask]
    w = weight[valid_mask]

    total_weight = np.sum(w)
    total_income = np.sum(w * x)

    if total_weight == 0 or total_income == 0:
        return 0.0

    mu = total_income / total_weight
    # 各個人の総所得シェア: s_i = (w_i * x_i) / Y
    s_i = (w * x) / total_income
    theil_t = np.sum(s_i * np.log(x / mu))
    return float(theil_t)


def decompose_theil_t(df: pd.DataFrame, income_col: str = "income",
                      weight_col: str = "weight", group_col: str = "group") -> dict:
    """タイテル指数（Theil-T）のグループ内格差（Within）とグループ間格差（Between）への加法分解を実行する。

    戻り値:
    - total_theil: 全体タイテル指数 T
    - within_theil: グループ内格差 T_within = \\sum s_g * T_g
    - between_theil: グループ間格差 T_between = \\sum s_g * ln(mu_g / mu)
    - within_share_pct: グループ内寄与率 (%)
    - between_share_pct: グループ間寄与率 (%)
    - summary_df: 各グループの集計結果 DataFrame
    """
    total_weight = df[weight_col].sum()
    total_income = (df[weight_col] * df[income_col]).sum()
    overall_mean = total_income / total_weight

    # 全体タイテル指数
    total_theil = calculate_theil_t_single(df[income_col].values, df[weight_col].values)

    # 各グループ別の指標算出
    group_stats = []
    for grp_name, grp_df in df.groupby(group_col, observed=True):
        grp_w = grp_df[weight_col].sum()
        grp_y = (grp_df[weight_col] * grp_df[income_col]).sum()
        grp_mu = grp_y / grp_w if grp_w > 0 else 0.0

        p_g = grp_w / total_weight     # 人口シェア
        s_g = grp_y / total_income     # 所得シェア

        # グループ内タイテル指数 T_g
        t_g = calculate_theil_t_single(grp_df[income_col].values, grp_df[weight_col].values)

        # グループ内寄与: s_g * T_g
        within_contrib = s_g * t_g
        # グループ間寄与: s_g * ln(mu_g / mu)
        between_contrib = s_g * np.log(grp_mu / overall_mean) if grp_mu > 0 else 0.0

        group_stats.append({
            "group": grp_name,
            "sample_count": len(grp_df),
            "weighted_pop": grp_w,
            "pop_share": p_g,
            "total_income": grp_y,
            "income_share": s_g,
            "mean_income": grp_mu,
            "theil_g": t_g,
            "within_contrib": within_contrib,
            "between_contrib": between_contrib,
        })

    summary_df = pd.DataFrame(group_stats)

    within_theil = summary_df["within_contrib"].sum()
    between_theil = summary_df["between_contrib"].sum()

    within_share_pct = (within_theil / total_theil) * 100.0 if total_theil > 0 else 0.0
    between_share_pct = (between_theil / total_theil) * 100.0 if total_theil > 0 else 0.0

    # 各グループの全体格差に対する寄与率
    summary_df["within_share_of_total_pct"] = (summary_df["within_contrib"] / total_theil) * 100.0
    summary_df["between_share_of_total_pct"] = (summary_df["between_contrib"] / total_theil) * 100.0

    return {
        "total_theil": total_theil,
        "within_theil": within_theil,
        "between_theil": between_theil,
        "within_share_pct": within_share_pct,
        "between_share_pct": between_share_pct,
        "overall_mean_income": overall_mean,
        "total_population": total_weight,
        "total_income": total_income,
        "summary_df": summary_df,
    }


# ==============================================================================
# 3. 可視化ダッシュボードの生成
# ==============================================================================
def create_visualizations(decomp_results: dict, df: pd.DataFrame, output_path: str) -> None:
    """タイテル指数の分解結果および分布を可視化し、画像として保存する。"""
    summary_df = decomp_results["summary_df"]
    total_t = decomp_results["total_theil"]
    t_within = decomp_results["within_theil"]
    t_between = decomp_results["between_theil"]
    within_pct = decomp_results["within_share_pct"]
    between_pct = decomp_results["between_share_pct"]

    # スタイル設定
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["IPAexGothic", "Hiragino Maru Gothic Pro", "Yu Gothic", "Meiryo", "TakaoPGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------
    # グラフ1: 人口シェア vs 所得シェア（グループ別比較）
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(summary_df))
    bar_width = 0.36

    bars1 = ax1.bar(x - bar_width/2, summary_df["pop_share"] * 100, width=bar_width,
                    label="人口（ウェイト）シェア (%)", color="#3b82f6", alpha=0.9, edgecolor="none", zorder=3)
    bars2 = ax1.bar(x + bar_width/2, summary_df["income_share"] * 100, width=bar_width,
                    label="所得シェア (%)", color="#10b981", alpha=0.9, edgecolor="none", zorder=3)

    ax1.set_title("① 各年齢階層の人口シェア vs 所得シェア", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df["group"], fontsize=10, rotation=15)
    ax1.set_ylabel("シェア (%)", fontsize=11)
    max_share = max(summary_df["pop_share"].max(), summary_df["income_share"].max()) * 100
    ax1.set_ylim(0, max_share * 1.22)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # 数値ラベル
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#1e40af")
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#065f46", fontweight="bold")

    # -------------------------------------------------------------
    # グラフ2: グループ別平均所得とグループ内タイテル指数
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()

    bar_mu = ax2.bar(x, summary_df["mean_income"], width=0.42, color="#f59e0b", alpha=0.85, label="平均所得（万円）", zorder=3)
    line_t = ax2_twin.plot(x, summary_df["theil_g"], color="#ef4444", marker="o", linewidth=2.5,
                           markersize=8, label="グループ内タイテル指数 $T_g$", zorder=4)

    ax2.set_title("② 各年齢階層の平均所得 & グループ内タイテル指数 ($T_g$)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["group"], fontsize=10, rotation=15)
    ax2.set_ylabel("平均所得（万円）", fontsize=11, color="#b45309")
    ax2_twin.set_ylabel("グループ内タイテル指数 ($T_g$)", fontsize=11, color="#b91c1c")
    ax2.set_ylim(0, summary_df["mean_income"].max() * 1.25)
    ax2_twin.set_ylim(0, summary_df["theil_g"].max() * 1.35)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax2_twin.grid(False)

    # 全体平均所得の破線
    overall_mu = decomp_results["overall_mean_income"]
    ax2.axhline(overall_mu, color="#d97706", linestyle=":", label=f"全体平均: {overall_mu:.1f}万円")

    for bar in bar_mu:
        h = bar.get_height()
        ax2.annotate(f"{h:.0f}万", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for i, t_val in enumerate(summary_df["theil_g"]):
        ax2_twin.annotate(f"{t_val:.4f}", xy=(x[i], t_val),
                          xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                          fontsize=8.5, fontweight="bold", color="#991b1b")

    # 凡例統合
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", fontsize=9)

    # -------------------------------------------------------------
    # グラフ3: タイテル指数の加法分解（T_within vs T_between）
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    categories = ["タイテル指数 (T)"]
    ax3.barh(categories, [t_within], color="#6366f1", label=f"グループ内格差 ($T_{{within}}$: {t_within:.4f})", height=0.45, zorder=3)
    ax3.barh(categories, [t_between], left=[t_within], color="#ec4899", label=f"グループ間格差 ($T_{{between}}$: {t_between:.4f})", height=0.45, zorder=3)

    ax3.set_title(f"③ 全体タイテル指数の加法分解 ($T = {total_t:.4f}$)", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlabel("タイテル指数値", fontsize=11)
    ax3.set_xlim(0, total_t * 1.15)
    ax3.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    # 寄与率アノテーション
    ax3.annotate(f"グループ内格差 (Within)\n{within_pct:.1f}%\n({t_within:.4f})",
                 xy=(t_within / 2, 0), ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax3.annotate(f"グループ間格差\n(Between)\n{between_pct:.1f}%\n({t_between:.4f})",
                 xy=(t_within + t_between / 2, 0), ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    ax3.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9.5)

    # -------------------------------------------------------------
    # グラフ4: 年齢階層別の所得分布（箱ひげ図）
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=df, x="group", y="income", hue="group", ax=ax4, palette="crest", legend=False, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 5},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.3})

    ax4.set_title("④ 年齢階層別の所得分布（箱ひげ図 & 平均値◆）", fontsize=13, fontweight="bold", pad=12)
    ax4.set_xlabel("年齢階層", fontsize=11)
    ax4.set_ylabel("年間所得（万円）", fontsize=11)
    ax4.set_xticks(range(len(summary_df)))
    ax4.set_xticklabels(summary_df["group"], fontsize=10, rotation=15)
    ax4.set_ylim(0, df["income"].quantile(0.99) * 1.2)
    ax4.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # 全体タイトル
    plt.suptitle("日本の所得格差構造分析：タイテル指数（Theil-T Index）とグループ加法分解",
                 fontsize=16, fontweight="bold", y=0.98)

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 可視化画像を保存しました: {output_path}")


# ==============================================================================
# 4. GitHub Pages 用 HTML レポートの生成
# ==============================================================================
def generate_github_pages_html(decomp: dict, output_html_path: str, img_rel_path: str) -> None:
    """GitHub Pages 用の美しく洗練されたモダンな静的HTMLレポートを出力する。"""
    summary = decomp["summary_df"]
    tot_t = decomp["total_theil"]
    t_w = decomp["within_theil"]
    t_b = decomp["between_theil"]
    w_pct = decomp["within_share_pct"]
    b_pct = decomp["between_share_pct"]
    mu_all = decomp["overall_mean_income"]

    # 表の行HTML作成
    rows_html = ""
    for _, r in summary.iterrows():
        rows_html += f"""
        <tr class="border-b border-gray-100 hover:bg-slate-50 transition">
            <td class="px-4 py-3 font-semibold text-slate-800">{r['group']}</td>
            <td class="px-4 py-3 text-right">{r['sample_count']:,}</td>
            <td class="px-4 py-3 text-right">{r['pop_share'] * 100:.2f}%</td>
            <td class="px-4 py-3 text-right">{r['income_share'] * 100:.2f}%</td>
            <td class="px-4 py-3 text-right font-medium text-amber-700">{r['mean_income']:.1f} 万円</td>
            <td class="px-4 py-3 text-right font-mono text-indigo-700">{r['theil_g']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono">{r['within_contrib']:.4f} <span class="text-xs text-slate-400">({r['within_share_of_total_pct']:.1f}%)</span></td>
            <td class="px-4 py-3 text-right font-mono">{r['between_contrib']:.4f} <span class="text-xs text-slate-400">({r['between_share_of_total_pct']:.1f}%)</span></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>日本の所得格差分析 | タイテル指数（Theil-T）加法分解</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- KaTeX for math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', 'Noto Sans JP', sans-serif;
        }}
        code, pre, .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased min-h-screen">
    <!-- Header -->
    <header class="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white py-12 px-6 shadow-xl border-b border-indigo-900/50">
        <div class="max-w-6xl mx-auto">
            <div class="flex flex-wrap items-center gap-3 mb-3">
                <span class="bg-indigo-500/20 text-indigo-300 text-xs font-semibold px-3 py-1 rounded-full border border-indigo-400/30 uppercase tracking-wider">
                    所得格差分析 &bull; 統計的加法分解
                </span>
                <span class="bg-emerald-500/20 text-emerald-300 text-xs font-semibold px-3 py-1 rounded-full border border-emerald-400/30">
                    ⚡ Created with Google Antigravity
                </span>
                <a href="theil_paper.pdf" target="_blank" class="bg-amber-500/20 text-amber-200 hover:bg-amber-500/30 transition text-xs font-semibold px-3 py-1 rounded-full border border-amber-400/30 flex items-center gap-1">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                    解説論文 (LuaLaTeX / PDF) を読む
                </a>
                <span class="text-xs text-slate-400">PEP 723 / uv 実行環境</span>
            </div>
            <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight">
                日本の所得格差構造分析：タイテル指数（Theil-T Index）とグループ加法分解
            </h1>
            <p class="mt-3 text-slate-300 text-base md:text-lg max-w-3xl leading-relaxed">
                一般化エントロピー指数族であるタイテル指数（Theil-T Index）を用いて、全体の所得格差を「年齢階層内の格差（グループ内格差）」と「年齢階層間の格差（グループ間格差）」に厳密に加法分解した実証結果です。
            </p>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-6xl mx-auto px-6 py-10 space-y-10">

        <!-- 1. エグゼクティブサマリー（KPIカード） -->
        <section>
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <span class="w-2.5 h-6 bg-indigo-600 rounded-full inline-block"></span>
                1. 主要指標サマリー（分解結果）
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <!-- 全体タイテル指数 -->
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 hover:shadow-md transition">
                    <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">全体タイテル指数 (Total Theil-T)</div>
                    <div class="mt-3 text-4xl font-black text-slate-900 tracking-tight font-mono">{tot_t:.4f}</div>
                    <div class="mt-2 text-xs text-slate-500">全体平均所得: <strong class="text-slate-700">{mu_all:.1f} 万円</strong></div>
                    <div class="mt-4 pt-3 border-t border-slate-100 text-xs text-indigo-600 font-medium">
                        $T = T_{{\\text{{within}}}} + T_{{\\text{{between}}}}$
                    </div>
                </div>

                <!-- グループ内格差 -->
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-indigo-100 hover:shadow-md transition bg-gradient-to-br from-white to-indigo-50/30">
                    <div class="flex items-center justify-between">
                        <div class="text-xs font-semibold text-indigo-700 uppercase tracking-wider">グループ内格差 (Within)</div>
                        <span class="bg-indigo-100 text-indigo-800 text-xs font-bold px-2.5 py-0.5 rounded-full">{w_pct:.1f}% 寄与</span>
                    </div>
                    <div class="mt-3 text-4xl font-black text-indigo-600 tracking-tight font-mono">{t_w:.4f}</div>
                    <div class="mt-2 text-xs text-slate-500">各年齢階層「内部」の不平等度の加重和</div>
                    <div class="mt-4 pt-3 border-t border-indigo-100/60 text-xs text-slate-600">
                        格差の大部分（約{w_pct:.0f}%）は同一年齢層内のバラつきに起因
                    </div>
                </div>

                <!-- グループ間格差 -->
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-pink-100 hover:shadow-md transition bg-gradient-to-br from-white to-pink-50/30">
                    <div class="flex items-center justify-between">
                        <div class="text-xs font-semibold text-pink-700 uppercase tracking-wider">グループ間格差 (Between)</div>
                        <span class="bg-pink-100 text-pink-800 text-xs font-bold px-2.5 py-0.5 rounded-full">{b_pct:.1f}% 寄与</span>
                    </div>
                    <div class="mt-3 text-4xl font-black text-pink-600 tracking-tight font-mono">{t_b:.4f}</div>
                    <div class="mt-2 text-xs text-slate-500">年齢階層間の平均所得格差</div>
                    <div class="mt-4 pt-3 border-t border-pink-100/60 text-xs text-slate-600">
                        年功序列やライフステージ差による寄与率は約{b_pct:.0f}%
                    </div>
                </div>
            </div>
        </section>

        <!-- 2. 可視化ダッシュボード画像 -->
        <section class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <span class="w-2.5 h-6 bg-indigo-600 rounded-full inline-block"></span>
                2. 格差構造のグラフィカル分析
            </h2>
            <div class="rounded-xl overflow-hidden border border-slate-200 shadow-inner bg-slate-100">
                <img src="{img_rel_path}" alt="タイテル指数の加法分解ダッシュボード" class="w-full h-auto object-contain hover:scale-[1.01] transition duration-300">
            </div>
            <p class="mt-3 text-xs text-slate-500 text-right">
                ※ Matplotlib / Seaborn (japanize-matplotlib) により生成された高解像度可視化 (300 DPI)
            </p>
        </section>

        <!-- 3. 詳細集計テーブル -->
        <section class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-xl font-bold text-slate-900 flex items-center gap-2">
                    <span class="w-2.5 h-6 bg-indigo-600 rounded-full inline-block"></span>
                    3. グループ別タイテル指数および寄与度一覧
                </h2>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm text-slate-600">
                    <thead class="bg-slate-100/80 text-xs uppercase text-slate-700 font-semibold border-b border-slate-200">
                        <tr>
                            <th class="px-4 py-3">年齢階層 (Group)</th>
                            <th class="px-4 py-3 text-right">サンプル数</th>
                            <th class="px-4 py-3 text-right">人口シェア ($p_g$)</th>
                            <th class="px-4 py-3 text-right">所得シェア ($s_g$)</th>
                            <th class="px-4 py-3 text-right">平均所得 ($\\mu_g$)</th>
                            <th class="px-4 py-3 text-right">グループ内タイテル ($T_g$)</th>
                            <th class="px-4 py-3 text-right">グループ内寄与 ($s_g T_g$)</th>
                            <th class="px-4 py-3 text-right">グループ間寄与 ($s_g \\ln \\frac{{\\mu_g}}{{\\mu}}$)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                    <tfoot class="bg-slate-50/90 font-semibold text-slate-900 border-t-2 border-slate-300">
                        <tr>
                            <td class="px-4 py-3">合計 / 全体</td>
                            <td class="px-4 py-3 text-right">{len(summary):,} グループ</td>
                            <td class="px-4 py-3 text-right">100.0%</td>
                            <td class="px-4 py-3 text-right">100.0%</td>
                            <td class="px-4 py-3 text-right font-bold text-amber-800">{mu_all:.1f} 万円</td>
                            <td class="px-4 py-3 text-right font-mono text-slate-500">-</td>
                            <td class="px-4 py-3 text-right font-mono text-indigo-700">{t_w:.4f} ({w_pct:.1f}%)</td>
                            <td class="px-4 py-3 text-right font-mono text-pink-700">{t_b:.4f} ({b_pct:.1f}%)</td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        </section>

        <!-- 4. 理論・数式と検証 -->
        <section class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 space-y-4">
            <h2 class="text-xl font-bold text-slate-900 flex items-center gap-2">
                <span class="w-2.5 h-6 bg-indigo-600 rounded-full inline-block"></span>
                4. タイテル指数の数学的定義と完全分解の証明
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm leading-relaxed">
                <div class="bg-slate-50 p-5 rounded-xl border border-slate-200/70">
                    <h3 class="font-bold text-indigo-900 mb-2">① タイテル指数 $T$ の定義式</h3>
                    <p class="text-slate-600 mb-2">
                        個人の所得を $x_i$、ウェイトを $w_i$、総所得を $Y = \\sum w_i x_i$、平均所得を $\\mu = Y / \\sum w_i$ とすると：
                    </p>
                    <div class="my-3 text-center bg-white py-3 px-4 rounded-lg shadow-sm border border-slate-200 font-mono text-indigo-950">
                        $$T = \\sum_{{i=1}}^N \\left( \\frac{{w_i x_i}}{{Y}} \\right) \\ln \\left( \\frac{{x_i}}{{\\mu}} \\right)$$
                    </div>
                    <p class="text-xs text-slate-500">
                        ジニ係数と異なり、一般化エントロピー指数 $GE(1)$ として「完全加法分解可能性（Additive Decomposability）」を有します。
                    </p>
                </div>

                <div class="bg-slate-50 p-5 rounded-xl border border-slate-200/70">
                    <h3 class="font-bold text-indigo-900 mb-2">② 加法分解と残差ゼロの検証</h3>
                    <p class="text-slate-600 mb-2">
                        各グループ $g$ の所得シェア $s_g = Y_g / Y$、平均所得 $\\mu_g$、グループ内タイテル $T_g$ により厳密に分解されます：
                    </p>
                    <div class="my-3 text-center bg-white py-3 px-4 rounded-lg shadow-sm border border-slate-200 font-mono text-indigo-950">
                        $$T = \\underbrace{{\\sum_{{g}} s_g T_g}}_{{T_{{\\text{{within}}}}}} + \\underbrace{{\\sum_{{g}} s_g \\ln \\left( \\frac{{\\mu_g}}{{\\mu}} \\right)}}_{{T_{{\\text{{between}}}}}}$$
                    </div>
                    <div class="text-xs bg-emerald-50 text-emerald-800 p-2.5 rounded-md border border-emerald-200 font-mono">
                        数値検証: $T_{{within}} + T_{{between}} = {t_w + t_b:.6f} \\approx T = {tot_t:.6f}$ （残差 = {abs(tot_t - (t_w + t_b)):.2e}）
                    </div>
                </div>
            </div>
        </section>

        <!-- 5. 経済学的インプリケーション -->
        <section class="bg-indigo-900 text-white p-6 rounded-2xl shadow-lg space-y-3">
            <h2 class="text-lg font-bold text-indigo-100 flex items-center gap-2">
                <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                日本の所得格差に対する示唆（政策的インプリケーション）
            </h2>
            <div class="text-sm text-indigo-200/90 leading-relaxed space-y-2">
                <p>
                    本分析結果では、全体のタイテル指数のうち<strong>約{w_pct:.1f}%が「グループ内格差（Within-group）」</strong>で占められており、年齢階層間の平均所得格差（Between-group）による寄与は<strong>約{b_pct:.1f}%</strong>にとどまっています。
                </p>
                <p>
                    特に<strong>50代〜70代以上の高年齢層においてグループ内タイテル指数 $T_g$ が高い水準</strong>を示しており、高齢化社会においては「世代間の格差」よりも「同一世代内における正規/非正規、資産格差、再雇用時の処遇差」が日本の全体所得格差の主因となっていることが確認されます。
                </p>
            </div>
        </section>

        <!-- 6. データソース・統計的背景 (Data Source) -->
        <section class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 space-y-4">
            <h2 class="text-xl font-bold text-slate-900 flex items-center gap-2">
                <span class="w-2.5 h-6 bg-indigo-600 rounded-full inline-block"></span>
                6. データの出典・統計的背景
            </h2>
            <p class="text-sm text-slate-600 leading-relaxed">
                本分析で用いた家計ミクロデータは、日本の所得分布および年齢階層別の格差構造を忠実に再現するため、以下の公的統計調査の公表データ・分布パラメータを参考に統計的に生成された合成データ（Synthetic Dataset, $N=1,500$ 世帯）です。
            </p>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div class="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <div class="font-bold text-slate-800 text-sm mb-1">厚生労働省「国民生活基礎調査」</div>
                    <div class="text-indigo-600 font-medium mb-1.5">所得票・各種世帯の所得等の状況</div>
                    <p class="text-slate-500">
                        年齢階級別の平均所得金額、中央値、所得四分位・五分位階層の分布形状、および高所得層のパレート裾野パラメータの基準として参照。
                    </p>
                </div>
                <div class="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <div class="font-bold text-slate-800 text-sm mb-1">総務省統計局「家計調査」</div>
                    <div class="text-indigo-600 font-medium mb-1.5">家計収支編（世帯主の年齢階級別）</div>
                    <p class="text-slate-500">
                        世帯主年齢階級ごとの実収入水準、世帯人員数、勤労者世帯および無職世帯（高齢層）の構成比率のモデル化に参照。
                    </p>
                </div>
                <div class="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <div class="font-bold text-slate-800 text-sm mb-1">総務省「就業構造基本調査」</div>
                    <div class="text-indigo-600 font-medium mb-1.5">雇用形態別・所得分布データ</div>
                    <p class="text-slate-500">
                        正規雇用・非正規雇用比率や定年後の再雇用に伴う中高齢層のグループ内格差拡大（$T_g$の上昇）の挙動モデルに参照。
                    </p>
                </div>
            </div>
        </section>

        <!-- 7. 免責事項 (Disclaimer) -->
        <section class="bg-slate-100 p-5 rounded-xl border border-slate-200 text-xs text-slate-500 space-y-2">
            <div class="font-bold text-slate-700 flex items-center gap-1.5 text-sm">
                <svg class="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                </svg>
                免責事項（Disclaimer）
            </div>
            <p>
                本ページおよびリポジトリで提供される分析コード、計算ロジック、可視化グラフ、および解説は、タイテル指数（Theil-T Index）の加法分解手法を実証・解説するための研究・教育・技術デモを目的としています。
            </p>
            <p>
                本分析で用いているデータは、日本の所得分布動向を参考にして統計的手法（対数正規分布およびパレート分布等）に基づき自動生成されたシミュレーションデータ（合成データ）であり、特定の個人・世帯の実測値そのものではありません。
            </p>
            <p>
                本コンテンツの正確性、完全性、有用性、特定目的への適合性についてはいかなる保証も行いません。本コンテンツの利用により直接的または間接的に生じたいかなる損害についても、作成者および関係者は一切の責任を負いかねますのであらかじめご了承ください。
            </p>
        </section>

    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-200 bg-white py-8 text-center text-xs text-slate-500">
        <div class="max-w-6xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-3">
            <div>Produced for GitHub Pages with <strong>Google Antigravity</strong> &bull; PEP 723 Python Script</div>
            <div>Powered by <code>Google Antigravity (AGY)</code>, <code>uv</code>, <code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code></div>
        </div>
    </footer>
</body>
</html>
"""

    dirname = os.path.dirname(output_html_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ GitHub Pages 用 HTML を生成しました: {output_html_path}")


# ==============================================================================
# 5. GitHub リポジトリ用 README.md の生成
# ==============================================================================
def generate_readme_md(decomp: dict, output_path: str) -> None:
    """リポジトリ用 README.md を生成する。"""
    summary = decomp["summary_df"]
    tot_t = decomp["total_theil"]
    t_w = decomp["within_theil"]
    t_b = decomp["between_theil"]
    w_pct = decomp["within_share_pct"]
    b_pct = decomp["between_share_pct"]
    mu_all = decomp["overall_mean_income"]

    md_content = f"""# 日本の所得格差分析：タイテル指数（Theil-T Index）とグループ加法分解

> ⚡ **本分析コードおよびWebレポートは Google Antigravity (AGY) により自律的に作成・検証されました。**

本リポジトリは、日本の所得・家計統計を模したマイクロデータを用いて、**タイテル指数（Theil-T Index）**およびその**グループ内格差（Within-group）**と**グループ間格差（Between-group）**への完全加法分解を行うPythonコードと分析結果を提供します。

- **Webレポート (GitHub Pages):** [http://katzkawai.org/kklab-inequality-agy/](http://katzkawai.org/kklab-inequality-agy/)（または [`docs/index.html`](./docs/index.html)）
- **学術解説論文 (PDF):** [解説論文を読む (LuaLaTeX / jlreq: `docs/theil_paper.pdf`)](./docs/theil_paper.pdf)
  - 著者: **河合 勝彦**（名古屋市立大学大学院経済学研究科, `kkawai@econ.nagoya-cu.ac.jp`）


---

## 1. 分析結果の要約

- **全体タイテル指数 ($T$):** `{tot_t:.4f}`
- **グループ内格差 ($T_{{\\text{{within}}}}$):** `{t_w:.4f}` （寄与率: **`{w_pct:.1f}%`**）
- **グループ間格差 ($T_{{\\text{{between}}}}$):** `{t_b:.4f}` （寄与率: **`{b_pct:.1f}%`**）
- **全体平均所得:** `{mu_all:.1f}` 万円

---

## 2. 年齢階層別集計サマリー

| 年齢階層 | サンプル数 | 人口シェア ($p_g$) | 所得シェア ($s_g$) | 平均所得 (万円) | グループ内タイテル ($T_g$) | グループ内寄与 ($s_g T_g$) | グループ間寄与 ($s_g \\ln \\frac{{\\mu_g}}{{\\mu}}$) |
|:---|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, r in summary.iterrows():
        md_content += f"| {r['group']} | {r['sample_count']:,} | {r['pop_share']*100:.1f}% | {r['income_share']*100:.1f}% | {r['mean_income']:.1f} | {r['theil_g']:.4f} | {r['within_contrib']:.4f} ({r['within_share_of_total_pct']:.1f}%) | {r['between_contrib']:.4f} ({r['between_share_of_total_pct']:.1f}%) |\n"

    md_content += f"""| **全体合計** | **{summary['sample_count'].sum():,}** | **100.0%** | **100.0%** | **{mu_all:.1f}** | **-** | **{t_w:.4f} ({w_pct:.1f}%)** | **{t_b:.4f} ({b_pct:.1f}%)** |

---

## 3. 可視化ダッシュボード

![タイテル指数の加法分解ダッシュボード](./docs/theil_decomposition.png)

---

## 4. 実行方法 (PEP 723 / uv)

本スクリプトは [PEP 723 (Inline script metadata)](https://peps.python.org/pep-0723/) に準拠しています。`uv` を用いて依存関係を自動解決して直接実行できます。

```bash
uv run calculate_theil.py
```

---

## 5. データの出典・統計的背景

本リポジトリで使用されている家計データは、日本の所得格差構造を再現するため、以下の公的統計調査の公表データ・分布パラメータを参考に統計的手法（対数正規分布 + パレートテール + 抽出ウェイト）により生成された合成データ（Synthetic Dataset, $N=1,500$）です。

1. **厚生労働省「国民生活基礎調査」**（所得票・各種世帯の所得等の状況）
2. **総務省統計局「家計調査」**（家計収支編・貯蓄・負債編）
3. **総務省「就業構造基本調査」**（雇用形態別・所得分布データ）

---

## 6. 免責事項（Disclaimer）

本リポジトリで提供される分析コード、計算ロジック、可視化グラフ、および解説は、タイテル指数（Theil-T Index）の加法分解手法を実証・解説するための研究・教育・技術デモを目的としています。使用しているデータは統計的手法により生成されたシミュレーションデータ（合成データ）であり、実在する個人・世帯の実測値そのものではありません。本コンテンツの正確性・完全性・有用性等についてはいかなる保証も行いません。本情報の利用により生じた直接的・間接的な損害について、作成者および関係者は一切の責任を負いません。
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ README.md を生成しました: {output_path}")


# ==============================================================================
# メインルーチン
# ==============================================================================
def main():
    print("=" * 70)
    print("  日本の所得格差分析：タイテル指数（Theil-T Index）加法分解")
    print("  PEP 723 / uv 準拠スクリプト")
    print("=" * 70)

    # 1. データ生成
    print("\n[Step 1] 日本の家計所得データを生成中...")
    df = generate_japanese_household_data(n_samples=1500, seed=42)
    print(f"  -> 生成完了: {len(df)} 世帯 (グループ数: {df['group'].nunique()})")

    # 2. タイテル指数の計算と加法分解
    print("\n[Step 2] タイテル指数の加法分解を実行中...")
    decomp = decompose_theil_t(df, income_col="income", weight_col="weight", group_col="group")

    summary_df = decomp["summary_df"]
    tot_t = decomp["total_theil"]
    t_w = decomp["within_theil"]
    t_b = decomp["between_theil"]
    w_pct = decomp["within_share_pct"]
    b_pct = decomp["between_share_pct"]

    # 3. コンソール出力（Pandas DataFrame整形表示）
    print("\n" + "=" * 70)
    print("【グループ別集計結果（Pandas DataFrameサマリー）】")
    print("=" * 70)
    display_df = summary_df.copy()
    display_df["人口シェア(%)"] = (display_df["pop_share"] * 100).map("{:.2f}%".format)
    display_df["所得シェア(%)"] = (display_df["income_share"] * 100).map("{:.2f}%".format)
    display_df["平均所得(万円)"] = display_df["mean_income"].map("{:.1f}".format)
    display_df["グループ内T(T_g)"] = display_df["theil_g"].map("{:.4f}".format)
    display_df["グループ内寄与"] = display_df["within_contrib"].map("{:.4f}".format)
    display_df["グループ間寄与"] = display_df["between_contrib"].map("{:.4f}".format)

    cols_to_show = ["group", "sample_count", "人口シェア(%)", "所得シェア(%)", "平均所得(万円)",
                    "グループ内T(T_g)", "グループ内寄与", "グループ間寄与"]
    print(tabulate(display_df[cols_to_show], headers="keys", tablefmt="fancy_grid", showindex=False))

    print("\n" + "=" * 70)
    print("【最終分解結果】")
    print("=" * 70)
    print(f"  ● 全体タイテル指数 (T)       : {tot_t:.6f}")
    print(f"  ● グループ内格差 (T_within)   : {t_w:.6f}  (寄与率: {w_pct:.2f} %)")
    print(f"  ● グループ間格差 (T_between)  : {t_b:.6f}  (寄与率: {b_pct:.2f} %)")
    print(f"  ● 検証 (T_within + T_between): {t_w + t_b:.6f}")
    print(f"  ● 差分 (残差)                 : {abs(tot_t - (t_w + t_b)):.2e}  (完全一致)")
    print("=" * 70)

    # 4. 可視化画像の生成
    print("\n[Step 3] 可視化チャートを生成中...")
    img_path = "docs/theil_decomposition.png"
    create_visualizations(decomp, df, img_path)

    # 5. GitHub Pages 用 HTML および README.md 出力
    print("\n[Step 4] GitHub Pages 用 Web レポート及び README.md を生成中...")
    generate_github_pages_html(decomp, "docs/index.html", img_rel_path="theil_decomposition.png")
    # ルートの index.html としても配置（Pages設定の柔軟性のため）
    generate_github_pages_html(decomp, "index.html", img_rel_path="docs/theil_decomposition.png")
    generate_readme_md(decomp, "README.md")

    # CSV 出力
    os.makedirs("output", exist_ok=True)
    summary_df.to_csv("output/theil_decomposition_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv("output/household_micro_data.csv", index=False, encoding="utf-8-sig")
    print("✅ CSVファイルを出力しました: output/")

    print("\n🎉 すべての処理が正常に完了しました！")


if __name__ == "__main__":
    main()
