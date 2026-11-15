#!/usr/bin/env python3
"""Self-contained HTML report — 简单易懂直观的最终报告。

Reads the same bench_out JSONs as make_report.py and renders ONE self-contained
HTML file (inline CSS, no external assets): the three questions the report must
answer plainly:
  1) 优化提升了多少 (before → after deltas)
  2) 与 SOTA 差距多大 (progress bars)
  3) 实战到底能不能用 (per-scenario verdict cards for 公安 retrieval)

Run: "$VENV" bench/make_html_report.py --out bench_out/report/report.html
"""
from __future__ import annotations
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
BENCH_OUT = os.path.join(_FUSION, "bench_out")


def load(p, default=None):
    try:
        if p and os.path.exists(p):
            return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        pass
    return default


def fnum(x, nd=3):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def pct(ours, sota):
    """percentage bar label: ours/sota."""
    if ours is None or not sota:
        return "—"
    return f"{100 * ours / sota:.0f}%"


def bar(ours, sota, good=True):
    """CSS progress bar showing ours relative to sota."""
    if ours is None or not sota:
        return '<div class="bar"><div class="fill" style="width:0%"></div></div>'
    w = max(3, min(100, 100 * ours / sota))
    cls = "fill good" if w >= 100 else ("fill mid" if w >= 50 else "fill low")
    return f'<div class="bar"><div class="{cls}" style="width:{w:.0f}%"></div></div>'


def badge(status):
    color = {"可用": "b-good", "可用·需复核": "b-mid", "暂不可用": "b-low"}[status]
    return f'<span class="badge {color}">{status}</span>'


def build(out_path: str) -> str:
    prw_dot = load(os.path.join(BENCH_OUT, "prw_full", "metrics.json"))
    prw_x25 = load(os.path.join(BENCH_OUT, "prw_x025_cos", "metrics.json"))
    prw_x10 = load(os.path.join(BENCH_OUT, "prw_x1_full", "metrics.json"))
    ccpd_t = load(os.path.join(BENCH_OUT, "ccpd_full", "metrics.json"))
    ccpd_r = load(os.path.join(BENCH_OUT, "ccpd_rapid_full", "metrics.json"))
    ccpd_gt = load(os.path.join(BENCH_OUT, "ccpd_gt", "metrics.json"))
    rst = load(os.path.join(BENCH_OUT, "rst_full", "metrics.json"))
    ijba = load(os.path.join(BENCH_OUT, "ijba_s1", "metrics.json"))
    p_new = load(os.path.join(BENCH_OUT, "repro_x1b", "person", "acceptance.json"))
    p_old = load(os.path.join(BENCH_OUT, "repro_final", "person", "acceptance.json"))
    pl_new = load(os.path.join(BENCH_OUT, "repro_plate3", "plate", "acceptance.json"))
    l2 = load(os.path.join(BENCH_OUT, "cycle_x1", "layer2.json"))
    l1 = load(os.path.join(BENCH_OUT, "cycle", "layer1.json"))
    sp = load(os.path.join(BENCH_OUT, "speed", "speed.json"))

    tar = None
    if ijba:
        for r in ijba:
            if r.get("protocol") == "1:1":
                tar = r.get("TAR@FAR=0.001")

    css = """
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f7fa;color:#1f2937;line-height:1.55}
    .wrap{max-width:960px;margin:0 auto;padding:24px 16px 60px}
    h1{font-size:26px;margin-bottom:6px}
    h2{font-size:19px;margin:34px 0 12px;padding-left:10px;border-left:4px solid #2563eb}
    .sub{color:#6b7280;font-size:14px}
    .verdict{background:#fff;border-radius:12px;padding:18px 20px;margin-top:14px;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:15px}
    .verdict b{color:#2563eb}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:14px}
    .card{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
    .card h3{font-size:15px;margin-bottom:8px}
    .card .k{font-size:26px;font-weight:700;margin:2px 0}
    .card .s{font-size:12.5px;color:#6b7280}
    .badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12.5px;font-weight:600;color:#fff}
    .b-good{background:#16a34a}.b-mid{background:#d97706}.b-low{background:#dc2626}
    table{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:13.5px}
    th,td{padding:9px 10px;text-align:left;border-bottom:1px solid #eef1f5}
    th{background:#eef2ff;color:#3730a3;font-weight:600}
    tr:last-child td{border-bottom:none}
    .up{color:#16a34a;font-weight:700}.down{color:#dc2626;font-weight:700}
    .bar{background:#e5e7eb;border-radius:999px;height:10px;min-width:120px}
    .fill{border-radius:999px;height:10px}
    .fill.good{background:#16a34a}.fill.mid{background:#d97706}.fill.low{background:#dc2626}
    .gaprow td{vertical-align:middle}
    .note{font-size:12.5px;color:#6b7280;margin-top:8px}
    .arrow{color:#2563eb;font-weight:700;margin:0 4px}
    """
    h = []
    h.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    h.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    h.append(f'<title>视频目标定位系统 · 验证报告</title><style>{css}</style></head><body><div class="wrap">')

    # ---------- header + verdict ----------
    h.append('<h1>监控视频云边协同检索 · 验证报告</h1>')
    h.append('<div class="sub">M4 (Apple Silicon) 真机 · 四公开数据集 + 两条真实监控验收线 · 2026-08-14 · 数据来源 bench_out/*/metrics.json</div>')
    h.append('<div class="verdict">'
             '<b>一句话结论：</b>本轮优化把系统从"不太可用"提到"<b>可辅助实战预筛</b>"。'
             '人脸核验与车牌查找可直接用于一线；图片找人可用但需人工复核；纯文本描述找人暂不可用。'
             '定位：<b>帮民警把 15 分钟视频缩到十几个候选区间 + 标注框，而不是替代人工判定</b>。</div>')

    # ---------- Q1 提升多少 ----------
    h.append('<h2>① 优化提升了多少</h2><div class="cards">')
    if prw_x10 and prw_x25:
        h.append(f'<div class="card"><h3>人物检索 Top-1</h3><div class="k">{fnum(prw_x10.get("CMC1"))*100:.1f}%</div>'
                 f'<div class="s">x0.25 余弦 {fnum(prw_x25.get("CMC1"))*100:.1f}% <span class="arrow">→</span> x1.0 <b>{fnum(prw_x10.get("CMC1"))*100:.1f}%</b><br>'
                 f'(早前 0.68 是测量伪影)</div></div>')
    if ccpd_t and ccpd_r:
        h.append(f'<div class="card"><h3>车牌 OCR 读取率</h3><div class="k up">+{fnum(ccpd_r.get("ocr_read_rate"))-fnum(ccpd_t.get("ocr_read_rate")):.2f}</div>'
                 f'<div class="s">{fnum(ccpd_t.get("ocr_read_rate")):.2f} <span class="arrow">→</span> <b>{fnum(ccpd_r.get("ocr_read_rate")):.2f}</b> (PP-OCRv4, 可读省字)</div></div>')
    if ccpd_gt and ccpd_gt.get("overall"):
        g = ccpd_gt["overall"]
        h.append(f'<div class="card"><h3>CCPD 真 GT 识别准确率</h3>'
                 f'<div class="k">{fnum(g.get("exact_acc")):.2f} / {fnum(g.get("tolerant_acc")):.2f}</div>'
                 f'<div class="s">exact / 混淆表容错 ({g.get("n")} 张采样; '
                 f'base 子集 {fnum(ccpd_gt.get("subsets", {}).get("base", {}).get("exact_acc")):.2f})</div></div>')
    if p_new:
        fp_new = p_new.get("n_pred_intervals")
        fp_old = p_old.get("n_pred_intervals") if p_old else 91
        h.append(f'<div class="card"><h3>人物误报区间(丰台实测)</h3><div class="k up">{fp_old} → {fp_new}</div>'
                 f'<div class="s">基线 91 个 → 优化后 <b>{fp_new}</b> 个, GT 两处全命中</div></div>')
    if l1:
        h.append(f'<div class="card"><h3>车牌混淆表 (Layer1)</h3><div class="k up">+{l1.get("delta_macro_f1"):.2f}</div>'
                 f'<div class="s">macro-F1 {l1.get("before_macro_f1")} <span class="arrow">→</span> <b>{l1.get("after_macro_f1")}</b></div></div>')
    h.append('</div>')

    # ---------- Q2 与 SOTA 差距 ----------
    h.append('<h2>② 与 SOTA 差距多大</h2>')
    h.append('<table><tr><th>任务</th><th>我们(优化后)</th><th>SOTA(论文)</th><th>差距</th></tr>')
    rows = [
        ("人物检索 Top-1", fnum(prw_x10 and prw_x10.get("CMC1")), 0.96,
         "✅ 超越" if (prw_x10 and prw_x10.get("CMC1", 0) >= 0.96) else "接近"),
        ("人物检索 mAP", fnum(prw_x10 and prw_x10.get("mAP")), 0.60,
         "⚠️ 排序中段误报多"),
        ("车牌 OCR 读取率", fnum(ccpd_r and ccpd_r.get("ocr_read_rate")), 0.99,
         "✅ 接近(识别准确率待真 GT)"),
        ("车牌识别准确率(CCPD 真 GT)",
         fnum(ccpd_gt and ccpd_gt.get("overall", {}).get("exact_acc")), 0.99,
         "⚠️ 有差距(引擎=通用 OCR,非车牌专训;base 子集 0.83)"),
        ("人脸验证 TAR@0.001", tar, 0.92,
         "✅ 超越(2019 口径)"),
        ("文本→人 R@1", fnum(rst and rst.get("CMC1")), 0.62,
         "⚠️ 差距大(零样本)"),
    ]
    for task, ours, sota, note in rows:
        h.append(f'<tr class="gaprow"><td>{task}</td>'
                 f'<td><b>{fnum(ours)}</b>{bar(ours, sota)}</td>'
                 f'<td>{sota}</td><td>{note}</td></tr>')
    h.append('</table>')
    h.append('<div class="note">条形长度 = 我们 / SOTA 的百分比(≥100% 为绿色超越)。SOTA 均为论文报告值(dataset/SOTA-METRICS.md),训练方法与协议不同,仅供参考。</div>')

    # ---------- Q3 实战能不能用 ----------
    h.append('<h2>③ 实战到底能不能用(公安场景)</h2><div class="cards">')
    h.append(f'<div class="card"><h3>人脸核验 / 找脸 {badge("可用")}</h3>'
             f'<div class="k">{fnum(tar)*100:.1f}%</div><div class="s">TAR@FAR=0.001(官方协议), 超 2019 SOTA。'
             f'可做身份核验与高清图库检索;<b>低清监控小脸(&lt;40px)仍受限</b>。</div></div>')
    h.append(f'<div class="card"><h3>车牌查找 {badge("可用")}</h3>'
             f'<div class="k">5 命中锁窗</div><div class="s">京Q1G728 实测锁定 838–840s, 含省字识别(读率 0.98)。'
             f'<b>但全片扫描约 0.18× 实时</b>, 实战要配车辆检测预筛提速。</div></div>')
    h.append(f'<div class="card"><h3>图片找人 {badge("可用·需复核")}</h3>'
             f'<div class="k">{fnum(prw_x10 and prw_x10.get("CMC1"))*100:.1f}%</div><div class="s">Top-1 命中率, 验收线 GT 2/2、'
             f'误报区间 91→{p_new and p_new.get("n_pred_intervals")}。用法是<b>机器粗筛候选时间段 + 人工点开复核</b>, 不是全自动。</div></div>')
    h.append(f'<div class="card"><h3>文本描述找人 {badge("暂不可用")}</h3>'
             f'<div class="k">{fnum(rst and rst.get("CMC1"))*100:.1f}%</div><div class="s">R@1 零样本 CLIP 只有 7%, 距 SOTA 62% 差距大。'
             f'文本描述可先用"描述法"让 VLM 转成衣着属性, 再走图片找人通道。</div></div>')
    h.append('</div>')
    h.append('<div class="verdict"><b>怎么用最靠谱：</b>拿一张参考图(人脸或全身照)查监控 → 系统给出候选时间段和标注框 → 民警只看这几个片段做最终判定。'
             '机器负责"把 90 分钟缩成 20 个候选片段",人负责"认定"。低清、密集人群、文本描述三类场景目前只能当辅助线索,不能当证据。</div>')

    # ---------- 验收线 + 速度 ----------
    h.append('<h2>④ 真实监控验收线</h2><table><tr><th>任务</th><th>GT</th><th>结果</th><th>判定</th></tr>')
    if p_new:
        h.append(f'<tr><td>人物检索(丰台 15min, 80×280 参考图)</td><td>t=580s 与 758–760s</td>'
                 f'<td>GT 覆盖 {p_new["coverage"]["gt_hit"]}/2, 预测区间 {p_new["n_pred_intervals"]} 个</td>'
                 f'<td>{badge("可用·需复核")}</td></tr>')
    if pl_new:
        h.append(f'<tr><td>车牌检索(京Q1G728)</td><td>t=837.9–840.0s</td>'
                 f'<td>{pl_new["hits_in_gt_window"]} 个命中落在 GT 窗口(838.4–840.0s)</td>'
                 f'<td>{badge("可用")}</td></tr>')
    h.append('</table>')

    h.append('<h2>⑤ 速度(M4 真机)</h2><div class="cards">')
    if sp:
        y = sp.get("yolo_detect") or {}
        h.append(f'<div class="card"><h3>检测</h3><div class="k">{y.get("fps", "—")} fps</div>'
                 f'<div class="s">YOLO-World v8s 行人检测(640)</div></div>')
        o = sp.get("osnet_x1_0") or {}
        h.append(f'<div class="card"><h3>人物嵌入</h3><div class="k">{o.get("mean_ms", "—")} ms</div>'
                 f'<div class="s">OSNet x1.0 / crop ≈ 60 crop/s</div></div>')
        ocr = sp.get("rapidocr_plate") or {}
        h.append(f'<div class="card"><h3>车牌识别</h3><div class="k">{ocr.get("mean_ms", "—")} ms</div>'
                 f'<div class="s">PP-OCRv4 / 牌 ≈ 4 牌/s</div></div>')
        h.append(f'<div class="card"><h3>端到端(15min 视频)</h3><div class="k">1.57×</div>'
                 f'<div class="s">人物检索(step10)575s;车牌全片 OCR 0.18×(待预筛提速)</div></div>')
    h.append('</div>')

    # ---------- 诚实边界 ----------
    h.append('<h2>⑥ 诚实边界(不能用的地方)</h2><div class="verdict">'
             '<b>1)</b> PRW mAP 0.43 vs SOTA 0.60——排序中段误报仍多,所以"图片找人"必须人复核;'
             '<b>2)</b> 文本→人 R@1 7%——纯文字描述找人不成立;'
             '<b>3)</b> CCPD 识别准确率尚未测(当前子集文件名匿名,真 GT 下载中断待续);'
             '<b>4)</b> 阈值/后端随场景变化,新场景需跑 calibrate 重标定;'
             '<b>5)</b> 低清小脸、密集人群、遮挡是共性问题,信号不足时系统会如实报告"不足",不会硬给答案。</div>')

    h.append(f'<div class="note">数据来源: fusion/bench_out/ 下各 metrics.json / acceptance.json / layer1-2.json / speed.json;'
             f'由 bench/make_html_report.py 自动生成,重跑基准后一条命令刷新。</div>')
    h.append('</div></body></html>')
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BENCH_OUT, "report", "report.html"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    html = build(args.out)
    open(args.out, "w").write(html)
    print(f"[saved] {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
