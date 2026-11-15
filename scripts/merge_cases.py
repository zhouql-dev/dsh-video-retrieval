#!/usr/bin/env python3
"""合并多份 cases.jsonl 到唯一状态根（P0-2）。

用法：
  python3 scripts/merge_cases.py --sources A B C --out PATH [--dry-run]

规则（用户 2026-11-15 定：不一致以最新文件为准）：
- 以 id 去重；无 id 的用 (query, ts) 兜底
- 同一 id 冲突时：优先 human_confirmed=true；其次 ts 更新者胜
- 输出按 ts 升序，保留原始字段
"""
import argparse, json, os, sys

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                print(f"  ! {path}:{i} 解析失败 {e}", file=sys.stderr)
                continue
            if not isinstance(r, dict):
                continue
            r["__src"] = path
            rows.append(r)
    return rows

def key(r):
    return r.get("id") or f"{r.get('query','')}|{r.get('ts','')}"

def better(a, b):
    ca, cb = bool(a.get("human_confirmed")), bool(b.get("human_confirmed"))
    if ca != cb:
        return a if ca else b
    return a if str(a.get("ts", "")) >= str(b.get("ts", "")) else b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    merged, per_src = {}, {}
    for p in a.sources:
        if not os.path.exists(p):
            print(f"  - 跳过（不存在）{p}")
            continue
        rows = load(p)
        per_src[p] = len(rows)
        print(f"  + {len(rows):3d} 行  {p}")
        for r in rows:
            k = key(r)
            merged[k] = better(merged[k], r) if k in merged else r

    total_in = sum(per_src.values())
    print(f"\n合并：输入 {total_in} 行 / 去重后 {len(merged)} 行（重复 {total_in - len(merged)}）")
    conf = sum(1 for r in merged.values() if r.get("human_confirmed"))
    print(f"      其中 human_confirmed=true: {conf}")
    print(f"      每个源里的 id 是否都在结果里: ", end="")
    all_ids = {}
    for p in per_src:
        for r in load(p):
            all_ids.setdefault(key(r), set()).add(p)
    missing = [k for k in all_ids if k not in merged]
    print("是" if not missing else f"否（缺 {len(missing)}）")

    if a.dry_run:
        print("\n[dry-run] 未写文件")
        return 0
    out = a.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for k in sorted(merged, key=lambda x: str(merged[x].get("ts", ""))):
            r = dict(merged[k]); r.pop("__src", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(f"\n写入 {out}（{len(merged)} 行）")
    return 0

if __name__ == "__main__":
    sys.exit(main())
