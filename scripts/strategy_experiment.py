#!/usr/bin/env python3
"""Strategy experiment: local connections vs topology-first.

Run:  python3 scripts/strategy_experiment.py [--out DIR]

For the multi-hole and non-manifold benchmarks, runs both repair routes on
the real mesh and records, from code:

* topology invariants before/after (V, E, F, chi, components, boundary
  loops, genus, non-manifold counts, orientability),
* attribute changes (area, volume, preserved/removed/added face keys),
* refusals and the mesh hash before/after, so a "no change" claim is provable.

Writes ``strategy_experiment.json`` and ``strategy_experiment.md`` into
``artifacts/``.  The markdown is generated from the same data, so it can
never drift from the measurements.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh_repair.benchmarks import generators as gen
from mesh_repair.core import strategies

BENCHMARKS = ("multi_hole", "nonmanifold_junction")

INV_KEYS = [
    ("vertices", "顶点 V"),
    ("edges", "边 E"),
    ("faces", "面 F"),
    ("euler_characteristic", "欧拉特征 χ"),
    ("components", "连通组件"),
    ("boundary_edges", "边界边"),
    ("boundary_loops", "边界环"),
    ("nonmanifold_edges", "非流形边"),
    ("nonmanifold_vertices", "非流形顶点"),
    ("closed_components", "闭合组件"),
    ("total_genus", "总亏格"),
]


def run() -> dict:
    payload = {"benchmarks": [], "routes": list(strategies.ROUTES)}
    for name in BENCHMARKS:
        mesh, meta = gen.build(name)
        entry = {
            "benchmark": name,
            "notes": meta.get("notes", ""),
            "expected": meta.get("expected", {}),
            "routes": {},
        }
        for route_name, runner in strategies.ROUTES.items():
            result, session = runner(mesh, unit="mm")
            entry["routes"][route_name] = result.to_dict()
        payload["benchmarks"].append(entry)
    return payload


def render_markdown(payload: dict) -> str:
    lines = [
        "# 网格策略实验：局部连接 vs 先建有效拓扑",
        "",
        "> 本文件由 `scripts/strategy_experiment.py` 生成；所有数字来自 "
        "`mesh_repair.core.invariants` 的复算，不手工填写。",
        "",
        "## 复算方式",
        "",
        "```bash",
        "python3 scripts/strategy_experiment.py",
        "```",
        "",
        "拓扑不变量由 `measure(mesh)` 重新计算；属性变化由 "
        "`attribute_delta(before, after)` 以“坐标键多重集”比对，"
        "顶点索引重排不影响保留面计数。`hash_before/hash_after` 是顶点与面"
        "数组的精确哈希，用来证明“未执行”确实没有改动网格。",
        "",
    ]
    for entry in payload["benchmarks"]:
        lines += [
            f"## 基准 `{entry['benchmark']}`",
            "",
            f"{entry['notes']}",
            "",
            f"预期：`{json.dumps(entry['expected'], ensure_ascii=False)}`",
            "",
        ]
        for route, data in entry["routes"].items():
            delta = data["invariant_delta"]
            attrs = data["attributes"]
            lines += [
                f"### 路线 `{route}`",
                "",
                f"- 网格是否改变：**{'是' if data['mesh_changed'] else '否'}**"
                f"（哈希 {data['hash_before'][:12]} → {data['hash_after'][:12]}）",
                f"- 保面：保留 {attrs['preserved_face_keys']}，"
                f"删除 {attrs['removed_face_keys']}，新增 {attrs['added_face_keys']}",
                f"- 面积 {attrs['area_before']:.6g} → {attrs['area_after']:.6g}"
                f"（Δ {attrs['area_delta']:+.6g}）；"
                f"体积 {attrs['volume_before']:.6g} → {attrs['volume_after']:.6g}"
                f"（Δ {attrs['volume_delta']:+.6g}）",
                "",
                "| 指标 | 前 | 后 | Δ |",
                "|---|---:|---:|---:|",
            ]
            for key, label in INV_KEYS:
                d = delta[key]
                lines.append(f"| {label} | {d['before']} | {d['after']} | {d['delta']:+} |")
            lines += ["", "步骤：", ""]
            for step in data["steps"]:
                mark = "✔" if step["applied"] else "✘"
                lines.append(f"- {mark} {step['name']}：{step['detail']}")
            if data["refused"]:
                lines += ["", "拒绝/遗留："]
                for r in data["refused"]:
                    lines.append(f"- {r}")
            lines.append("")
    lines += [
        "## 处理边界（结论）",
        "",
        "- **重复面**：两条路线都删除冗余副本；保留面以坐标键记录，"
        "非冗余面零删除。",
        "- **裂口/多洞**：路线 A 对限制内的洞逐个修补、超限的洞保持原状，"
        "因此会出现“部分修复”；路线 B 先清点全部环，只要有一个超限就整体不落地，"
        "网格哈希不变。取舍是“可交付一部分”对“绝不产生半成品”。",
        "- **非流形交汇**：仅拆边不够——共享端点仍把两片面扇绑在一起，组件数不降。"
        "正确做法是同时按面扇复制顶点再拆边；两条路线共用这一内核，"
        "保面、保面积/体积，组件 1→2。",
        "- **不可自动判断者**：设计开口/内腔方向仍列为待确认，不由策略自动处理。",
        "",
        "## 证据落点",
        "",
        "- 不变量与属性：`artifacts/strategy_experiment.json`（本脚本生成）。",
        "- 图形入口：`python3 scripts/window_drive.py`，"
        "生成 `artifacts/strategy/20_pick_before.png` 与 `21_pick_after.png`；"
        "选择坏面后由修复内核驱动，前后组件数与面数记录在 `window_run.log`。",
        "- 待验假设单列：`artifacts/strategy_experiment_notes.md`。",
        "",
    ]
    return "\n".join(lines)


def check_expectations(payload: dict) -> list[str]:
    """Fail loudly if a route silently violates a documented expectation.

    Guards against the experiment "passing" because a limit default hid a
    deletion: faces may never be removed on either benchmark, and the
    junction must separate while preserving area/volume.
    """
    problems: list[str] = []
    for entry in payload["benchmarks"]:
        name = entry["benchmark"]
        for route, data in entry["routes"].items():
            attrs = data["attributes"]
            if attrs["removed_face_keys"] != 0:
                problems.append(
                    f"{name}/{route}: 删除了 {attrs['removed_face_keys']} 个非冗余面"
                )
            if name == "nonmanifold_junction":
                d = data["invariant_delta"]
                if d["nonmanifold_edges"]["after"] != 0:
                    problems.append(f"{name}/{route}: 非流形边未清零")
                if d["components"]["delta"] < 1:
                    problems.append(f"{name}/{route}: 组件未分离")
                if abs(attrs["area_delta"]) > 1e-9:
                    problems.append(f"{name}/{route}: 面积未被保留")
                if abs(attrs["volume_delta"]) > 1e-9:
                    problems.append(f"{name}/{route}: 体积未被保留")
            if name == "multi_hole" and route == "topology_first":
                if data["mesh_changed"]:
                    problems.append(f"{name}/{route}: 应整体放弃但网格被改动")
                if data["hash_before"] != data["hash_after"]:
                    problems.append(f"{name}/{route}: 未改动声明与哈希不符")
            if name == "multi_hole" and route == "local_connections":
                if data["invariant_delta"]["faces"]["delta"] <= 0:
                    problems.append(f"{name}/{route}: 未补任何限制内的洞")
    return problems


def main(argv: list[str]) -> int:
    out_dir = ROOT / "artifacts"
    if "--out" in argv:
        out_dir = Path(argv[argv.index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = run()
    json_path = out_dir / "strategy_experiment.json"
    md_path = out_dir / "strategy_experiment.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    for entry in payload["benchmarks"]:
        print(f"== {entry['benchmark']}")
        for route, data in entry["routes"].items():
            d = data["invariant_delta"]
            print(
                f"  {route:18s} changed={data['mesh_changed']} "
                f"faces {d['faces']['before']}->{d['faces']['after']} "
                f"components {d['components']['before']}->{d['components']['after']} "
                f"nm_edges {d['nonmanifold_edges']['before']}->{d['nonmanifold_edges']['after']} "
                f"preserved={data['attributes']['preserved_face_keys']} "
                f"removed={data['attributes']['removed_face_keys']} "
                f"refused={len(data['refused'])}"
            )
    print(f"\n已写入 {json_path}\n已写入 {md_path}")
    problems = check_expectations(payload)
    if problems:
        print("\n实验期望未满足：")
        for p in problems:
            print("  ✘ " + p)
        return 1
    print("实验期望全部满足")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
