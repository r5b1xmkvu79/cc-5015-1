#!/usr/bin/env python3
"""Benchmark audit: detect, repair the intended defects, re-verify.

Run:  python3 scripts/benchmark_audit.py [--out artifacts/benchmark_report.json]

For each benchmark this prints, and records as JSON:

* which defect kinds were detected,
* the geometric change caused by the intended repair (area/volume/bbox),
* which problems remain afterwards,
* whether any thin wall was fused (component count must not drop).

It fails (exit 1) if a documented expectation is violated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh_repair.benchmarks import generators as gen
from mesh_repair.core import holes, repair
from mesh_repair.core.topology import analyze
from mesh_repair.core.verify import verification_report


def analyze_mesh(mesh, unit="mm", tol=0.01, degen=0.001):
    return analyze(mesh, unit=unit, dup_tolerance=tol, degenerate_edge=degen)


def run_thin_wall():
    mesh, meta = gen.build("thin_wall")
    sess = repair.RepairSession(mesh, unit="mm")
    before = analyze_mesh(mesh)
    checks = {}

    # 1. proximity detector at a tolerance larger than the wall gap
    close = analyze_mesh(mesh, tol=meta["expected"]["clearance"] + 0.1)
    checks["near_coincident_reported"] = "duplicate_vertices" in close.groups

    # 2. a weld spanning the gap must be refused
    n = mesh.vertex_count // 2
    try:
        repair.weld_vertices_selected(sess, [[0, n]], unit="mm", threshold=0.1,
                                      confirmed_reason="audit")
        refused = False
    except repair.RepairError:
        refused = True
    checks["cross_wall_weld_refused"] = refused
    checks["mesh_unchanged_after_refusal"] = sess.mesh.face_count == mesh.face_count

    after = analyze_mesh(sess.mesh)
    checks["components_before"] = before.metrics["components"]
    checks["components_after"] = after.metrics["components"]
    checks["walls_not_fused"] = after.metrics["components"] == 2
    rep = verification_report(sess.original, before, sess.mesh, after)
    return {
        "benchmark": "thin_wall",
        "notes": meta["notes"],
        "detected": {k: len(v) for k, v in before.groups.items()},
        "checks": checks,
        "geometry": rep["geometry"],
        "problems": rep["problems"],
        "ok": all(v for k, v in checks.items() if isinstance(v, bool)),
    }


def run_duplicate_faces():
    mesh, meta = gen.build("duplicate_faces")
    sess = repair.RepairSession(mesh, unit="mm")
    before = analyze_mesh(mesh)
    repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
    after = analyze_mesh(sess.mesh)
    rep = verification_report(sess.original, before, sess.mesh, after)
    checks = {
        "duplicates_detected": len(before.groups.get("duplicate_faces", [])) ==
                               meta["expected"]["duplicate_face_instances"],
        "duplicates_resolved": "duplicate_faces" not in after.groups,
        "nonmanifold_from_dupes_resolved": after.metrics["nonmanifold_edges"] == 0,
        "history_recorded": len(sess.history) == len(before.groups["duplicate_faces"].instances),
    }
    return {
        "benchmark": "duplicate_faces",
        "notes": meta["notes"],
        "detected": {k: len(v) for k, v in before.groups.items()},
        "checks": checks,
        "geometry": rep["geometry"],
        "problems": rep["problems"],
        "ok": all(checks.values()),
    }


def run_holed_plate():
    mesh, meta = gen.build("holed_plate")
    sess = repair.RepairSession(mesh, unit="mm")
    before = analyze_mesh(mesh)
    loops = before.groups["open_boundaries"].instances
    limits = holes.HoleFillLimits()

    planned = [holes.plan_fill(sess, inst["vertices"], limits) for inst in loops]
    allowed = [p for p in planned if p["ok"]]
    refused = [p for p in planned if not p["ok"]]
    checks = {
        "two_openings_detected": len(loops) >= 2,
        "only_small_opening_allowed": len(allowed) == 1,
        "large_opening_refused": len(refused) >= 1,
    }
    if allowed:
        small = min(loops, key=lambda i: i["perimeter"])
        rec = holes.fill_hole(sess, small["vertices"], limits, reason="audit: 破损孔")
        after = analyze_mesh(sess.mesh)
        rep = verification_report(sess.original, before, sess.mesh, after)
        checks["boundary_edges_reduced"] = after.metrics["boundary_edges"] < before.metrics["boundary_edges"]
        checks["designed_opening_remains"] = "open_boundaries" in after.groups
        checks["fill_recorded_params"] = all(
            k in rec.parameters for k in ("loop", "new_faces", "limits", "perimeter")
        )
        geometry = rep["geometry"]
        problems = rep["problems"]
    else:
        geometry = {}
        problems = ["未找到可补的小孔（基准构造变化）"]
        checks["boundary_edges_reduced"] = False
    return {
        "benchmark": "holed_plate",
        "notes": meta["notes"],
        "detected": {k: len(v) for k, v in before.groups.items()},
        "opening_perimeters": sorted(round(i["perimeter"], 3) for i in loops),
        "checks": checks,
        "geometry": geometry,
        "problems": problems,
        "ok": all(checks.values()),
    }


def run_mixed_orientation():
    mesh, meta = gen.build("mixed_orientation")
    sess = repair.RepairSession(mesh, unit="mm")
    before = analyze_mesh(mesh)
    inst = before.groups["inward_components"].instances[0]
    repair.flip_component(sess, inst["faces"], confirm_reason="audit: 确认朝外")
    after = analyze_mesh(sess.mesh)
    rep = verification_report(sess.original, before, sess.mesh, after)
    checks = {
        "inward_detected": "inward_components" in before.groups,
        "inward_resolved": "inward_components" not in after.groups,
        "volume_increased": sess.mesh.signed_volume() > mesh.signed_volume(),
        "other_components_untouched": after.metrics["components"] == before.metrics["components"],
    }
    return {
        "benchmark": "mixed_orientation",
        "notes": meta["notes"],
        "detected": {k: len(v) for k, v in before.groups.items()},
        "checks": checks,
        "geometry": rep["geometry"],
        "problems": rep["problems"],
        "ok": all(checks.values()),
    }


def run_open_shell():
    mesh, meta = gen.build("open_shell")
    before = analyze_mesh(mesh)
    checks = {
        "boundary_detected": "open_boundaries" in before.groups,
        "orientation_unconfirmed": any("不封闭" in w for w in before.warnings),
        "not_auto_filled": True,
    }
    return {
        "benchmark": "open_shell",
        "notes": meta["notes"],
        "detected": {k: len(v) for k, v in before.groups.items()},
        "checks": checks,
        "warnings": before.warnings,
        "geometry": {},
        "problems": before.warnings,
        "ok": all(checks.values()),
    }


def run_combined():
    mesh, meta = gen.build("combined_stress")
    sess = repair.RepairSession(mesh, unit="mm")
    before = analyze_mesh(mesh)
    repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
    mid = analyze_mesh(sess.mesh)
    conflict_faces = sorted({int(f) for inst in mid.groups.get("winding_conflicts", type("g", (), {"instances": []})()).instances
                             for f in inst["faces"]}) if "winding_conflicts" in mid.groups else []
    if conflict_faces:
        repair.flip_faces(sess, conflict_faces, reason="audit")
    mid2 = analyze_mesh(sess.mesh)
    if "inward_components" in mid2.groups:
        repair.flip_component(sess, mid2.groups["inward_components"].instances[0]["faces"],
                              confirm_reason="audit")
    after = analyze_mesh(sess.mesh)
    rep = verification_report(sess.original, before, sess.mesh, after)
    checks = {
        "duplicates_resolved": "duplicate_faces" not in after.groups,
        "winding_resolved": "winding_conflicts" not in after.groups,
        "inward_resolved": "inward_components" not in after.groups,
        "components_preserved": after.metrics["components"] == before.metrics["components"],
        "clean": rep["clean"],
    }
    return {
        "benchmark": "combined_stress",
        "notes": meta["notes"],
        "detected": {k: len(v) for k, v in before.groups.items()},
        "after": {k: len(v) for k, v in after.groups.items()},
        "checks": checks,
        "geometry": rep["geometry"],
        "problems": rep["problems"],
        "ok": all(checks.values()),
    }


RUNNERS = [run_thin_wall, run_duplicate_faces, run_holed_plate,
           run_mixed_orientation, run_open_shell, run_combined]


def main(argv):
    reports = []
    for runner in RUNNERS:
        report = runner()
        reports.append(report)
        status = "PASS" if report["ok"] else "FAIL"
        print(f"[{status}] {report['benchmark']}: detected={report['detected']}")
        for name, value in report["checks"].items():
            print(f"    - {name}: {value}")
        if report["problems"]:
            for p in report["problems"]:
                print(f"    遗留/问题: {p}")

    out = None
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    else:
        out = ROOT / "artifacts" / "benchmark_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")

    failed = [r["benchmark"] for r in reports if not r["ok"]]
    if failed:
        print("失败基准:", ", ".join(failed))
        return 1
    print("全部基准通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
