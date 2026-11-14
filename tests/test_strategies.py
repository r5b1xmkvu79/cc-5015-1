"""Tests for the strategy experiment, invariants, and production junction.

Run:  python3 -m unittest tests.test_strategies -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh_repair.benchmarks import generators as gen
from mesh_repair.core import production, repair, strategies
from mesh_repair.core.invariants import (
    attribute_delta,
    face_key_multiset,
    invariant_delta,
    measure,
    mesh_hash,
)
from mesh_repair.core.repair import RepairSession
from mesh_repair.core.topology import analyze


def analyze_mesh(mesh, unit="mm", tol=0.01, degen=0.001):
    return analyze(mesh, unit=unit, dup_tolerance=tol, degenerate_edge=degen)


class TestInvariants(unittest.TestCase):
    def test_closed_box_invariants(self):
        mesh, _ = gen.build("duplicate_faces")
        # strip the duplicates to get a clean closed box first
        sess = RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
        inv = measure(sess.mesh)
        self.assertEqual(inv.components, 1)
        self.assertEqual(inv.euler_characteristic, 2)
        self.assertEqual(inv.boundary_edges, 0)
        self.assertEqual(inv.nonmanifold_edges, 0)
        self.assertTrue(inv.orientable)
        self.assertEqual(inv.per_component[0]["genus"], 0)
        self.assertGreater(inv.signed_volume, 0)

    def test_genus_one_torus(self):
        # a coarse torus: 9 rings x 6 segments, closed, genus 1
        rings, segs = 9, 8
        verts = []
        for i in range(rings):
            a = 2 * np.pi * i / rings
            for j in range(segs):
                b = 2 * np.pi * j / segs
                r = 3 + np.cos(b)
                verts.append((r * np.cos(a), r * np.sin(a), np.sin(b)))
        idx = lambda i, j: (i % rings) * segs + (j % segs)
        faces = []
        for i in range(rings):
            for j in range(segs):
                a, b = idx(i, j), idx(i + 1, j)
                c, d = idx(i + 1, j + 1), idx(i, j + 1)
                faces.append((a, b, c))
                faces.append((a, c, d))
        from mesh_repair.core.mesh import Mesh

        inv = measure(Mesh(np.array(verts), np.array(faces, dtype=np.int64)))
        self.assertEqual(inv.boundary_edges, 0)
        self.assertEqual(inv.per_component[0]["genus"], 1)
        self.assertEqual(inv.euler_characteristic, 0)
        self.assertEqual(inv.total_genus, 1)

    def test_attribute_delta_counts_preserved_by_coordinate_key(self):
        mesh, meta = gen.multi_hole()
        sess = RepairSession(mesh, unit="mm")
        after_vertices = np.vstack([mesh.vertices, [[0.0, 0.0, 0.0]]])
        remapped = mesh.faces.copy()
        # reorder vertices: swap rows 0 and 1 and remap faces
        swapped = mesh.vertices.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        idmap = np.arange(mesh.vertex_count)
        idmap[0], idmap[1] = 1, 0
        delta = attribute_delta(mesh, type(mesh)(swapped, idmap[mesh.faces], mesh.name))
        self.assertEqual(delta["removed_face_keys"], 0)
        self.assertEqual(delta["added_face_keys"], 0)
        self.assertEqual(delta["preserved_face_keys"], mesh.face_count)


class TestMultiHoleRoutes(unittest.TestCase):
    def test_route_a_fills_only_repairable_holes(self):
        mesh, meta = gen.multi_hole()
        result, sess = strategies.route_local_connections(mesh, unit="mm")
        d = result.to_dict()
        applied = sum(len(s["records"]) for s in d["steps"])
        self.assertGreaterEqual(applied, 2)
        # faces only increase; nothing removed
        self.assertGreaterEqual(d["invariant_delta"]["faces"]["delta"], 0)
        self.assertEqual(d["attributes"]["removed_face_keys"], 0)
        # the over-limit holes are explicitly refused
        self.assertGreaterEqual(len(d["refused"]), 1)
        self.assertTrue(any("超过上限" in r for r in d["refused"]))

    def test_route_b_is_transactional_on_over_limit_hole(self):
        mesh, _ = gen.multi_hole()
        result, sess = strategies.route_topology_first(mesh, unit="mm")
        d = result.to_dict()
        self.assertFalse(d["mesh_changed"])
        self.assertEqual(d["hash_before"], d["hash_after"])
        self.assertIsNone(sess)
        self.assertEqual(d["attributes"]["added_face_keys"], 0)
        self.assertTrue(any("整体放弃" in s["detail"] for s in d["steps"]))

    def test_route_b_applies_when_all_loops_fit(self):
        # A finely tessellated plate with one small (perimeter 8) opening;
        # every loop is within the default limits, so the transactional route
        # has nothing to refuse and must apply.
        from mesh_repair.core.mesh import Mesh

        p, f = gen._grid_plate(0, 0, 40, 40, nx=20, ny=20)
        solid = gen._sheet_to_solid(p, f, 0.0, 2.0, "one_hole")
        # remove the two top triangles of the cell at (10, 10)
        n = 21
        idx = lambda i, j: j * n + i
        cell = {(idx(10, 10), idx(11, 10), idx(11, 11)),
                (idx(10, 10), idx(11, 11), idx(10, 11))}
        drop = [k for k, tri in enumerate(solid.faces)
                if tuple(sorted(int(v) for v in tri)) in
                {tuple(sorted(c)) for c in cell}]
        out = Mesh(solid.vertices, np.delete(solid.faces, drop, axis=0), "one_hole")
        res0 = analyze_mesh(out)
        self.assertIn("open_boundaries", res0.groups)
        result, sess = strategies.route_topology_first(out, unit="mm")
        d = result.to_dict()
        self.assertTrue(d["mesh_changed"])
        self.assertGreater(d["invariant_delta"]["faces"]["delta"], 0)
        self.assertEqual(d["attributes"]["removed_face_keys"], 0)


class TestNonmanifoldJunction(unittest.TestCase):
    def test_benchmark_reports_one_nonmanifold_edge(self):
        mesh, meta = gen.nonmanifold_junction()
        res = analyze_mesh(mesh)
        self.assertEqual(res.metrics["nonmanifold_edges"], 1)
        self.assertEqual(res.metrics["components"], 1)

    def test_junction_split_preserves_faces_area_volume(self):
        mesh, _ = gen.nonmanifold_junction()
        sess = RepairSession(mesh, unit="mm")
        recs = repair.repair_nonmanifold_junctions(sess, reason="test")
        self.assertTrue(recs)
        before = measure(mesh)
        after = measure(sess.mesh)
        self.assertEqual(after.faces, before.faces)
        self.assertAlmostEqual(after.area, before.area, places=9)
        self.assertAlmostEqual(after.signed_volume, before.signed_volume, places=9)
        self.assertEqual(after.nonmanifold_edges, 0)
        self.assertEqual(after.nonmanifold_vertices, 0)
        self.assertEqual(after.components, 2)
        self.assertEqual(before.components, 1)
        # each box survives as a closed genus-0 shell
        for comp in after.per_component:
            self.assertTrue(comp["closed"])
            self.assertEqual(comp["genus"], 0)
            self.assertAlmostEqual(comp["signed_volume"], 1000.0, places=6)

    def test_edge_only_split_is_insufficient(self):
        """Detaching faces without splitting the vertex fans leaves one
        component: this documents *why* both operations are needed."""
        mesh, _ = gen.nonmanifold_junction()
        sess = RepairSession(mesh, unit="mm")
        repair.split_all_nonmanifold_edges(sess, reason="edge only")
        inv = measure(sess.mesh)
        self.assertEqual(inv.nonmanifold_edges, 0)
        # components did not separate because endpoints are still shared
        self.assertEqual(inv.components, 1)
        self.assertGreater(inv.boundary_edges, 0)

    def test_undo_restores_exact_junction(self):
        mesh, _ = gen.nonmanifold_junction()
        sess = RepairSession(mesh, unit="mm")
        h0 = mesh_hash(sess.mesh)
        repair.repair_nonmanifold_junctions(sess, reason="test")
        self.assertNotEqual(mesh_hash(sess.mesh), h0)
        while sess.can_undo:
            sess.undo()
        self.assertEqual(mesh_hash(sess.mesh), h0)

    def test_both_routes_agree_on_junction(self):
        mesh, _ = gen.nonmanifold_junction()
        a, sa = strategies.route_local_connections(mesh, unit="mm")
        b, sb = strategies.route_topology_first(mesh, unit="mm")
        self.assertIsNotNone(sa)
        self.assertIsNotNone(sb)
        for res in (a, b):
            d = res.to_dict()
            self.assertEqual(d["attributes"]["removed_face_keys"], 0)
            self.assertEqual(d["invariant_delta"]["components"]["delta"], 1)
            self.assertEqual(d["invariant_delta"]["nonmanifold_edges"]["after"], 0)


class TestProductionJunction(unittest.TestCase):
    def test_pick_bad_faces_and_repair(self):
        mesh, _ = gen.nonmanifold_junction()
        sess = RepairSession(mesh, unit="mm")
        result = analyze_mesh(sess.mesh)
        non_manifold = result.groups["nonmanifold_edges"].instances[0]
        victim_faces = non_manifold["faces"]
        outcome = production.apply_from_selection(
            sess, result, faces=victim_faces, reason="ui test"
        )
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.kind, "nonmanifold_edges")
        self.assertTrue(outcome.mesh_changed)
        self.assertEqual(outcome.attributes["removed_face_keys"], 0)
        stuff = _dict(outcome)
        self.assertEqual(stuff["delta"]["components"]["delta"], 1)
        self.assertEqual(stuff["delta"]["faces"]["delta"], 0)
        # focus geometry points at real faces in the live mesh
        self.assertTrue(all(f < sess.mesh.face_count for f in outcome.focus_faces))

    def test_unknown_face_selection_is_refused_without_change(self):
        mesh, _ = gen.nonmanifold_junction()
        sess = RepairSession(mesh, unit="mm")
        result = analyze_mesh(sess.mesh)
        h = mesh_hash(sess.mesh)
        with self.assertRaises(repair.RepairError):
            production.apply_from_selection(
                sess, result, faces=[0], reason="not a defect"
            )
        self.assertEqual(mesh_hash(sess.mesh), h)
        self.assertEqual(len(sess.history), 0)

    def test_hole_fill_refusal_keeps_mesh(self):
        mesh, _ = gen.multi_hole()
        sess = RepairSession(mesh, unit="mm")
        result = analyze_mesh(sess.mesh)
        big_idx = max(
            range(len(result.groups["open_boundaries"].instances)),
            key=lambda i: result.groups["open_boundaries"].instances[i]["perimeter"],
        )
        h = mesh_hash(sess.mesh)
        with self.assertRaises(repair.RepairError):
            production.apply_to_instance(sess, "open_boundaries", big_idx,
                                         reason="超限必须拒绝")
        self.assertEqual(mesh_hash(sess.mesh), h)
        self.assertEqual(len(sess.history), 0)

    def test_outcome_serialises_for_audit(self):
        mesh, _ = gen.nonmanifold_junction()
        sess = RepairSession(mesh, unit="mm")
        result = analyze_mesh(sess.mesh)
        outcome = production.apply_to_instance(sess, "nonmanifold_edges", 0)
        payload = outcome.to_dict()
        import json

        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("attributes", text)
        for key in ("before", "after", "delta", "attributes", "hash_before"):
            self.assertIn(key, payload)
        self.assertEqual(payload["delta"]["nonmanifold_edges"]["after"], 0)


def _dict(outcome):
    return outcome.to_dict()


class TestExperimentArtifacts(unittest.TestCase):
    """The experiment script's data must match a fresh recomputation."""

    def test_script_expectations_pass_and_files_exist(self):
        import json
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "strategy_experiment.py")],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        data = json.loads(
            (ROOT / "artifacts" / "strategy_experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(data["benchmarks"] and
                             [b["benchmark"] for b in data["benchmarks"]]),
                         {"multi_hole", "nonmanifold_junction"})
        md = (ROOT / "artifacts" / "strategy_experiment.md").read_text(encoding="utf-8")
        self.assertIn("复算方式", md)
        self.assertIn("处理边界", md)

    def test_recomputation_consistency(self):
        """Delta fields must equal after-minus-before from the same run."""
        from mesh_repair.core import strategies as st

        for name in ("multi_hole", "nonmanifold_junction"):
            mesh, _ = gen.build(name)
            for route, runner in st.ROUTES.items():
                result, _session = runner(mesh, unit="mm")
                d = result.to_dict()
                attrs = d["attributes"]
                delta = d["invariant_delta"]
                self.assertEqual(
                    delta["faces"]["delta"], attrs["faces_after"] - attrs["faces_before"]
                )
                self.assertEqual(
                    delta["nonmanifold_edges"]["delta"],
                    d["invariants_after"]["nonmanifold_edges"]
                    - d["invariants_before"]["nonmanifold_edges"],
                )
                self.assertEqual(
                    delta["components"]["delta"],
                    attrs["components_after"] - attrs["components_before"],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
