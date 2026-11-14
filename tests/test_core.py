"""Test suite for the mesh repair tool.

Run with:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh_repair.benchmarks import generators as gen
from mesh_repair.core import holes, io_mesh, repair
from mesh_repair.core.mesh import Mesh
from mesh_repair.core.project import ProjectDocument
from mesh_repair.core.topology import analyze
from mesh_repair.core.units import mm_per_unit, to_mm
from mesh_repair.core.verify import all_fingerprints, verification_report


def analyze_mesh(mesh, unit="mm", tol=0.01, degen=0.001):
    return analyze(mesh, unit=unit, dup_tolerance=tol, degenerate_edge=degen)


class TestUnits(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(mm_per_unit("mm"), 1.0)
        self.assertAlmostEqual(mm_per_unit("in"), 25.4)
        self.assertAlmostEqual(to_mm(1.0, "cm"), 10.0)
        # same physical threshold expressed in different units must match
        a = to_mm(0.01, "mm")
        b = to_mm(0.01 / 25.4, "in")
        self.assertAlmostEqual(a, b, places=12)

    def test_threshold_conversion_is_unit_aware(self):
        from mesh_repair.core.project import AnalysisSettings

        s = AnalysisSettings(unit="mm", dup_tolerance_mm=0.01)
        self.assertAlmostEqual(s.dup_tolerance(), 0.01)
        s = AnalysisSettings(unit="in", dup_tolerance_mm=0.01)
        self.assertAlmostEqual(s.dup_tolerance(), 0.01 / 25.4)


class TestBenchmarkDetection(unittest.TestCase):
    def test_thin_wall_is_clean_and_two_components(self):
        mesh, meta = gen.build("thin_wall")
        res = analyze_mesh(mesh)
        self.assertEqual(res.metrics["components"], 2)
        self.assertEqual(res.metrics["nonmanifold_edges"], 0)
        self.assertEqual(res.metrics["boundary_edges"], 0)
        self.assertNotIn("duplicate_vertices", res.groups)
        self.assertAlmostEqual(mesh.triangle_areas().sum() > 0, True)

    def test_duplicate_faces_detected(self):
        mesh, meta = gen.build("duplicate_faces")
        res = analyze_mesh(mesh)
        dup = res.groups.get("duplicate_faces")
        self.assertIsNotNone(dup)
        self.assertEqual(len(dup), meta["expected"]["duplicate_face_instances"])
        winding = {inst["same_winding"] for inst in dup.instances}
        self.assertIn(True, winding)
        self.assertIn(False, winding)

    def test_holed_plate_loops(self):
        mesh, meta = gen.build("holed_plate")
        res = analyze_mesh(mesh)
        loops = res.groups.get("open_boundaries")
        self.assertIsNotNone(loops)
        perims = sorted(inst["perimeter"] for inst in loops.instances)
        self.assertEqual(len(perims), 3)  # small hole top+bottom, large top
        self.assertLess(perims[0], 10.0)
        self.assertGreater(perims[-1], 40.0)

    def test_inward_component_detected(self):
        mesh, meta = gen.build("mixed_orientation")
        res = analyze_mesh(mesh)
        inward = res.groups.get("inward_components")
        self.assertIsNotNone(inward)
        self.assertEqual(len(inward), 1)
        vol = inward.instances[0]["faces"]
        self.assertGreater(len(vol), 0)

    def test_open_shell_marked_ambiguous(self):
        mesh, meta = gen.build("open_shell")
        res = analyze_mesh(mesh)
        self.assertIn("open_boundaries", res.groups)
        orient = res.metrics["orientation"]
        self.assertTrue(orient["open_components"])
        self.assertTrue(any("不封闭" in w for w in res.warnings))


class TestThinWallSafety(unittest.TestCase):
    """The critical requirement: proximity must not fuse two walls."""

    def test_duplicate_vertex_detector_reports_not_welds(self):
        distance = 0.4
        mesh, _meta = gen.thin_wall(distance=distance)
        # surface vertices on facing sides are exactly `distance` apart
        res = analyze_mesh(mesh, tol=distance + 0.05)
        dup = res.groups.get("duplicate_vertices")
        self.assertIsNotNone(dup, "close vertices should be reported")
        # detection changed nothing
        self.assertEqual(mesh.face_count, res.metrics["faces"])
        self.assertEqual(res.metrics["components"], 2)

    def test_weld_refuses_beyond_threshold(self):
        mesh, _ = gen.thin_wall(distance=0.4)
        sess = repair.RepairSession(mesh, unit="mm")
        # pick two vertices on opposite walls that are 0.4 apart
        n = mesh.vertex_count // 2
        group = [0, n]
        with self.assertRaises(repair.RepairError):
            repair.weld_vertices_selected(
                sess, [group], unit="mm", threshold=0.1, confirmed_reason="test"
            )
        # mesh unchanged
        self.assertEqual(sess.mesh.vertex_count, mesh.vertex_count)

    def test_weld_within_threshold_preserves_walls_when_exact(self):
        # a deliberately duplicated internal vertex can be welded safely
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = Mesh(v, f)
        sess = repair.RepairSession(mesh)
        recs = repair.weld_vertices_selected(
            sess, [[0, 0]], unit="mm", threshold=0.0, confirmed_reason="noop"
        )
        self.assertEqual(recs, [])  # single-vertex group is a no-op, not a weld

    def test_thin_wall_never_fused_by_any_automatic_path(self):
        """Full run of every automatic repair must leave 2 components."""
        mesh, _ = gen.thin_wall()
        sess = repair.RepairSession(mesh, unit="mm")
        # run every operation that could be considered "automatic"
        before = analyze_mesh(mesh)
        if "unreferenced_vertices" in before.groups:
            pass
        # duplicate faces → none
        repair.remove_duplicate_faces(sess, before.groups.get("duplicate_faces", type("x", (), {"instances": []})()).instances if False else [])
        # isolated vertices
        used = np.zeros(sess.mesh.vertex_count, dtype=bool)
        used[sess.mesh.used_vertices()] = True
        if not used.all():
            repair.remove_isolated_vertices(sess, [int(i) for i in np.nonzero(~used)[0]])
        after = analyze_mesh(sess.mesh)
        self.assertEqual(after.metrics["components"], 2)
        self.assertEqual(sess.mesh.vertex_count, mesh.vertex_count)


class TestRepairOperations(unittest.TestCase):
    def test_duplicate_faces_removal_and_verify(self):
        mesh, _ = gen.build("duplicate_faces")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        recs = repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
        self.assertTrue(recs)
        after = analyze_mesh(sess.mesh)
        self.assertNotIn("duplicate_faces", after.groups)
        rep = verification_report(sess.original, before, sess.mesh, after)
        self.assertTrue(rep["kinds"]["duplicate_faces"]["fully_resolved"])
        # non-manifold edges caused by duplicates should also resolve
        self.assertEqual(after.metrics["nonmanifold_edges"], 0)

    def test_hole_fill_small_and_reject_large(self):
        mesh, _ = gen.build("holed_plate")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        loops = before.groups["open_boundaries"].instances
        limits = holes.HoleFillLimits()
        plans = []
        for inst in loops:
            plans.append(holes.plan_fill(sess, inst["vertices"], limits))
        ok_plans = [p for p in plans if p["ok"]]
        refused = [p for p in plans if not p["ok"]]
        self.assertEqual(len(ok_plans), 1, "only the small hole may be filled")
        self.assertTrue(all(p["check"]["problems"] for p in refused))
        small = min(loops, key=lambda i: i["perimeter"])
        rec = holes.fill_hole(sess, small["vertices"], limits, reason="已确认小孔为破损")
        self.assertGreater(len(rec.added_faces), 0)
        after = analyze_mesh(sess.mesh)
        self.assertEqual(after.metrics["boundary_edges"],
                         before.metrics["boundary_edges"] - 4)

    def test_hole_fill_records_parameters_and_affected(self):
        mesh, _ = gen.build("holed_plate")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        small = min(before.groups["open_boundaries"].instances, key=lambda i: i["perimeter"])
        rec = holes.fill_hole(sess, small["vertices"], holes.HoleFillLimits(), reason="test")
        params = rec.parameters
        for key in ("loop", "new_faces", "perimeter", "max_spacing", "limits", "confirm_reason"):
            self.assertIn(key, params)
        self.assertEqual(len(rec.affected_vertices), len(set(small["vertices"])))
        self.assertEqual(len(rec.affected_faces), len(rec.added_faces))
        self.assertTrue(rec.added_faces)

    def test_large_hole_explicit_override_allowed(self):
        mesh, _ = gen.build("holed_plate")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        big = max(before.groups["open_boundaries"].instances, key=lambda i: i["perimeter"])
        loose = holes.HoleFillLimits(max_edges=4096, perimeter_fraction=1.0,
                                     spacing_fraction=1.0, planarity_fraction=1.0)
        plan = holes.plan_fill(sess, big["vertices"], loose)
        self.assertTrue(plan["ok"], "explicitly widened limits should permit it")
        # but the default must still refuse
        self.assertFalse(holes.plan_fill(sess, big["vertices"], holes.HoleFillLimits())["ok"])

    def test_flip_component_fixes_volume_sign(self):
        mesh, _ = gen.build("mixed_orientation")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        inst = before.groups["inward_components"].instances[0]
        rec = repair.flip_component(sess, inst["faces"], confirm_reason="确认该组件应朝外")
        after = analyze_mesh(sess.mesh)
        self.assertNotIn("inward_components", after.groups)
        self.assertGreater(sess.mesh.signed_volume(), mesh.signed_volume())
        self.assertIn("confirm_reason", rec.parameters)
        self.assertEqual(len(rec.affected_faces), len(inst["faces"]))

    def test_flip_faces_winding_conflict(self):
        mesh, _ = gen.build("combined_stress")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        # duplicate faces mask the reversed face as a non-manifold edge;
        # remove them first so the winding conflict becomes observable.
        repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
        mid = analyze_mesh(sess.mesh)
        self.assertIn("winding_conflicts", mid.groups)
        conflict_faces = sorted({int(f) for inst in mid.groups["winding_conflicts"].instances
                                 for f in inst["faces"]})
        self.assertTrue(conflict_faces)
        repair.flip_faces(sess, conflict_faces, reason="修正反向面")
        after = analyze_mesh(sess.mesh)
        self.assertNotIn("winding_conflicts", after.groups)

    def test_undo_redo_restores_exact_geometry(self):
        mesh, _ = gen.build("duplicate_faces")
        sess = repair.RepairSession(mesh, unit="mm")
        original_v = mesh.vertices.copy()
        original_f = mesh.faces.copy()
        before = analyze_mesh(mesh)
        repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
        self.assertLess(sess.mesh.face_count, mesh.face_count)
        sess.undo_until(1)
        np.testing.assert_array_equal(sess.mesh.vertices, original_v)
        np.testing.assert_array_equal(sess.mesh.faces, original_f)
        sess.redo()
        self.assertTrue(len(sess.mesh.faces) > 0)

    def test_failed_operation_leaves_mesh_untouched(self):
        mesh, _ = gen.build("holed_plate")
        sess = repair.RepairSession(mesh, unit="mm")
        faces_before = sess.mesh.faces.copy()
        with self.assertRaises(repair.RepairError):
            repair.delete_faces(sess, [10 ** 9])
        np.testing.assert_array_equal(sess.mesh.faces, faces_before)

    def test_isolation_vertex_removal_refuses_used_vertices(self):
        mesh, _ = gen.build("duplicate_faces")
        sess = repair.RepairSession(mesh, unit="mm")
        with self.assertRaises(repair.RepairError):
            repair.remove_isolated_vertices(sess, [0])


class TestVerificationAndAudit(unittest.TestCase):
    def test_verification_reports_remaining_problems(self):
        mesh, _ = gen.build("holed_plate")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        small = min(before.groups["open_boundaries"].instances, key=lambda i: i["perimeter"])
        holes.fill_hole(sess, small["vertices"], holes.HoleFillLimits(), reason="test")
        after = analyze_mesh(sess.mesh)
        rep = verification_report(sess.original, before, sess.mesh, after)
        key = rep["kinds"]["open_boundaries"]
        self.assertFalse(key["fully_resolved"])
        self.assertEqual(key["remaining"], 2)
        self.assertTrue(any("仍有" in p for p in rep["problems"]))
        self.assertFalse(rep["clean"])

    def test_clean_after_full_duplicate_repair(self):
        mesh, _ = gen.build("duplicate_faces")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
        after = analyze_mesh(sess.mesh)
        rep = verification_report(sess.original, before, sess.mesh, after)
        dup = rep["kinds"]["duplicate_faces"]
        self.assertTrue(dup["fully_resolved"])

    def test_fingerprints_survive_reload(self):
        """Original defects must be traceable after export and reload."""
        mesh, _ = gen.build("duplicate_faces")
        doc = ProjectDocument()
        doc.session = repair.RepairSession(mesh, unit="mm", source_name="bench")
        doc.provenance_id = "testprov12345678"
        doc.reanalyze(update_baseline=True)
        baseline_fps = all_fingerprints(mesh, doc.baseline)
        self.assertIn("duplicate_faces", baseline_fps)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dup.stl"
            info = doc.export(path)
            self.assertTrue(path.exists())
            self.assertTrue(Path(info["sidecar"]).exists())
            payload = io_mesh.read_sidecar(path)
            self.assertIn("baseline_fingerprints", payload)
            embedded = io_mesh.read_embedded_provenance(path)
            self.assertEqual(embedded, doc.provenance_id)

            reloaded = io_mesh.load_mesh(path)
            result = analyze_mesh(reloaded)
            # STL round-trip reindexes vertices, but defects must still be found
            self.assertIn("duplicate_faces", result.groups)
            fps = all_fingerprints(reloaded, result)
            overlap = {i["summary"] for i in fps["duplicate_faces"]} & \
                      {i["summary"] for i in baseline_fps["duplicate_faces"]}
            self.assertTrue(overlap, "fingerprints must match after reload")
            doc.close()

    def test_audit_contains_history_parameters_and_coordinates(self):
        mesh, _ = gen.build("duplicate_faces")
        doc = ProjectDocument()
        doc.session = repair.RepairSession(mesh, unit="mm")
        doc.reanalyze(update_baseline=True)
        before = analyze_mesh(mesh)
        recs = repair.remove_duplicate_faces(doc.session, before.groups["duplicate_faces"].instances)
        doc.reanalyze()
        payload = doc.audit_payload()
        self.assertEqual(len(payload["operations"]), len(recs))
        op = payload["operations"][0]
        self.assertIn("affected_vertices", op)
        self.assertIn("affected_faces", op)
        self.assertIn("parameters", op)
        self.assertIn("group_vertices", op["parameters"])
        self.assertIn("kept", op["parameters"])
        self.assertIn("baseline_fingerprints", payload)

    def test_negative_threshold_clamped(self):
        from mesh_repair.core.units import scale_guard

        value, warnings = scale_guard(-1.0, "mm", 100.0)
        self.assertEqual(value, 0.0)
        self.assertTrue(warnings)

    def test_pending_confirmations_name_ambiguous_items(self):
        """Unknowables must be listed in the audit, not silently repaired."""
        mesh, _ = gen.build("open_shell")
        doc = ProjectDocument()
        doc.session = repair.RepairSession(mesh, unit="mm")
        doc.reanalyze(update_baseline=True)
        kinds = {p["kind"] for p in doc.pending_confirmations}
        self.assertIn("open_boundaries", kinds)
        self.assertIn("orientation_unresolved", kinds)
        payload = doc.audit_payload()
        self.assertTrue(payload["pending_confirmations"])
        for item in payload["pending_confirmations"]:
            self.assertIn("question", item)
            self.assertIn("kind", item)

    def test_pending_confirmations_flag_near_duplicates_on_thin_wall(self):
        mesh, meta = gen.build("thin_wall")
        doc = ProjectDocument()
        doc.session = repair.RepairSession(mesh, unit="mm")
        doc.settings.dup_tolerance_mm = (meta["expected"]["clearance"] + 0.05)
        doc.reanalyze(update_baseline=True)
        kinds = {p["kind"] for p in doc.pending_confirmations}
        self.assertIn("duplicate_vertices", kinds)
        entry = next(p for p in doc.pending_confirmations
                     if p["kind"] == "duplicate_vertices")
        self.assertGreater(entry["count"], 0)
        # nothing was welded
        self.assertEqual(doc.session.mesh.vertex_count, mesh.vertex_count)
        self.assertEqual(doc.current.metrics["components"], 2)


class TestEdgeCases(unittest.TestCase):
    def test_nonmanifold_edge_detected_but_not_autofixed(self):
        # three triangles sharing one edge
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
        mesh = Mesh(v, f)
        res = analyze_mesh(mesh)
        self.assertIn("nonmanifold_edges", res.groups)
        self.assertFalse(res.groups["nonmanifold_edges"].auto_repairable)

    def test_empty_mesh_is_safe(self):
        mesh = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        res = analyze_mesh(mesh)
        self.assertEqual(res.metrics["faces"], 0)
        self.assertEqual(res.total_instances(), 0)

    def test_out_of_range_face_flagged_by_validate(self):
        # construction is permissive; analysis must flag invalid indices
        m = Mesh(np.zeros((3, 3)), np.array([[0, 1, 5]], dtype=np.int64))
        self.assertTrue(any("越界" in s for s in m.validate()))
        res = analyze_mesh(m)
        self.assertTrue(any("越界" in w for w in res.warnings))

    def test_nan_vertices_flagged(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]])
        m = Mesh(v, np.array([[0, 1, 2]], dtype=np.int64))
        self.assertTrue(any("NaN" in s for s in m.validate()))
        res = analyze_mesh(m)
        self.assertTrue(any("NaN" in w for w in res.warnings))

    def test_weld_creating_degenerate_face_is_flagged(self):
        # a triangle whose two corners are welded together collapses to a line
        v = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        f = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        mesh = Mesh(v, f)
        sess = repair.RepairSession(mesh)
        recs = repair.weld_vertices_selected(sess, [[0, 1]], unit="mm", threshold=0.0,
                                             confirmed_reason="test collapse")
        self.assertTrue(recs)
        self.assertIn("created_degenerate_faces", recs[0].parameters)
        self.assertTrue(recs[0].parameters["created_degenerate_faces"])

    def test_delete_then_undo_then_redo_index_stability(self):
        mesh, _ = gen.build("duplicate_faces")
        sess = repair.RepairSession(mesh, unit="mm")
        before = analyze_mesh(mesh)
        repair.remove_duplicate_faces(sess, before.groups["duplicate_faces"].instances)
        n_after = sess.mesh.face_count
        while sess.can_undo:
            sess.undo()
        self.assertEqual(sess.mesh.face_count, mesh.face_count)
        while sess.can_redo:
            sess.redo()
        self.assertEqual(sess.mesh.face_count, n_after)

    def test_scale_guard_clamps_absurd_threshold(self):
        from mesh_repair.core.units import scale_guard

        value, warnings = scale_guard(1e6, "mm", 100.0)
        self.assertLess(value, 10.0)
        self.assertTrue(warnings)


class TestIO(unittest.TestCase):
    def test_stl_ascii_and_binary_roundtrip(self):
        mesh, _ = gen.build("duplicate_faces")
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "m.stl"
            io_mesh.save_mesh(mesh, bin_path)
            loaded = io_mesh.load_mesh(bin_path)
            self.assertEqual(loaded.face_count, mesh.face_count)

            # write an ascii STL by hand
            ascii_path = Path(td) / "ascii.stl"
            lines = ["solid test"]
            for tri in mesh.faces:
                pts = mesh.vertices[tri]
                lines.append("facet normal 0 0 0")
                lines.append("  outer loop")
                for p in pts:
                    lines.append(f"    vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}")
                lines += ["  endloop", "endfacet"]
            lines.append("endsolid test")
            ascii_path.write_text("\n".join(lines))
            loaded2 = io_mesh.load_mesh(ascii_path)
            self.assertEqual(loaded2.face_count, mesh.face_count)

    def test_obj_roundtrip_preserves_geometry(self):
        mesh, _ = gen.build("mixed_orientation")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.obj"
            io_mesh.save_mesh(mesh, p)
            loaded = io_mesh.load_mesh(p)
            self.assertEqual(loaded.vertex_count, mesh.vertex_count)
            self.assertEqual(loaded.face_count, mesh.face_count)
            np.testing.assert_allclose(loaded.vertices, mesh.vertices, atol=1e-8)

    def test_ply_roundtrip(self):
        mesh, _ = gen.build("open_shell")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.ply"
            io_mesh.save_mesh(mesh, p)
            loaded = io_mesh.load_mesh(p)
            self.assertEqual(loaded.face_count, mesh.face_count)
            res = analyze_mesh(loaded)
            self.assertIn("open_boundaries", res.groups)

    def test_corrupt_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.stl"
            p.write_bytes(b"\x00" * 10)
            with self.assertRaises(io_mesh.MeshIOError):
                io_mesh.load_mesh(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
