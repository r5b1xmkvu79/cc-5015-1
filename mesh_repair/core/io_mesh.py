"""Mesh file IO without third-party dependencies.

Supports binary/ascii STL, OBJ, and ascii/binary-little-endian PLY.  Unit
metadata is not part of these formats; the importer asks the user and the
exporter embeds a provenance token plus a JSON sidecar so a reloaded model
can be traced back to the original defect baseline.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .mesh import Mesh

SUPPORTED_SUFFIXES = {".stl", ".obj", ".ply"}
PROVENANCE_COMMENT = "MESHREPAIR_PROVENANCE"


class MeshIOError(RuntimeError):
    pass


def load_mesh(path: str | Path) -> Mesh:
    p = Path(path)
    if not p.exists():
        raise MeshIOError(f"文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix == ".stl":
        return _load_stl(p)
    if suffix == ".obj":
        return _load_obj(p)
    if suffix == ".ply":
        return _load_ply(p)
    raise MeshIOError(f"不支持的格式: {suffix}")


def save_mesh(mesh: Mesh, path: str | Path, *, provenance_id: str = "", comment: str = "") -> None:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".stl":
        _save_stl(mesh, p, provenance_id=provenance_id)
    elif suffix == ".obj":
        _save_obj(mesh, p, provenance_id=provenance_id, comment=comment)
    elif suffix == ".ply":
        _save_ply(mesh, p, provenance_id=provenance_id, comment=comment)
    else:
        raise MeshIOError(f"不支持的导出格式: {suffix}")


def read_embedded_provenance(path: str | Path) -> str:
    """Return the provenance id embedded in the file header/comment, if any."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return ""
    text = head.decode("latin-1", errors="ignore")
    # binary STL stores a fixed 80-byte header NUL-padded with no line break;
    # treat NUL as a line separator so the token is isolated from binary data.
    for line in text.replace("\x00", "\n").splitlines():
        raw = line.strip().lstrip("#").strip()
        if raw.startswith(PROVENANCE_COMMENT):
            parts = raw.split(":", 1)
            return parts[1].strip() if len(parts) == 2 else ""
    return ""


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------


def _looks_ascii_stl(data: bytes) -> bool:
    head = data[:512].lstrip()
    return head.startswith(b"solid") and b"facet" in data[:2048]


def _load_stl(path: Path) -> Mesh:
    data = path.read_bytes()
    if len(data) < 84:
        raise MeshIOError("STL 文件过短")
    if _looks_ascii_stl(data):
        return _load_ascii_stl(data)
    return _load_binary_stl(data)


def _load_binary_stl(data: bytes) -> Mesh:
    if len(data) < 84:
        raise MeshIOError("二进制 STL 文件头不完整")
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + count * 50
    if count == 0:
        raise MeshIOError("二进制 STL 声明 0 个三角形")
    if len(data) < expected:
        raise MeshIOError(
            f"二进制 STL 截断：声明 {count} 个三角形，需 {expected} 字节，实际 {len(data)}"
        )
    rec = np.frombuffer(data, dtype=np.uint8, count=count * 50, offset=84)
    rec = rec.reshape(count, 50)
    tri = rec[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)
    vertices = tri.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = Mesh(vertices, faces, name=path_name_fallback(""))
    return _dedup_exact(mesh)


def _load_ascii_stl(data: bytes) -> Mesh:
    text = data.decode("latin-1", errors="ignore")
    verts: list[list[float]] = []
    current: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            parts = s.split()
            if len(parts) < 4:
                raise MeshIOError("ASCII STL 顶点行不完整")
            try:
                current.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError as exc:
                raise MeshIOError(f"ASCII STL 无法解析顶点: {s}") from exc
            if len(current) == 3:
                verts.extend(current)
                current = []
    if not verts:
        raise MeshIOError("ASCII STL 中未找到三角形")
    v = np.array(verts, dtype=np.float64)
    f = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    return _dedup_exact(Mesh(v, f))


def _dedup_exact(mesh: Mesh) -> Mesh:
    """STL has no shared vertices; merge bit-identical coordinates only.

    Bit-identical is safe across thin walls because opposite wall surfaces
    have different coordinates.  Near-coincident points are left for the
    detector to report.
    """
    v = mesh.vertices
    if len(v) == 0:
        return mesh
    order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
    sorted_v = v[order]
    group_starts = np.ones(len(sorted_v), dtype=bool)
    group_starts[1:] = np.any(sorted_v[1:] != sorted_v[:-1], axis=1)
    group_ids_sorted = np.cumsum(group_starts) - 1
    inverse = np.empty(len(v), dtype=np.int64)
    inverse[order] = group_ids_sorted
    new_v = sorted_v[group_starts]
    new_f = inverse[mesh.faces]
    return Mesh(new_v, new_f, name=mesh.name)


def path_name_fallback(name: str) -> str:
    return name or "mesh"


def _save_stl(mesh: Mesh, path: Path, *, provenance_id: str = "") -> None:
    header = f"{PROVENANCE_COMMENT}:{provenance_id}".encode("ascii", errors="ignore")[:80]
    header = header.ljust(80, b"\0")
    tri = mesh.vertices[mesh.faces].astype("<f4")
    n = len(tri)
    buf = bytearray()
    buf += header
    buf += struct.pack("<I", n)
    rec = np.zeros((n, 50), dtype=np.uint8)
    # normal per face
    v = tri.astype(np.float64)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = (normals / lengths).astype("<f4")
    rec[:, 0:12] = normals.view(np.uint8).reshape(n, 12)
    rec[:, 12:48] = tri.astype("<f4").view(np.uint8).reshape(n, 36)
    # attribute byte count left zero
    buf += rec.tobytes()
    path.write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def _load_obj(path: Path) -> Mesh:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                raise MeshIOError(f"OBJ 第 {lineno} 行顶点不足 3 个坐标")
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx = []
            for token in parts:
                # v, v/vt, v//vn, v/vt/vn
                first = token.split("/")[0]
                try:
                    i = int(first)
                except ValueError as exc:
                    raise MeshIOError(f"OBJ 第 {lineno} 行面索引无效: {token}") from exc
                idx.append(i - 1 if i > 0 else len(verts) + i)
            if len(idx) < 3:
                raise MeshIOError(f"OBJ 第 {lineno} 行面顶点少于 3 个")
            # fan triangulation for quads and n-gons
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        raise MeshIOError("OBJ 未包含可用顶点/面")
    return Mesh(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64))


def _save_obj(mesh: Mesh, path: Path, *, provenance_id: str = "", comment: str = "") -> None:
    lines = [f"# {PROVENANCE_COMMENT}:{provenance_id}"]
    if comment:
        for ln in comment.splitlines():
            lines.append(f"# {ln}")
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
    for f in mesh.faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------


def _load_ply(path: Path) -> Mesh:
    data = path.read_bytes()
    if not data.startswith(b"ply"):
        raise MeshIOError("不是 PLY 文件")
    end_header = data.find(b"end_header")
    if end_header < 0:
        raise MeshIOError("PLY 缺少 end_header")
    body_start = data.find(b"\n", end_header) + 1
    header = data[:body_start].decode("latin-1")
    fmt = "ascii"
    vertex_count = face_count = 0
    vertex_props: list[str] = []
    face_props: list[str] = []
    current = None
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                vertex_count = int(parts[2])
            elif current == "face":
                face_count = int(parts[2])
        elif parts[0] == "property":
            if current == "vertex":
                vertex_props.append(parts[-1])
            elif current == "face":
                face_props.append(parts[-1])
        elif parts[0] == "comment":
            continue

    if "vertex" not in vertex_props and vertex_props:
        pass
    if not ({"x", "y", "z"} <= set(vertex_props)):
        raise MeshIOError("PLY 顶点缺少 x/y/z 属性")

    if fmt == "ascii":
        return _load_ply_ascii(data[body_start:], vertex_count, face_count, vertex_props, face_props)
    if fmt.startswith("binary_little_endian"):
        return _load_ply_binary(data[body_start:], vertex_count, face_count, vertex_props, face_props)
    raise MeshIOError(f"不支持的 PLY 编码: {fmt}")


_PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


def _load_ply_ascii(blob: bytes, nv: int, nf: int, vprops: list[str], fprops: list[str]) -> Mesh:
    lines = blob.decode("latin-1").splitlines()
    idx = 0
    if "x" not in vprops:
        raise MeshIOError("PLY 顶点属性顺序异常")
    xi, yi, zi = vprops.index("x"), vprops.index("y"), vprops.index("z")
    verts = np.zeros((nv, 3), dtype=np.float64)
    while idx < len(lines) and len(lines[idx].strip().split()) < len(vprops):
        idx += 1
    for i in range(nv):
        parts = lines[idx].split()
        idx += 1
        verts[i] = (float(parts[xi]), float(parts[yi]), float(parts[zi]))
    faces: list[tuple[int, int, int]] = []
    for _ in range(nf):
        parts = lines[idx].split()
        idx += 1
        # face line: <count> i0 i1 ... [properties after]
        n = int(parts[0])
        ids = [int(x) for x in parts[1:1 + n]]
        if len(ids) < 3:
            continue
        for k in range(1, len(ids) - 1):
            faces.append((ids[0], ids[k], ids[k + 1]))
    if not faces:
        raise MeshIOError("PLY 未包含三角/多边形面")
    return Mesh(verts, np.array(faces, dtype=np.int64))


def _load_ply_binary(blob: bytes, nv: int, nf: int, vprops: list[str], fprops: list[str]) -> Mesh:
    dtype = np.dtype([(p, "<" + _PLY_DTYPES.get(p, "f4")) for p in vprops])
    if dtype.itemsize * nv > len(blob):
        raise MeshIOError("PLY 二进制顶点数据截断")
    vertex_block = np.frombuffer(blob, dtype=dtype, count=nv)
    verts = np.stack([vertex_block["x"], vertex_block["y"], vertex_block["z"]], axis=1).astype(np.float64)
    offset = dtype.itemsize * nv
    # faces: assume list property with uint8 count + int32 indices, the common case
    faces: list[tuple[int, int, int]] = []
    pos = offset
    for _ in range(nf):
        if pos >= len(blob):
            raise MeshIOError("PLY 二进局面数据截断")
        n = blob[pos]
        pos += 1
        ids = list(struct.unpack_from(f"<{n}i", blob, pos))
        pos += 4 * n
        if n < 3:
            continue
        for k in range(1, n - 1):
            faces.append((ids[0], ids[k], ids[k + 1]))
    if not faces:
        raise MeshIOError("PLY 未包含面")
    return Mesh(verts, np.array(faces, dtype=np.int64))


def _save_ply(mesh: Mesh, path: Path, *, provenance_id: str = "", comment: str = "") -> None:
    header_lines = ["ply", "format ascii 1.0",
                    f"comment {PROVENANCE_COMMENT}:{provenance_id}"]
    if comment:
        for ln in comment.splitlines():
            header_lines.append(f"comment {ln}")
    header_lines.append(f"element vertex {mesh.vertex_count}")
    header_lines += ["property double x", "property double y", "property double z"]
    header_lines.append(f"element face {mesh.face_count}")
    header_lines += ["property list uchar int vertex_indices", "end_header"]
    out = ["\n".join(header_lines)]
    for v in mesh.vertices:
        out.append(f"{v[0]:.12g} {v[1]:.12g} {v[2]:.12g}")
    for f in mesh.faces:
        out.append(f"3 {int(f[0])} {int(f[1])} {int(f[2])}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Sidecar provenance
# ---------------------------------------------------------------------------


def sidecar_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(p.suffix + ".meshrepair.json")


def write_sidecar(path: str | Path, payload: dict) -> Path:
    p = sidecar_path(path)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_sidecar(path: str | Path) -> dict | None:
    p = sidecar_path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
