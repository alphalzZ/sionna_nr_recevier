"""Validated, canonical Sionna RT scene bundles and scene asset identities.

This module deliberately supports a small subset of Mitsuba XML and ASCII PLY;
scene files are data, never general-purpose Mitsuba projects.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping
import zipfile
import zlib
import xml.etree.ElementTree as ET

MAX_ZIP_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_SCENE_XML_BYTES = 256 * 1024
MAX_PLY_BYTES = 8 * 1024 * 1024
MAX_FILES = 33
MAX_SHAPES = 32
MAX_MATERIALS = 32
MAX_VERTICES = 100_000
MAX_FACES = 200_000
MAX_COORDINATE_M = 10_000.0
MAX_COMPRESSION_RATIO = 200

MAX_BUILTIN_SCENE_BYTES = 32 * 1024 * 1024
MAX_BUILTIN_SCENE_XML_BYTES = 4 * 1024 * 1024
MAX_BUILTIN_SCENE_FILES = 8_192

BUILTIN_SCENE_PRESETS = (
    ("empty", "LoS（空场景）", "无几何遮挡的 LoS 基线。"),
    ("ground", "地面", "内置地面平面。"),
    ("ground_wall", "地面＋墙面", "内置地面与墙面。"),
    ("box", "Sionna RT · 箱体", "内置箱体反射场景；位置沿用当前 RT 配置。"),
    ("box_knife", "Sionna RT · 箱体＋刀形体", "内置箱体与刀形体；位置沿用当前 RT 配置。"),
    ("box_one_screen", "Sionna RT · 单屏障", "内置箱体与单屏障；位置沿用当前 RT 配置。"),
    ("box_two_screens", "Sionna RT · 双屏障", "内置箱体与双屏障；位置沿用当前 RT 配置。"),
    ("double_reflector", "Sionna RT · 双反射板", "内置双反射板；位置沿用当前 RT 配置。"),
    ("etoile", "Sionna RT · Étoile", "内置城市场景（约 2.5 MiB）；追踪开销较高，位置沿用当前 RT 配置。"),
    ("floor_wall", "Sionna RT · 地面＋墙体", "内置地面与墙体；不同于可参数化地面＋墙面。"),
    ("florence", "Sionna RT · Florence 城区", "大型内置场景（约 12 MiB）；追踪和内存开销较高。"),
    ("munich", "Sionna RT · Munich 城区", "大型内置场景（约 11 MiB）；追踪和内存开销较高。"),
    ("san_francisco", "Sionna RT · San Francisco 城区", "大型内置场景（约 25 MiB）；追踪和内存开销较高。"),
    ("simple_reflector", "Sionna RT · 单反射板", "内置单反射板；位置沿用当前 RT 配置。"),
    ("simple_street_canyon", "Sionna RT · 简单街谷", "内置街谷；位置沿用当前 RT 配置。"),
    ("simple_street_canyon_with_cars", "Sionna RT · 含车辆街谷", "内置街谷与车辆；位置沿用当前 RT 配置。"),
    ("simple_wedge", "Sionna RT · 楔形场景", "内置楔形几何；当前传播模型不启用衍射。"),
    ("triple_reflector", "Sionna RT · 三反射板", "内置三反射板；位置沿用当前 RT 配置。"),
)
BUILTIN_SCENE_IDS = frozenset(scene for scene, _, _ in BUILTIN_SCENE_PRESETS)
SIONNA_RT_SCENE_IDS = frozenset(scene for scene, _, _ in BUILTIN_SCENE_PRESETS[3:])


def builtin_scene_catalog() -> list[dict[str, str]]:
    return [
        {"id": scene, "label": label, "description": description}
        for scene, label, description in BUILTIN_SCENE_PRESETS
    ]

_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z", re.ASCII)
_SCENE_FILE_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\.xml\Z", re.ASCII)
_PLY_PATH_RE = re.compile(r"meshes/([A-Za-z0-9_-]{1,64})\.ply\Z", re.ASCII)
_BUILTIN_PLY_PATH_RE = re.compile(r"meshes/[\w-]{1,128}\.ply\Z")
_XML_DECLARATION_RE = re.compile(br"^(?:\xef\xbb\xbf)?\s*<\?xml\s+([^?]*)\?>", re.IGNORECASE)
_ENCODING_RE = re.compile(br"\bencoding\s*=\s*(['\"])([^'\"]+)\1", re.IGNORECASE)


@dataclass(frozen=True)
class RtSceneAssets:
    """A validated scene root and path-independent identity for its assets."""

    root: Path
    scene_file: str
    file_sha256: dict[str, str]
    bundle_sha256: str
    source: str


@dataclass(frozen=True)
class ValidatedRtSceneBundle:
    """Canonical no-write result returned by :func:`parse_scene_bundle_zip`."""

    scene_file: str
    files: dict[str, bytes]
    file_sha256: dict[str, str]
    bundle_sha256: str
    mesh_summary: dict[str, dict[str, Any]]


def _source_name(source: str) -> str:
    if source not in {"builtin", "parameterized", "imported"}:
        raise ValueError("scene source must be builtin, parameterized, or imported")
    return source


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_sha256_for(file_sha256: Mapping[str, str]) -> str:
    """Hash a canonical sorted relative-path-to-file-hash JSON object."""
    normalized: dict[str, str] = {}
    for path, digest in file_sha256.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("scene file hashes must map relative paths to SHA-256 strings")
        if path in normalized:
            raise ValueError(f"duplicate scene asset path: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid SHA-256 for scene asset: {path}")
        normalized[path] = digest
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256(canonical.encode("ascii"))


def _hash_files(files: Mapping[str, bytes]) -> dict[str, str]:
    return {path: _sha256(files[path]) for path in sorted(files)}


def _bundle_identity(files: Mapping[str, bytes]) -> tuple[dict[str, str], str]:
    hashes = _hash_files(files)
    return hashes, bundle_sha256_for(hashes)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _float_text(number: float) -> str:
    """Stable shortest-round-trip decimal representation for canonical assets."""
    return repr(float(number))


def _safe_relative_path(path: str) -> bool:
    if not path or "\\" in path or "\x00" in path or path.startswith("/"):
        return False
    pure = PurePosixPath(path)
    return not pure.is_absolute() and all(part not in {"", ".", ".."} for part in pure.parts)


def _validate_xml(raw: bytes, scene_file: str) -> tuple[bytes, list[dict[str, str]], dict[str, dict[str, Any]]]:
    if len(raw) > MAX_SCENE_XML_BYTES:
        raise ValueError(f"{scene_file} exceeds the {MAX_SCENE_XML_BYTES}-byte XML limit")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{scene_file} must be UTF-8 XML") from exc
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.IGNORECASE):
        raise ValueError("DOCTYPE and ENTITY declarations are not allowed in scene XML")
    declaration = _XML_DECLARATION_RE.match(raw)
    if declaration:
        encoding = _ENCODING_RE.search(declaration.group(1))
        if encoding and encoding.group(2).decode("ascii", errors="ignore").lower() != "utf-8":
            raise ValueError("scene XML declaration must use UTF-8")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"invalid scene XML: {exc}") from exc
    if root.tag != "scene" or root.attrib != {"version": "2.1.0"}:
        raise ValueError('scene XML root must be <scene version="2.1.0"> with no extra attributes')
    if root.text and root.text.strip():
        raise ValueError("scene XML may contain only top-level bsdf and shape elements")

    materials: dict[str, tuple[str, float]] = {}
    shapes: list[dict[str, str]] = []
    identifiers: set[str] = set()
    referenced_files: set[str] = set()

    def check_whitespace(element: ET.Element, context: str) -> None:
        if element.text and element.text.strip():
            raise ValueError(f"unexpected text in {context}")
        for child in element:
            if child.tail and child.tail.strip():
                raise ValueError(f"unexpected text in {context}")

    for element in root:
        check_whitespace(element, f"<{element.tag}>")
        if element.tag == "bsdf":
            if element.attrib.get("type") != "itu-radio-material" or set(element.attrib) != {"type", "id"}:
                raise ValueError("only itu-radio-material bsdf elements with type and id are allowed")
            identifier = element.attrib["id"]
            if not _ID_RE.fullmatch(identifier):
                raise ValueError("material id must be 1-64 ASCII letters, digits, '_' or '-'")
            if identifier in identifiers:
                raise ValueError(f"duplicate scene id: {identifier}")
            identifiers.add(identifier)
            if len(element) != 3:
                raise ValueError(f"material {identifier} must define type, thickness, and scattering_coefficient")
            values: dict[str, str] = {}
            expected = (
                ("string", "type"),
                ("float", "thickness"),
                ("float", "scattering_coefficient"),
            )
            for child, (tag, name) in zip(element, expected):
                if child.tag != tag or child.attrib.keys() != {"name", "value"}:
                    raise ValueError(f"material {identifier} has an unsupported child element or attribute")
                if child.attrib["name"] != name or len(child):
                    raise ValueError(f"material {identifier} has an unsupported or repeated property")
                if child.text and child.text.strip():
                    raise ValueError(f"unexpected text in material {identifier}")
                values[name] = child.attrib["value"]
            material_type = values["type"]
            if material_type not in {"concrete", "brick"}:
                raise ValueError(f"material {identifier} type must be concrete or brick")
            thickness = _finite_float(values["thickness"], f"material {identifier} thickness")
            scattering = _finite_float(values["scattering_coefficient"], f"material {identifier} scattering_coefficient")
            if thickness <= 0.0 or thickness > 10.0:
                raise ValueError(f"material {identifier} thickness must be in (0,10] m")
            if scattering != 0.0:
                raise ValueError(f"material {identifier} scattering_coefficient must be 0")
            materials[identifier] = (material_type, thickness)
            if len(materials) > MAX_MATERIALS:
                raise ValueError(f"scene may contain at most {MAX_MATERIALS} materials")
        elif element.tag == "shape":
            if element.attrib.get("type") != "ply" or set(element.attrib) != {"type", "id"}:
                raise ValueError("only ply shape elements with type and id are allowed")
            identifier = element.attrib["id"]
            if not _ID_RE.fullmatch(identifier):
                raise ValueError("shape id must be 1-64 ASCII letters, digits, '_' or '-'")
            if identifier in identifiers:
                raise ValueError(f"duplicate scene id: {identifier}")
            identifiers.add(identifier)
            if len(shapes) >= MAX_SHAPES:
                raise ValueError(f"scene may contain at most {MAX_SHAPES} shapes")
            if len(element) != 3:
                raise ValueError(f"shape {identifier} must contain filename, face_normals, and one bsdf ref")
            filename: str | None = None
            material_id: str | None = None
            for child in element:
                if child.tag == "string" and child.attrib.keys() == {"name", "value"}:
                    if child.attrib["name"] != "filename" or filename is not None or len(child):
                        raise ValueError(f"shape {identifier} has an unsupported string property")
                    filename = child.attrib["value"]
                elif child.tag == "boolean" and child.attrib.keys() == {"name", "value"}:
                    if child.attrib["name"] != "face_normals" or child.attrib["value"] != "true" or len(child):
                        raise ValueError(f"shape {identifier} requires face_normals=true")
                elif child.tag == "ref" and child.attrib.keys() == {"id", "name"}:
                    if child.attrib["name"] != "bsdf" or material_id is not None or len(child):
                        raise ValueError(f"shape {identifier} requires one bsdf ref")
                    material_id = child.attrib["id"]
                else:
                    raise ValueError(f"shape {identifier} has an unsupported child element or attribute")
                if child.text and child.text.strip():
                    raise ValueError(f"unexpected text in shape {identifier}")
            if filename is None or material_id is None:
                raise ValueError(f"shape {identifier} requires a filename and one bsdf ref")
            if not _safe_relative_path(filename) or not _PLY_PATH_RE.fullmatch(filename):
                raise ValueError(f"shape {identifier} filename must be meshes/<ASCII-id>.ply")
            if filename in referenced_files:
                raise ValueError(f"PLY file is referenced by more than one shape: {filename}")
            referenced_files.add(filename)
            shapes.append({"id": identifier, "filename": filename, "material_id": material_id})
        else:
            raise ValueError(f"unsupported scene element: <{element.tag}>")

    if len(shapes) > MAX_SHAPES:
        raise ValueError(f"scene may contain at most {MAX_SHAPES} shapes")
    for shape in shapes:
        if shape["material_id"] not in materials:
            raise ValueError(f"shape {shape['id']} references missing material {shape['material_id']}")

    canonical: list[str] = ['<scene version="2.1.0">']
    for identifier in sorted(materials):
        material_type, thickness = materials[identifier]
        canonical.extend(
            [
                f'  <bsdf type="itu-radio-material" id="{identifier}">',
                f'    <string name="type" value="{material_type}"/>',
                f'    <float name="thickness" value="{_float_text(thickness)}"/>',
                '    <float name="scattering_coefficient" value="0.0"/>',
                "  </bsdf>",
            ]
        )
    for shape in sorted(shapes, key=lambda value: value["id"]):
        canonical.extend(
            [
                f'  <shape type="ply" id="{shape["id"]}">',
                f'    <string name="filename" value="{shape["filename"]}"/>',
                '    <boolean name="face_normals" value="true"/>',
                f'    <ref id="{shape["material_id"]}" name="bsdf"/>',
                "  </shape>",
            ]
        )
    canonical.append("</scene>")
    return ("\n".join(canonical) + "\n").encode("utf-8"), shapes, {
        identifier: {"type": value[0], "thickness_m": value[1]}
        for identifier, value in sorted(materials.items())
    }


def _ply_int(token: str, name: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]+", token, re.ASCII):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(token, 10)
    except ValueError as exc:  # defensive; the grammar above is exact
        raise ValueError(f"{name} must be an integer") from exc


def _validate_ply(raw: bytes, filename: str) -> tuple[bytes, dict[str, Any]]:
    if len(raw) > MAX_PLY_BYTES:
        raise ValueError(f"{filename} exceeds the {MAX_PLY_BYTES}-byte PLY limit")
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filename} must be ASCII PLY") from exc
    lines = text.splitlines()
    if len(lines) < 11 or lines[0].strip() != "ply" or lines[1].split() != ["format", "ascii", "1.0"]:
        raise ValueError(f"{filename} must use PLY format ascii 1.0")

    elements: list[tuple[str, int]] = []
    properties: dict[str, list[tuple[str, ...]]] = {}
    current: str | None = None
    index = 2
    found_end = False
    while index < len(lines):
        line = lines[index]
        index += 1
        fields = line.split()
        if not fields:
            raise ValueError(f"{filename} has a blank PLY header line")
        if fields[0] == "comment":
            continue
        if fields[0] == "element":
            if len(fields) != 3 or fields[1] not in {"vertex", "face"}:
                raise ValueError(f"{filename} contains an unsupported PLY element")
            count = _ply_int(fields[2], f"{filename} {fields[1]} count")
            if count < 0:
                raise ValueError(f"{filename} element counts must be non-negative")
            if fields[1] in properties:
                raise ValueError(f"{filename} repeats a PLY element")
            elements.append((fields[1], count))
            properties[fields[1]] = []
            current = fields[1]
            continue
        if fields[0] == "property":
            if current is None:
                raise ValueError(f"{filename} property appears before an element")
            properties[current].append(tuple(fields[1:]))
            continue
        if fields == ["end_header"]:
            found_end = True
            break
        raise ValueError(f"{filename} contains an unsupported PLY header directive")
    if not found_end:
        raise ValueError(f"{filename} is missing end_header")
    if elements != [("vertex", dict(elements).get("vertex", -1)), ("face", dict(elements).get("face", -1))]:
        raise ValueError(f"{filename} must declare vertex then face elements exactly once")
    vertex_count = dict(elements)["vertex"]
    face_count = dict(elements)["face"]
    if vertex_count < 3 or face_count < 1:
        raise ValueError(f"{filename} must contain at least 3 vertices and 1 triangle")
    if vertex_count > MAX_VERTICES or face_count > MAX_FACES:
        raise ValueError(f"{filename} exceeds the per-mesh vertex or face safety limit")
    if properties != {
        "vertex": [("float", "x"), ("float", "y"), ("float", "z")],
        "face": [("list", "uchar", "int", "vertex_indices")],
    }:
        raise ValueError(f"{filename} must contain only float x/y/z and list uchar int vertex_indices")

    cursor = index
    vertices: list[tuple[float, float, float]] = []
    for vertex_index in range(vertex_count):
        if cursor >= len(lines):
            raise ValueError(f"{filename} has fewer vertex rows than declared")
        fields = lines[cursor].split()
        cursor += 1
        if len(fields) != 3:
            raise ValueError(f"{filename} vertex {vertex_index} must have x, y, and z")
        point = tuple(_finite_float(value, f"{filename} vertex {vertex_index}") for value in fields)
        if any(abs(value) > MAX_COORDINATE_M for value in point):
            raise ValueError(f"{filename} coordinates must have absolute value at most {MAX_COORDINATE_M:g} m")
        vertices.append(point)  # type: ignore[arg-type]

    faces: list[tuple[int, int, int]] = []
    for face_index in range(face_count):
        if cursor >= len(lines):
            raise ValueError(f"{filename} has fewer face rows than declared")
        fields = lines[cursor].split()
        cursor += 1
        if len(fields) != 4 or fields[0] != "3":
            raise ValueError(f"{filename} face {face_index} must contain exactly three vertex indices")
        face = tuple(_ply_int(value, f"{filename} face {face_index} index") for value in fields[1:])
        if any(value < 0 or value >= vertex_count for value in face):
            raise ValueError(f"{filename} face {face_index} has an out-of-range vertex index")
        a, b, c = (vertices[value] for value in face)
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        if math.sqrt(sum(component * component for component in cross)) <= 1e-12:
            raise ValueError(f"{filename} face {face_index} is a degenerate triangle")
        faces.append(face)  # type: ignore[arg-type]

    if any(line.strip() for line in lines[cursor:]):
        raise ValueError(f"{filename} contains trailing data after the declared PLY rows")

    canonical = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {face_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    canonical.extend(" ".join(_float_text(value) for value in point) for point in vertices)
    canonical.extend("3 " + " ".join(str(value) for value in face) for face in faces)
    min_xyz = [min(point[axis] for point in vertices) for axis in range(3)]
    max_xyz = [max(point[axis] for point in vertices) for axis in range(3)]
    summary = {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bounds_m": {"min": min_xyz, "max": max_xyz},
        "bounds_xy_m": {"min": min_xyz[:2], "max": max_xyz[:2]},
    }
    return ("\n".join(canonical) + "\n").encode("ascii"), summary


def _validated_files(
    files: Mapping[str, bytes], scene_file: str, *, canonicalize: bool
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if not _SCENE_FILE_RE.fullmatch(scene_file):
        raise ValueError("scene_file must be a safe root-level XML filename")
    if scene_file not in files:
        raise ValueError(f"scene bundle is missing {scene_file}")
    if len(files) > MAX_FILES:
        raise ValueError(f"scene bundle may contain at most {MAX_FILES} files")
    xml_raw = files[scene_file]
    canonical_xml, shapes, _ = _validate_xml(xml_raw, scene_file)
    referenced = {shape["filename"] for shape in shapes}
    if set(files) != {scene_file, *referenced}:
        missing = sorted(({scene_file, *referenced}) - set(files))
        extra = sorted(set(files) - {scene_file, *referenced})
        if missing:
            raise ValueError(f"scene bundle is missing referenced asset(s): {', '.join(missing)}")
        raise ValueError(f"scene bundle contains unreferenced or unsupported file(s): {', '.join(extra)}")
    total_bytes = 0
    validated: dict[str, bytes] = {}
    summaries: dict[str, dict[str, Any]] = {}
    total_vertices = 0
    total_faces = 0
    for path, raw in files.items():
        if not isinstance(raw, bytes):
            raise ValueError(f"scene asset {path} content must be bytes")
        if path == scene_file:
            if len(raw) > MAX_SCENE_XML_BYTES:
                raise ValueError(f"{path} exceeds the {MAX_SCENE_XML_BYTES}-byte XML limit")
            validated[path] = canonical_xml if canonicalize else raw
        else:
            match = _PLY_PATH_RE.fullmatch(path)
            if not match:
                raise ValueError(f"unsupported scene asset path: {path}")
            if len(raw) > MAX_PLY_BYTES:
                raise ValueError(f"{path} exceeds the {MAX_PLY_BYTES}-byte PLY limit")
            canonical_ply, summary = _validate_ply(raw, path)
            total_vertices += summary["vertex_count"]
            total_faces += summary["face_count"]
            if total_vertices > MAX_VERTICES:
                raise ValueError(f"scene bundle exceeds {MAX_VERTICES} total vertices")
            if total_faces > MAX_FACES:
                raise ValueError(f"scene bundle exceeds {MAX_FACES} total faces")
            validated[path] = canonical_ply if canonicalize else raw
            summaries[path] = summary
        total_bytes += len(raw)
        if total_bytes > MAX_EXPANDED_BYTES:
            raise ValueError(f"scene bundle exceeds the {MAX_EXPANDED_BYTES}-byte expanded limit")
    return {path: validated[path] for path in sorted(validated)}, summaries


def parse_scene_bundle_zip(data: bytes | bytearray | memoryview) -> ValidatedRtSceneBundle:
    """Validate a ZIP bundle completely without writing any member to disk.

    The return value contains canonical ``scene.xml``/PLY bytes, per-file hashes,
    a host-independent bundle hash, and per-mesh count/bounds summaries.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("scene ZIP content must be bytes")
    archive_bytes = bytes(data)
    if len(archive_bytes) > MAX_ZIP_BYTES:
        raise ValueError(f"scene ZIP exceeds the {MAX_ZIP_BYTES}-byte compressed limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"invalid scene ZIP: {exc}") from exc

    file_data: dict[str, bytes] = {}
    seen_casefold: dict[str, str] = {}
    expanded_declared = 0
    expanded_actual = 0
    infos = archive.infolist()
    if len(infos) > MAX_FILES + 1:
        archive.close()
        raise ValueError(f"scene ZIP may contain at most {MAX_FILES} files plus one meshes/ directory entry")
    try:
        for info in infos:
            name = info.filename
            original_name = getattr(info, "orig_filename", name)
            if "\x00" in original_name or original_name != name:
                raise ValueError("scene ZIP contains a filename with a NUL byte")
            if not _safe_relative_path(name.rstrip("/") if info.is_dir() else name):
                raise ValueError(f"unsafe scene ZIP path: {name!r}")
            folded = name.casefold()
            if folded in seen_casefold:
                raise ValueError(f"duplicate or case-colliding scene ZIP path: {seen_casefold[folded]} and {name}")
            seen_casefold[folded] = name
            if info.flag_bits & 0x41:
                raise ValueError(f"encrypted scene ZIP entries are not supported: {name}")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(f"unsupported ZIP compression method for {name}")
            mode = (info.external_attr >> 16) & 0xFFFF
            mode_type = stat.S_IFMT(mode)
            is_directory = info.is_dir()
            if info.create_system == 3 and mode_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"scene ZIP contains a symlink or special file: {name}")
            if is_directory:
                if name != "meshes/" or info.file_size != 0 or mode_type not in {0, stat.S_IFDIR}:
                    raise ValueError("only a zero-byte meshes/ directory entry is allowed")
                try:
                    with archive.open(info, "r") as stream:
                        if stream.read(1):
                            raise ValueError(f"scene ZIP directory entry contains data: {name}")
                except (zipfile.BadZipFile, RuntimeError, EOFError, OSError, zlib.error) as exc:
                    raise ValueError(f"scene ZIP directory entry is corrupt or has an invalid CRC: {name}") from exc
                continue
            if mode_type == stat.S_IFDIR:
                raise ValueError(f"scene ZIP file has directory attributes: {name}")
            if name != "scene.xml" and not _PLY_PATH_RE.fullmatch(name):
                raise ValueError(f"scene ZIP may contain only scene.xml and meshes/<ASCII-id>.ply: {name}")
            if name == "scene.xml" and info.file_size > MAX_SCENE_XML_BYTES:
                raise ValueError(f"scene.xml exceeds the {MAX_SCENE_XML_BYTES}-byte XML limit")
            if name != "scene.xml" and info.file_size > MAX_PLY_BYTES:
                raise ValueError(f"{name} exceeds the {MAX_PLY_BYTES}-byte PLY limit")
            if info.file_size < 0 or info.compress_size < 0:
                raise ValueError(f"invalid ZIP member size: {name}")
            expanded_declared += info.file_size
            if expanded_declared > MAX_EXPANDED_BYTES:
                raise ValueError(f"scene ZIP exceeds the {MAX_EXPANDED_BYTES}-byte expanded limit")
            if info.file_size and (info.compress_size == 0 or info.file_size > MAX_COMPRESSION_RATIO * info.compress_size):
                raise ValueError(f"scene ZIP compression ratio exceeds {MAX_COMPRESSION_RATIO}:1 for {name}")
            if len(file_data) >= MAX_FILES:
                raise ValueError(f"scene ZIP may contain at most {MAX_FILES} files")
            chunks: list[bytes] = []
            actual_size = 0
            try:
                with archive.open(info, "r") as stream:
                    while True:
                        chunk = stream.read(64 * 1024)
                        if not chunk:
                            break
                        actual_size += len(chunk)
                        expanded_actual += len(chunk)
                        if actual_size > (MAX_SCENE_XML_BYTES if name == "scene.xml" else MAX_PLY_BYTES):
                            raise ValueError(f"{name} expanded beyond its per-file limit")
                        if expanded_actual > MAX_EXPANDED_BYTES:
                            raise ValueError(f"scene ZIP exceeds the {MAX_EXPANDED_BYTES}-byte expanded limit")
                        chunks.append(chunk)
            except (zipfile.BadZipFile, RuntimeError, EOFError, OSError, zlib.error) as exc:
                raise ValueError(f"scene ZIP member is corrupt or has an invalid CRC: {name}") from exc
            if actual_size != info.file_size:
                raise ValueError(f"scene ZIP member size does not match its directory entry: {name}")
            file_data[name] = b"".join(chunks)
    finally:
        archive.close()

    if "scene.xml" not in file_data:
        raise ValueError("scene ZIP must contain a root scene.xml")
    canonical, summaries = _validated_files(file_data, "scene.xml", canonicalize=True)
    hashes, bundle_hash = _bundle_identity(canonical)
    return ValidatedRtSceneBundle(
        scene_file="scene.xml",
        files=canonical,
        file_sha256=hashes,
        bundle_sha256=bundle_hash,
        mesh_summary=summaries,
    )


def _read_limited_regular_file(path: Path, limit: int, context: str) -> bytes:
    try:
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"{context} is not a regular file")
            if os.fstat(stream.fileno()).st_size > limit:
                raise ValueError(f"{context} exceeds its {limit}-byte read limit")
            content = stream.read(limit + 1)
    except OSError as exc:
        raise ValueError(f"cannot read {context}: {exc}") from exc
    if len(content) > limit:
        raise ValueError(f"{context} exceeds its {limit}-byte read limit")
    return content


def _builtin_shape_files(raw: bytes, scene_file: str) -> list[str]:
    if len(raw) > MAX_BUILTIN_SCENE_XML_BYTES:
        raise ValueError(f"{scene_file} exceeds the trusted built-in XML limit")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{scene_file} must be UTF-8 XML") from exc
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.IGNORECASE):
        raise ValueError("DOCTYPE and ENTITY declarations are not allowed in built-in scene XML")
    declaration = _XML_DECLARATION_RE.match(raw)
    if declaration:
        encoding = _ENCODING_RE.search(declaration.group(1))
        if encoding and encoding.group(2).decode("ascii", errors="ignore").lower() != "utf-8":
            raise ValueError("built-in scene XML declaration must use UTF-8")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"invalid built-in scene XML: {exc}") from exc
    if root.tag != "scene" or root.attrib != {"version": "2.1.0"}:
        raise ValueError('built-in scene XML root must be <scene version="2.1.0">')
    if root.text and root.text.strip():
        raise ValueError("built-in scene XML may contain only top-level bsdf and shape elements")

    shapes = []
    for element in root:
        if element.tag not in {"bsdf", "shape"}:
            raise ValueError(f"unsupported built-in scene element: <{element.tag}>")
        if element.tag == "shape":
            if element.attrib.get("type") != "ply":
                raise ValueError("built-in scene shapes must use PLY meshes")
            shapes.append(element)
    if not shapes or len(shapes) + 1 > MAX_BUILTIN_SCENE_FILES:
        raise ValueError("built-in scene has no PLY meshes or too many shapes")

    paths: set[str] = set()
    filename_nodes: set[int] = set()
    for shape in shapes:
        filename_elements = [
            child
            for child in shape.iter()
            if child.tag == "string" and child.attrib.get("name") == "filename"
        ]
        if len(filename_elements) != 1:
            raise ValueError("each built-in PLY shape must reference exactly one mesh file")
        filename_element = filename_elements[0]
        if filename_element.attrib.keys() != {"name", "value"} or len(filename_element):
            raise ValueError("built-in mesh references must contain only name and value")
        filename = filename_element.attrib["value"]
        if not _safe_relative_path(filename) or not _BUILTIN_PLY_PATH_RE.fullmatch(filename):
            raise ValueError(f"built-in mesh path must be meshes/<safe-id>.ply: {filename}")
        paths.add(filename)
        filename_nodes.add(id(filename_element))

    for element in root.iter():
        if element.tag.lower() == "include":
            raise ValueError("built-in scene XML may not include external XML")
        if "filename" in element.attrib:
            raise ValueError("built-in scene XML may not use filename attributes")
        if (
            element.tag == "string"
            and element.attrib.get("name") == "filename"
            and id(element) not in filename_nodes
        ):
            raise ValueError("built-in scene XML has an external filename reference")
    return sorted(paths)


def _check_builtin_root_assets(
    root: Path,
    scene_file: str,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if not _SCENE_FILE_RE.fullmatch(scene_file):
        raise ValueError("scene_file must be a safe root-level XML filename")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"scene root does not exist: {root}") from exc
    if not resolved_root.is_dir():
        raise ValueError("scene root must be a directory")
    scene_path = resolved_root / scene_file
    if scene_path.is_symlink() or not scene_path.is_file():
        raise ValueError(f"scene root is missing regular file {scene_file}")
    scene_bytes = _read_limited_regular_file(
        scene_path, MAX_BUILTIN_SCENE_XML_BYTES, scene_file
    )
    mesh_files = _builtin_shape_files(scene_bytes, scene_file)
    if len(mesh_files) + 1 > MAX_BUILTIN_SCENE_FILES:
        raise ValueError("built-in scene contains too many unique mesh files")
    files: dict[str, bytes] = {scene_file: scene_bytes}
    total_bytes = len(scene_bytes)
    mesh_parent = resolved_root / "meshes"
    if mesh_parent.is_symlink() or not mesh_parent.is_dir():
        raise ValueError("built-in scene is missing its meshes directory")
    try:
        resolved_mesh_parent = mesh_parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot resolve built-in meshes directory: {exc}") from exc
    for relative in mesh_files:
        mesh_path = resolved_root.joinpath(*PurePosixPath(relative).parts)
        if mesh_path.is_symlink() or not mesh_path.is_file():
            raise ValueError(f"built-in scene is missing regular file {relative}")
        try:
            resolved_mesh = mesh_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"cannot resolve built-in scene asset {relative}: {exc}") from exc
        if resolved_mesh.parent != resolved_mesh_parent:
            raise ValueError(f"built-in scene asset escapes its root: {relative}")
        mesh_content = _read_limited_regular_file(
            mesh_path,
            MAX_BUILTIN_SCENE_BYTES - total_bytes,
            relative,
        )
        total_bytes += len(mesh_content)
        if total_bytes > MAX_BUILTIN_SCENE_BYTES:
            raise ValueError("built-in scene exceeds its 32-MiB asset limit")
        files[relative] = mesh_content
    return files, {}
def _check_root_assets(root: Path, scene_file: str, *, allow_unreferenced: bool) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if not _SCENE_FILE_RE.fullmatch(scene_file):
        raise ValueError("scene_file must be a safe root-level XML filename")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"scene root does not exist: {root}") from exc
    if not resolved_root.is_dir():
        raise ValueError("scene root must be a directory")
    scene_path = resolved_root / scene_file
    if scene_path.is_symlink() or not scene_path.is_file():
        raise ValueError(f"scene root is missing regular file {scene_file}")
    scene_bytes = _read_limited_regular_file(scene_path, MAX_SCENE_XML_BYTES, scene_file)
    _, shapes, _ = _validate_xml(scene_bytes, scene_file)
    files: dict[str, bytes] = {scene_file: scene_bytes}
    total_bytes = len(scene_bytes)
    for shape in shapes:
        relative = shape["filename"]
        mesh_path = resolved_root.joinpath(*PurePosixPath(relative).parts)
        mesh_parent = resolved_root / "meshes"
        if mesh_parent.is_symlink() or not mesh_parent.is_dir() or mesh_path.is_symlink() or not mesh_path.is_file():
            raise ValueError(f"scene root is missing regular file {relative}")
        try:
            resolved_mesh = mesh_path.resolve(strict=True)
            resolved_mesh_parent = mesh_parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"cannot resolve scene asset {relative}: {exc}") from exc
        if resolved_mesh.parent != resolved_mesh_parent:
            raise ValueError(f"scene asset escapes its root: {relative}")
        mesh_content = _read_limited_regular_file(
            mesh_path, min(MAX_PLY_BYTES, MAX_EXPANDED_BYTES - total_bytes), relative
        )
        total_bytes += len(mesh_content)
        if total_bytes > MAX_EXPANDED_BYTES:
            raise ValueError(f"scene root exceeds the {MAX_EXPANDED_BYTES}-byte expanded limit")
        files[relative] = mesh_content
    validated, summaries = _validated_files(files, scene_file, canonicalize=False)
    if not allow_unreferenced:
        actual_files: set[str] = set()
        for current, dirs, names in os.walk(resolved_root, followlinks=False):
            current_path = Path(current)
            for directory in dirs:
                item = current_path / directory
                if item.is_symlink() or not item.is_dir() or (item.relative_to(resolved_root).as_posix() != "meshes"):
                    raise ValueError(f"scene root contains an unsupported directory: {item.relative_to(resolved_root)}")
            for name in names:
                item = current_path / name
                if item.is_symlink() or not item.is_file():
                    raise ValueError(f"scene root contains a symlink or special file: {item.relative_to(resolved_root)}")
                actual_files.add(item.relative_to(resolved_root).as_posix())
                if len(actual_files) > MAX_FILES:
                    raise ValueError(f"scene root may contain at most {MAX_FILES} files")
        if actual_files != set(validated):
            extra = sorted(actual_files - set(validated))
            missing = sorted(set(validated) - actual_files)
            if missing:
                raise ValueError(f"scene root is missing asset(s): {', '.join(missing)}")
            raise ValueError(f"scene root contains unreferenced or unsupported file(s): {', '.join(extra)}")
    return validated, summaries


_LOCAL_BUILTIN_SCENES = frozenset({"empty", "ground", "ground_wall"})


def _check_builtin_scene_assets(
    root: Path,
    scene_file: str,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if Path(scene_file).stem in _LOCAL_BUILTIN_SCENES:
        return _check_root_assets(root, scene_file, allow_unreferenced=True)
    return _check_builtin_root_assets(root, scene_file)


def resolve_scene_assets(
    root: str | Path,
    scene_file: str = "scene.xml",
    *,
    source: str = "imported",
) -> RtSceneAssets:
    """Validate and hash an explicit scene root under its source-specific policy."""
    source = _source_name(source)
    try:
        resolved_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"scene root does not exist: {root}") from exc
    if source == "builtin":
        files, _ = _check_builtin_scene_assets(resolved_root, scene_file)
    else:
        files, _ = _check_root_assets(resolved_root, scene_file, allow_unreferenced=False)
    hashes, bundle_hash = _bundle_identity(files)
    if source == "builtin":
        expected = resolve_builtin_scene_assets(Path(scene_file).stem)
        if hashes != expected.file_sha256 or bundle_hash != expected.bundle_sha256:
            raise ValueError("built-in scene root does not match the installed scene package")
    return RtSceneAssets(resolved_root, scene_file, hashes, bundle_hash, source)


def resolve_builtin_scene_assets(scene: str) -> RtSceneAssets:
    """Resolve and hash a local or pinned Sionna-RT packaged scene."""
    local_scenes = _LOCAL_BUILTIN_SCENES
    if scene in local_scenes:
        root = Path(__file__).resolve().parent / "rt_scenes"
        scene_file = f"{scene}.xml"
    elif scene in SIONNA_RT_SCENE_IDS:
        try:
            package_root = Path(
                distribution("sionna-rt").locate_file("sionna/rt/scenes")
            ).resolve(strict=True)
        except PackageNotFoundError as exc:
            raise ValueError("Sionna RT packaged scenes are unavailable") from exc
        except OSError as exc:
            raise ValueError(f"cannot resolve Sionna RT scene package: {exc}") from exc
        root = package_root / scene
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Sionna RT package is missing scene {scene}")
        try:
            resolved_scene_root = root.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"cannot resolve Sionna RT scene {scene}: {exc}") from exc
        if resolved_scene_root.parent != package_root:
            raise ValueError(f"Sionna RT scene escapes its package root: {scene}")
        root = resolved_scene_root
        scene_file = f"{scene}.xml"
    else:
        raise ValueError(f"unknown built-in RT scene: {scene}")

    resolved_root = root.resolve(strict=True)
    if scene in local_scenes:
        files, _ = _check_root_assets(resolved_root, scene_file, allow_unreferenced=True)
    else:
        files, _ = _check_builtin_root_assets(resolved_root, scene_file)
    hashes, bundle_hash = _bundle_identity(files)
    return RtSceneAssets(resolved_root, scene_file, hashes, bundle_hash, "builtin")




def persist_scene_bundle(
    bundle: ValidatedRtSceneBundle,
    root: str | Path,
    *,
    source: str = "imported",
) -> RtSceneAssets:
    """Atomically persist a validated bundle under a new destination directory."""
    source = _source_name(source)
    checked, _ = _validated_files(bundle.files, bundle.scene_file, canonicalize=True)
    hashes, bundle_hash = _bundle_identity(checked)
    if hashes != bundle.file_sha256 or bundle_hash != bundle.bundle_sha256:
        raise ValueError("validated scene bundle identity does not match its contents")
    destination = Path(root).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"scene destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".rt-scene-", dir=parent))
    try:
        os.chmod(temporary, 0o700)
        for relative, content in checked.items():
            target = temporary.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(target.parent, 0o700)
            with target.open("xb") as stream:
                stream.write(content)
            os.chmod(target, 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"scene destination already exists: {destination}")
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return resolve_scene_assets(destination, bundle.scene_file, source=source)
def copy_scene_assets(assets: RtSceneAssets, root: str | Path) -> RtSceneAssets:
    """Copy one resolved scene to a job-owned root without changing asset identity."""
    resolved_source = Path(assets.root).resolve(strict=True)
    if assets.source == "builtin":
        files, _ = _check_builtin_scene_assets(resolved_source, assets.scene_file)
    else:
        files, _ = _check_root_assets(
            resolved_source,
            assets.scene_file,
            allow_unreferenced=False,
        )
    hashes, bundle_hash = _bundle_identity(files)
    if hashes != assets.file_sha256 or bundle_hash != assets.bundle_sha256:
        raise ValueError("scene assets changed after resolution")
    destination = Path(root).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"scene destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".rt-scene-copy-", dir=parent))
    try:
        os.chmod(temporary, 0o700)
        for relative, content in files.items():
            target = temporary.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(target.parent, 0o700)
            with target.open("xb") as stream:
                stream.write(content)
            os.chmod(target, 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"scene destination already exists: {destination}")
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    copied = resolve_scene_assets(destination, assets.scene_file, source=assets.source)
    if copied.file_sha256 != hashes or copied.bundle_sha256 != bundle_hash:
        shutil.rmtree(destination, ignore_errors=True)
        raise ValueError("copied scene assets changed identity")
    return copied




def _geometry_value(geometry: Any, name: str, default: Any = None) -> Any:
    if isinstance(geometry, Mapping):
        return geometry.get(name, default)
    return getattr(geometry, name, default)


def _geometry_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"geometry.{name} must contain {length} finite numbers")
    return tuple(_finite_float(component, f"geometry.{name}") for component in value)


def _check_geometry_coordinate(number: float, name: str) -> float:
    if abs(number) > MAX_COORDINATE_M:
        raise ValueError(f"geometry.{name} must have absolute value at most {MAX_COORDINATE_M:g} m")
    return number


def build_parameterized_scene_bundle(scene: str, geometry: Any) -> ValidatedRtSceneBundle:
    """Generate canonical ground/ground-wall XML and PLY assets without writing them.

    ``geometry`` may be the frozen ``RtGeometrySettings`` object or its dictionary
    representation. The caller remains responsible for validating whether the
    selected RT config is allowed to carry geometry.
    """
    if scene not in {"ground", "ground_wall"}:
        raise ValueError("parameterized geometry is supported only for ground and ground_wall scenes")
    defaults: dict[str, Any] = {
        "ground_bounds_m": [-500.0, 500.0, -500.0, 500.0],
        "ground_height_m": 0.0,
        "ground_material": "concrete",
        "ground_thickness_m": 0.1,
        "wall_start_xy_m": [-100.0, 150.0],
        "wall_end_xy_m": [500.0, 150.0],
        "wall_base_height_m": 0.0,
        "wall_height_m": 50.0,
        "wall_material": "brick",
        "wall_thickness_m": 0.1,
    }
    value = lambda key: _geometry_value(geometry, key, defaults[key])

    xmin, xmax, ymin, ymax = _geometry_vector(value("ground_bounds_m"), 4, "ground_bounds_m")
    ground_height = _finite_float(value("ground_height_m"), "geometry.ground_height_m")
    ground_material = value("ground_material")
    ground_thickness = _finite_float(value("ground_thickness_m"), "geometry.ground_thickness_m")
    for name, number in (("ground_bounds_m", xmin), ("ground_bounds_m", xmax), ("ground_bounds_m", ymin), ("ground_bounds_m", ymax), ("ground_height_m", ground_height)):
        _check_geometry_coordinate(number, name)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("geometry.ground_bounds_m must be strictly increasing")
    if ground_material not in {"concrete", "brick"}:
        raise ValueError("geometry.ground_material must be concrete or brick")
    if ground_thickness <= 0.0 or ground_thickness > 10.0:
        raise ValueError("geometry.ground_thickness_m must be in (0,10] m")

    vertices: dict[str, list[tuple[float, float, float]]] = {
        "ground": [
            (xmin, ymin, ground_height),
            (xmax, ymin, ground_height),
            (xmax, ymax, ground_height),
            (xmin, ymax, ground_height),
        ]
    }
    faces: dict[str, list[tuple[int, int, int]]] = {"ground": [(0, 1, 2), (0, 2, 3)]}
    materials: list[tuple[str, str, float]] = [(f"ground-{ground_material}", ground_material, ground_thickness)]
    if scene == "ground_wall":
        start_x, start_y = _geometry_vector(value("wall_start_xy_m"), 2, "wall_start_xy_m")
        end_x, end_y = _geometry_vector(value("wall_end_xy_m"), 2, "wall_end_xy_m")
        base_height = _finite_float(value("wall_base_height_m"), "geometry.wall_base_height_m")
        wall_height = _finite_float(value("wall_height_m"), "geometry.wall_height_m")
        wall_material = value("wall_material")
        wall_thickness = _finite_float(value("wall_thickness_m"), "geometry.wall_thickness_m")
        for name, number in (
            ("wall_start_xy_m", start_x), ("wall_start_xy_m", start_y),
            ("wall_end_xy_m", end_x), ("wall_end_xy_m", end_y),
            ("wall_base_height_m", base_height), ("wall_height_m", base_height + wall_height),
        ):
            _check_geometry_coordinate(number, name)
        if (start_x, start_y) == (end_x, end_y):
            raise ValueError("geometry.wall_start_xy_m and wall_end_xy_m must differ")
        if wall_height <= 0.0:
            raise ValueError("geometry.wall_height_m must be positive")
        if wall_material not in {"concrete", "brick"}:
            raise ValueError("geometry.wall_material must be concrete or brick")
        if wall_thickness <= 0.0 or wall_thickness > 10.0:
            raise ValueError("geometry.wall_thickness_m must be in (0,10] m")
        top_height = base_height + wall_height
        vertices["wall"] = [
            (start_x, start_y, base_height),
            (end_x, end_y, base_height),
            (end_x, end_y, top_height),
            (start_x, start_y, top_height),
        ]
        faces["wall"] = [(0, 1, 2), (0, 2, 3)]
        materials.append((f"wall-{wall_material}", wall_material, wall_thickness))

    files: dict[str, bytes] = {}
    xml: list[str] = ['<scene version="2.1.0">']
    for identifier, material_type, thickness in materials:
        xml.extend(
            [
                f'  <bsdf type="itu-radio-material" id="{identifier}">',
                f'    <string name="type" value="{material_type}"/>',
                f'    <float name="thickness" value="{_float_text(thickness)}"/>',
                '    <float name="scattering_coefficient" value="0.0"/>',
                "  </bsdf>",
            ]
        )
    for shape in vertices:
        xml.extend(
            [
                f'  <shape type="ply" id="{shape}">',
                f'    <string name="filename" value="meshes/{shape}.ply"/>',
                '    <boolean name="face_normals" value="true"/>',
                f'    <ref id="{shape}-{ground_material if shape == "ground" else wall_material}" name="bsdf"/>',
                "  </shape>",
            ]
        )
        ply = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices[shape])}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(faces[shape])}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        ply.extend(" ".join(_float_text(component) for component in point) for point in vertices[shape])
        ply.extend("3 " + " ".join(str(index) for index in face) for face in faces[shape])
        files[f"meshes/{shape}.ply"] = ("\n".join(ply) + "\n").encode("ascii")
    xml.append("</scene>")
    files["scene.xml"] = ("\n".join(xml) + "\n").encode("utf-8")
    canonical, summaries = _validated_files(files, "scene.xml", canonicalize=True)
    hashes, bundle_hash = _bundle_identity(canonical)
    return ValidatedRtSceneBundle("scene.xml", canonical, hashes, bundle_hash, summaries)


def generate_parameterized_scene_assets(
    scene: str,
    geometry: Any,
    root: str | Path,
) -> RtSceneAssets:
    """Generate parameterized assets and atomically persist them under ``root``."""
    return persist_scene_bundle(
        build_parameterized_scene_bundle(scene, geometry), root, source="parameterized"
    )


def make_scene_template_zip() -> bytes:
    """Build a small ZIP template from the packaged ground-plus-wall scene."""
    assets = resolve_builtin_scene_assets("ground_wall")
    files = {
        ("scene.xml" if path == assets.scene_file else path): (assets.root / Path(path)).read_bytes()
        for path in assets.file_sha256
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, files[name])
    return buffer.getvalue()
