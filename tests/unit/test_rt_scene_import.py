from __future__ import annotations

import base64
from io import BytesIO, StringIO
import random
import struct
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import (
    build_parameterized_scene_bundle,
    copy_scene_assets,
    make_scene_template_zip,
    parse_scene_bundle_zip,
    persist_scene_bundle,
    resolve_builtin_scene_assets,
    resolve_scene_assets,
)
from nr_pusch.web import ApiError, SimulationWebApp


class RtSceneImportTest(unittest.TestCase):
    @staticmethod
    def _template_files() -> dict[str, bytes]:
        with zipfile.ZipFile(BytesIO(make_scene_template_zip())) as archive:
            return {name: archive.read(name) for name in archive.namelist()}

    @staticmethod
    def _zip(
        files: list[tuple[str, bytes, int | None]],
        *,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> bytes:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
            for name, content, mode in files:
                info = zipfile.ZipInfo(name)
                info.compress_type = compression
                if mode is not None:
                    info.create_system = 3
                    info.external_attr = mode << 16
                archive.writestr(info, content)
        return buffer.getvalue()

    def test_template_is_valid_and_scene_copy_preserves_identity(self):
        bundle = parse_scene_bundle_zip(make_scene_template_zip())
        self.assertEqual(bundle.scene_file, "scene.xml")
        self.assertEqual(set(bundle.files), {"scene.xml", "meshes/ground.ply", "meshes/wall.ply"})
        self.assertEqual(bundle.mesh_summary["meshes/ground.ply"]["vertex_count"], 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stored = persist_scene_bundle(bundle, root / "uploaded")
            job_copy = copy_scene_assets(stored, root / "job-scene")
            self.assertEqual(job_copy.bundle_sha256, stored.bundle_sha256)
            self.assertEqual(job_copy.file_sha256, stored.file_sha256)
            (stored.root / "meshes" / "ground.ply").write_bytes(b"modified after copy")
            replay = resolve_scene_assets(job_copy.root, job_copy.scene_file)
            self.assertEqual(replay.bundle_sha256, stored.bundle_sha256)

    def test_default_parameterized_geometry_keeps_ground_wall_topology(self):
        settings = RtBeamSettings.from_toml(
            Path(__file__).resolve().parents[2] / "configs" / "rt_beam_ground_wall.toml"
        )
        bundle = build_parameterized_scene_bundle("ground_wall", settings.geometry or {})
        original = resolve_builtin_scene_assets("ground_wall")
        self.assertEqual(bundle.mesh_summary["meshes/ground.ply"]["bounds_m"], {
            "min": [-500.0, -500.0, 0.0], "max": [500.0, 500.0, 0.0]
        })
        self.assertEqual(bundle.mesh_summary["meshes/wall.ply"]["bounds_xy_m"], {
            "min": [-100.0, 150.0], "max": [500.0, 150.0]
        })
        self.assertNotEqual(bundle.bundle_sha256, original.bundle_sha256)

    def test_rejects_xml_plugin_url_dtd_and_unknown_attributes(self):
        files = self._template_files()
        xml = files["scene.xml"]
        invalid_xml = (
            (xml.replace(b"<scene version=", b"<!DOCTYPE scene [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><scene version="), "DOCTYPE"),
            (xml.replace(b"</scene>", b"  <include filename=\"external.xml\"/>\n</scene>"), "unsupported scene element"),
            (xml.replace(b"meshes/ground.ply", b"https://example.invalid/ground.ply"), "filename"),
            (xml.replace(b'<scene version="2.1.0">', b'<scene version="2.1.0" extra="1">'), "extra attributes"),
        )
        for modified, message in invalid_xml:
            with self.subTest(message=message):
                candidate = dict(files)
                candidate["scene.xml"] = modified
                with self.assertRaisesRegex(ValueError, message):
                    parse_scene_bundle_zip(self._zip([(name, data, None) for name, data in candidate.items()]))

    def test_rejects_xml_id_limits_nul_member_names_and_bad_crc(self):
        files = self._template_files()
        xml = files["scene.xml"]
        material = (
            '<bsdf type="itu-radio-material" id="extra">'
            '<string name="type" value="concrete"/>'
            '<float name="thickness" value="0.1"/>'
            '<float name="scattering_coefficient" value="0.0"/>'
            "</bsdf>"
        )
        too_many_materials = xml.replace(
            b"</scene>",
            "".join(
                material.replace('id="extra"', f'id="extra-{index}"')
                for index in range(31)
            ).encode("ascii") + b"</scene>",
        )
        with self.assertRaisesRegex(ValueError, "at most 32 materials"):
            candidate = dict(files)
            candidate["scene.xml"] = too_many_materials
            parse_scene_bundle_zip(self._zip([
                (name, data, None) for name, data in candidate.items()
            ]))

        extra_shapes = "".join(
            f'<shape type="ply" id="extra-{index}">'
            f'<string name="filename" value="meshes/extra-{index}.ply"/>'
            '<boolean name="face_normals" value="true"/>'
            '<ref id="ground-concrete" name="bsdf"/>'
            "</shape>"
            for index in range(31)
        ).encode("ascii")
        too_many_shapes = xml.replace(b"</scene>", extra_shapes + b"</scene>")
        with self.assertRaisesRegex(ValueError, "at most 32 shapes"):
            candidate = dict(files)
            candidate["scene.xml"] = too_many_shapes
            parse_scene_bundle_zip(self._zip([
                (name, data, None) for name, data in candidate.items()
            ]))

        nul_member = self._zip([
            (name, data, None) for name, data in files.items()
        ]).replace(b"scene.xml", b"scene\x00xml")
        with self.assertRaisesRegex(ValueError, "NUL"):
            parse_scene_bundle_zip(nul_member)

        corrupted = bytearray(self._zip([
            (name, data, None) for name, data in files.items()
        ]))
        with zipfile.ZipFile(BytesIO(corrupted)) as archive:
            info = archive.getinfo("scene.xml")
        name_size, extra_size = struct.unpack_from(
            "<HH", corrupted, info.header_offset + 26
        )
        data_start = info.header_offset + 30 + name_size + extra_size
        corrupted[data_start] ^= 0x80
        with self.assertRaisesRegex(ValueError, "corrupt or has an invalid CRC"):
            parse_scene_bundle_zip(corrupted)

    def test_rejects_unsafe_duplicate_symlink_and_invalid_ply_members(self):
        files = self._template_files()
        cases = (
            (
                [("../escape", b"x", None), *((name, data, None) for name, data in files.items())],
                "unsafe scene ZIP path",
            ),
            (
                [("/absolute.xml", b"x", None), *((name, data, None) for name, data in files.items())],
                "unsafe scene ZIP path",
            ),
            (
                [("meshes\\escape.ply", b"x", None), *((name, data, None) for name, data in files.items())],
                "unsafe scene ZIP path",
            ),
            (
                [("meshes/GROUND.ply", files["meshes/ground.ply"], None), *((name, data, None) for name, data in files.items())],
                "case-colliding",
            ),
            (
                [("meshes/link.ply", b"meshes/ground.ply", stat.S_IFLNK | 0o777), *((name, data, None) for name, data in files.items())],
                "symlink or special file",
            ),
            (
                [(name, data.replace(b"-500 -500 0", b"nan -500 0") if name == "meshes/ground.ply" else data, None) for name, data in files.items()],
                "finite number",
            ),
            (
                [(name, data.replace(b"3 0 1 2", b"3 0 4 2") if name == "meshes/ground.ply" else data, None) for name, data in files.items()],
                "out-of-range",
            ),
            (
                [
                    (
                        name,
                        data.replace(b"-500 -500 0", b"10001 -500 0", 1)
                        if name == "meshes/ground.ply" else data,
                        None,
                    )
                    for name, data in files.items()
                ],
                "absolute value at most",
            ),
        )
        for entries, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_scene_bundle_zip(self._zip(entries))

    def test_rejects_compression_bomb_and_duplicate_member(self):
        files = self._template_files()
        duplicate = [(name, data, None) for name, data in files.items()]
        duplicate.append(("scene.xml", files["scene.xml"], None))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaisesRegex(ValueError, "duplicate or case-colliding"):
                parse_scene_bundle_zip(self._zip(duplicate))

        vertex_count = 20_000
        ply = StringIO()
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {vertex_count}\nproperty float x\nproperty float y\nproperty float z\n")
        ply.write("element face 2\nproperty list uchar int vertex_indices\nend_header\n")
        ply.write("0 0 0\n1 0 0\n1 1 0\n0 1 0\n")
        ply.write("0 0 0\n" * (vertex_count - 4))
        ply.write("3 0 1 2\n3 0 2 3\n")
        bomb_files = dict(files)
        bomb_files["meshes/ground.ply"] = ply.getvalue().encode("ascii")
        with self.assertRaisesRegex(ValueError, "compression ratio exceeds"):
            parse_scene_bundle_zip(self._zip([(name, data, None) for name, data in bomb_files.items()]))

    def test_rejects_zip_limits_encryption_and_unsupported_compression(self):
        files = self._template_files()
        entries = [(name, data, None) for name, data in files.items()]
        with self.assertRaisesRegex(ValueError, "compressed limit"):
            parse_scene_bundle_zip(b"x" * (8 * 1024 * 1024 + 1))

        with self.assertRaisesRegex(ValueError, "unsupported ZIP compression"):
            parse_scene_bundle_zip(self._zip(entries, compression=zipfile.ZIP_BZIP2))

        encrypted = bytearray(self._zip(entries))
        local_header = encrypted.find(b"PK\x03\x04")
        central_header = encrypted.find(b"PK\x01\x02")
        self.assertGreaterEqual(local_header, 0)
        self.assertGreaterEqual(central_header, 0)
        encrypted[local_header + 6] |= 1
        encrypted[central_header + 8] |= 1
        with self.assertRaisesRegex(ValueError, "encrypted"):
            parse_scene_bundle_zip(encrypted)

        too_many = [("scene.xml", files["scene.xml"], None)]
        too_many.extend((f"meshes/m{index}.ply", b"unused", None) for index in range(34))
        with self.assertRaisesRegex(ValueError, "at most 33 files"):
            parse_scene_bundle_zip(self._zip(too_many))

        with self.assertRaisesRegex(ValueError, "XML limit"):
            oversized_xml = dict(files)
            oversized_xml["scene.xml"] += b" " * (256 * 1024)
            parse_scene_bundle_zip(self._zip([
                (name, data, None) for name, data in oversized_xml.items()
            ]))

        with self.assertRaisesRegex(ValueError, "PLY limit"):
            oversized_ply = dict(files)
            oversized_ply["meshes/ground.ply"] += b"comment " + b"x" * (8 * 1024 * 1024)
            parse_scene_bundle_zip(self._zip([
                (name, data, None) for name, data in oversized_ply.items()
            ]))

    def test_rejects_invalid_triangle_trailing_ply_and_bad_material(self):
        files = self._template_files()
        invalid = (
            (
                "meshes/ground.ply",
                files["meshes/ground.ply"].replace(b"3 0 1 2", b"3 0 1 1", 1),
                "degenerate triangle",
            ),
            (
                "meshes/ground.ply",
                files["meshes/ground.ply"] + b"unexpected trailing row\n",
                "trailing data",
            ),
            (
                "scene.xml",
                files["scene.xml"].replace(
                    b'<float name="scattering_coefficient" value="0.0"/>',
                    b'<float name="scattering_coefficient" value="0.1"/>',
                    1,
                ),
                "must be 0",
            ),
        )
        for path, content, message in invalid:
            with self.subTest(path=path, message=message):
                candidate = dict(files)
                candidate[path] = content
                with self.assertRaisesRegex(ValueError, message):
                    parse_scene_bundle_zip(self._zip([
                        (name, data, None) for name, data in candidate.items()
                    ]))

    def test_expanded_limit_is_enforced_while_streaming_zip_members(self):
        material = (
            '<bsdf type="itu-radio-material" id="m">'
            '<string name="type" value="concrete"/>'
            '<float name="thickness" value="0.1"/>'
            '<float name="scattering_coefficient" value="0.0"/>'
            "</bsdf>"
        )
        shapes = "".join(
            f'<shape type="ply" id="{name}">'
            f'<string name="filename" value="meshes/{name}.ply"/>'
            '<boolean name="face_normals" value="true"/>'
            '<ref id="m" name="bsdf"/>'
            "</shape>"
            for name in "abcd"
        )
        scene = f'<scene version="2.1.0">{material}{shapes}</scene>'.encode()
        prefix = b"ply\nformat ascii 1.0\ncomment "
        suffix = (
            b"\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            b"element face 2\nproperty list uchar int vertex_indices\nend_header\n"
            b"0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2\n3 0 2 3\n"
        )
        mesh_size = 8 * 1024 * 1024
        random_bits = random.Random(13)
        comment = bytes(65 + random_bits.getrandbits(1) for _ in range(mesh_size - len(prefix) - len(suffix)))
        mesh = prefix + comment + suffix
        self.assertEqual(len(mesh), mesh_size)
        entries = [("scene.xml", scene, None)]
        entries.extend((f"meshes/{name}.ply", mesh, None) for name in "abcd")
        archive = self._zip(entries)
        self.assertLess(len(archive), 8 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError, "expanded limit"):
            parse_scene_bundle_zip(archive)

    def test_upload_endpoint_persists_only_valid_scene_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = SimulationWebApp(
                Path(__file__).resolve().parents[2] / "configs", root / "runs"
            )
            try:
                payload = make_scene_template_zip()
                uploaded = app.upload_scene({
                    "filename": "template.zip",
                    "content_base64": base64.b64encode(payload).decode("ascii"),
                })
                self.assertEqual(len(uploaded["scene_id"]), 12)
                assets = resolve_scene_assets(app.scene_store_dir / uploaded["scene_id"])
                self.assertEqual(assets.bundle_sha256, uploaded["bundle_sha256"])
                before = sorted(path.name for path in app.scene_store_dir.iterdir())
                with self.assertRaises(ApiError):
                    app.upload_scene({"filename": "bad.zip", "content_base64": "bm90LXppcA=="})
                self.assertEqual(sorted(path.name for path in app.scene_store_dir.iterdir()), before)
            finally:
                app.shutdown()


if __name__ == "__main__":
    unittest.main()
