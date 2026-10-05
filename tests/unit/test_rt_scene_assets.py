import gc
import os
from importlib import resources
from pathlib import Path
import tempfile
import unittest

from sionna.rt import load_scene

from nr_pusch.rt_scene_assets import (
    BUILTIN_SCENE_IDS,
    SIONNA_RT_SCENE_IDS,
    builtin_scene_catalog,
    copy_scene_assets,
    resolve_builtin_scene_assets,
)

class RtSceneAssetsTest(unittest.TestCase):
    def test_scene_assets_resolve_outside_repository_cwd(self):
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                package = resources.files("nr_pusch").joinpath("rt_scenes")
                for filename in ("empty.xml", "ground.xml", "ground_wall.xml"):
                    with self.subTest(scene=filename):
                        scene_path = package.joinpath(filename)
                        self.assertTrue(scene_path.is_file())
                        scene = load_scene(str(scene_path))
                        scene.frequency = 3.5e9
                        self.assertAlmostEqual(
                            float(scene.wavelength.numpy()[0]),
                            299_792_458.0 / 3.5e9,
                            places=7,
                        )
                for filename in ("ground.ply", "wall.ply"):
                    self.assertTrue(package.joinpath("meshes", filename).is_file())
        finally:
            os.chdir(original_cwd)

    def test_resolves_copies_and_loads_every_sionna_packaged_scene(self):
        presets = builtin_scene_catalog()
        self.assertEqual({item["id"] for item in presets}, BUILTIN_SCENE_IDS)
        self.assertEqual(
            {item["id"] for item in presets if item["id"] not in {"empty", "ground", "ground_wall"}},
            set(SIONNA_RT_SCENE_IDS),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene_id in sorted(SIONNA_RT_SCENE_IDS):
                with self.subTest(scene=scene_id):
                    assets = resolve_builtin_scene_assets(scene_id)
                    self.assertEqual(assets.scene_file, f"{scene_id}.xml")
                    copied = copy_scene_assets(assets, root / scene_id)
                    self.assertEqual(copied.file_sha256, assets.file_sha256)
                    self.assertEqual(copied.bundle_sha256, assets.bundle_sha256)
                    scene = load_scene(str(copied.root / copied.scene_file))
                    self.assertIsNotNone(scene)
                    del scene
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
