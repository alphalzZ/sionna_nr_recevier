import os
from importlib import resources
from pathlib import Path
import tempfile
import unittest

from sionna.rt import load_scene


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


if __name__ == "__main__":
    unittest.main()
