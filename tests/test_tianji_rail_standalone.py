"""Real-world evaluation configurations and bundled runtime resources."""

import hashlib
import json
from pathlib import Path
import unittest

from diffusion_policy.third_party import bimanual_teleop
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG as ARM_CONFIG
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import DEFAULT_CONFIG as HAND_CONFIG
from diffusion_policy.third_party.bimanual_teleop.recording.config import DEFAULT_CONFIG as CAMERA_CONFIG


class StandaloneTests(unittest.TestCase):
    def test_runtime_default_configs_are_bundled(self):
        for config in (ARM_CONFIG, HAND_CONFIG, CAMERA_CONFIG):
            self.assertTrue(config.is_file(), config)
            self.assertEqual(config.parent.name, 'config')
            self.assertEqual(config.parent.parent.name, 'diffusion_policy')

    def test_snapshot_files_match_manifest_and_are_regular_files(self):
        vendor = Path(bimanual_teleop.__file__).parent.parent
        manifest = json.loads((vendor / 'bimanual_teleop/manifest.json').read_text())
        for item in manifest['files']:
            with self.subTest(path=item['path']):
                path = vendor / item['path']
                self.assertFalse(path.is_symlink())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['vendored_sha256'])
                self.assertEqual(len(item['source_sha256']), 64)

if __name__ == '__main__':
    unittest.main()
