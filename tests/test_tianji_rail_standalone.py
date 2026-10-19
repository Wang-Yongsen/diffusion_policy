"""Real-world evaluation configurations and bundled runtime resources."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

from diffusion_policy.third_party import bimanual_teleop
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG as ARM_CONFIG
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import DEFAULT_CONFIG as HAND_CONFIG
from diffusion_policy.third_party.bimanual_teleop.recording.camera import CameraRig
from diffusion_policy.third_party.bimanual_teleop.recording.config import (
    DEFAULT_CONFIG as CAMERA_CONFIG, load_config as load_camera_config)


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

    def test_default_cameras_follow_main_left_right_recording_order(self):
        self.assertEqual(load_camera_config().cameras, ('405622076394', '408322072104', '408322070938'))

    def test_only_left_arm_rgb_is_rotated_180_degrees(self):
        rig = CameraRig(load_camera_config(), rs=NS(timestamp_domain=NS(global_time='global')))
        rig._np, rig._bridge = np, NS(map=lambda ms: round(ms * 1_000_000))
        rig._active.set()
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        for name in ('camera_0', 'camera_1', 'camera_2'):
            rig._accept(name, 'rgb', NS(get_frame_timestamp_domain=lambda: 'global', get_timestamp=lambda: 1.,
                                        get_frame_number=lambda: 1, get_data=lambda: image))
        frames = {name: pixels for name, kind, pixels, _record in rig.poll()}
        self.assertEqual(set(frames), {'camera_0', 'camera_1', 'camera_2'})
        np.testing.assert_array_equal(frames['camera_0'], image)
        np.testing.assert_array_equal(frames['camera_1'], image[::-1, ::-1])
        np.testing.assert_array_equal(frames['camera_2'], image)
        self.assertTrue(all(pixels.flags.c_contiguous for pixels in frames.values()))

if __name__ == '__main__':
    unittest.main()
