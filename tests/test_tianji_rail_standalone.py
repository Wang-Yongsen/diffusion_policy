"""Bundled resources and recording archives remain usable after relocation."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from diffusion_policy.third_party import bimanual_teleop
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG as ARM_CONFIG
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import DEFAULT_CONFIG as HAND_CONFIG
from diffusion_policy.third_party.bimanual_teleop.recording.config import DEFAULT_CONFIG as CAMERA_CONFIG
from diffusion_policy.third_party.bimanual_teleop.recording.convert import convert_recordings
from diffusion_policy.third_party.bimanual_teleop.recording.finalize import archive_spool, _restore_spool
from diffusion_policy.recording.convert import convert_recordings as public_convert


class StandaloneTests(unittest.TestCase):
    def test_public_converter_and_default_configs_are_bundled(self):
        self.assertIs(public_convert, convert_recordings)
        self.assertTrue(public_convert.__module__.startswith('diffusion_policy.third_party.'))
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

    def test_archived_spool_survives_moving_session_and_archive_together(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'original'
            episode = bundle / 'recordings/session/episode'
            spool = episode / 'raw_spool'
            spool.mkdir(parents=True)
            (spool / 'payload').write_bytes(b'raw recording')
            archive = bundle / 'raw_spools'
            self.assertTrue(archive_spool(episode, archive))
            self.assertFalse(spool.readlink().is_absolute())
            moved = Path(directory) / 'another_location'
            shutil.move(bundle, moved)
            relocated = moved / 'recordings/session/episode/raw_spool'
            self.assertEqual((relocated / 'payload').read_bytes(), b'raw recording')
            self.assertFalse(archive_spool(relocated.parent, moved / 'raw_spools'))

    def test_restored_archive_link_is_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'recordings/session/episode'
            episode.mkdir(parents=True)
            archive = root / 'raw_spools'
            target = archive / 'session/episode'
            target.mkdir(parents=True)
            (episode / 'raw_spool').symlink_to('/missing/old/project/raw_spools/session/episode')
            _restore_spool(episode, archive)
            link = episode / 'raw_spool'
            self.assertEqual(link.resolve(), target.resolve())
            self.assertFalse(link.readlink().is_absolute())


if __name__ == '__main__':
    unittest.main()
