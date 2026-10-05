"""将 raw_spool 整理为 MP4、raw.zarr 和 episode.json。"""

import argparse
from pathlib import Path

from diffusion_policy.third_party.bimanual_teleop.common.console import EpisodeProgress
from diffusion_policy.recording.finalize import finalize_recordings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sdk-root", type=Path, help="省略时使用 bimanual 包内 SDK")
    parser.add_argument("--spool-archive", type=Path, default=Path("raw_spools"),
                        help="成功后归档原始分段并留下符号链接；不能位于 input 内")
    parser.add_argument("--refinalize", action="store_true",
                        help="显式重新整理已完成条目，覆盖它的派生产物")
    args = parser.parse_args(argv)
    with EpisodeProgress("整理条目") as progress:
        report = finalize_recordings(args.input, sdk_root=args.sdk_root,
                                    spool_archive=args.spool_archive,
                                    refinalize=args.refinalize, progress=progress)
    print(f"完成 {report['complete']}，作废删除 {report['discarded']}，"
          f"跳过 {report['skipped']}，失败 {report['failed']}")
    for path, error in report["errors"]:
        print(f"{path}: {error}")
    return int(bool(report["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
