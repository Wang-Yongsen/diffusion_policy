"""Extract a few episodes from a converted Zarr, preserving original split provenance.

Can run on a remote host through ``ssh ... python - ... < this_file``.
The original dataset is opened read-only; outputs must not already exist.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket

import numpy as np
import zarr


def export_episodes(dataset_path, output_dir, episodes, prefix=None):
    dataset_path, output_dir = (
        Path(dataset_path).expanduser().resolve(),
        Path(output_dir).expanduser().resolve(),
    )
    if output_dir == dataset_path or dataset_path in output_dir.parents:
        raise ValueError("Output must be outside the original read-only dataset")
    if (output_dir / "episodes_manifest.json").exists():
        raise FileExistsError(output_dir / "episodes_manifest.json")
    source = zarr.open_group(str(dataset_path), mode="r")
    data, meta = source["data"], source["meta"]
    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    segment_ends = np.asarray(meta["segment_ends"][:], dtype=np.int64)
    described = meta.attrs.get("segments", [])
    if len(described) != len(segment_ends):
        raise ValueError("Export requires source_episode metadata for every continuous segment")
    sources = list(dict.fromkeys(item["source_episode"] for item in described))
    episode_ids = np.searchsorted(ends, segment_ends)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "attrs": dict(source.attrs),
                "segments": described,
                "episode_ends": ends.tolist(),
                "segment_ends": segment_ends.tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    selected = []
    for episode in episodes:
        if str(episode).isdigit():
            index = int(episode)
            if not 0 <= index < len(ends):
                raise ValueError(f"No global episode index {index}")
        else:
            matching = {
                int(episode_ids[i])
                for i, item in enumerate(described)
                if item["source_episode"] == episode
            }
            if len(matching) != 1:
                raise ValueError(
                    f"Expected one original episode for {episode}, found {len(matching)}"
                )
            index = matching.pop()
        if index in selected:
            raise ValueError(f"Duplicate episode {episode}")
        selected.append(index)
    prefix = prefix or dataset_path.stem
    if Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("prefix must be a plain filename component")
    outputs = []
    for index in selected:
        low, high = (0 if index == 0 else int(ends[index - 1])), int(ends[index])
        segments = np.flatnonzero(episode_ids == index).tolist()
        names = {described[i]["source_episode"] for i in segments}
        if len(names) != 1:
            raise ValueError("Segments in one episode must share source_episode")
        name = names.pop()
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("source_episode must be a plain filename component")
        destination = output_dir / f"{prefix}_{name}.zarr"
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(destination), mode="w-")
        root.attrs.update(dict(source.attrs))
        provenance = {
            "version": 1,
            "complete": False,
            "source_dataset": str(dataset_path),
            "source_host": socket.gethostname(),
            "source_dataset_metadata_sha256": fingerprint,
            "source_episodes": sources,
            "source_episode": name,
            "source_episode_index": index,
            "source_start_frame": low,
            "source_end_frame": high,
            "source_segment_indices": segments,
        }
        root.attrs["episode_export"] = provenance
        target_data, target_meta = root.create_group("data"), root.create_group("meta")
        for key in data.array_keys():
            array = data[key]
            if array.ndim < 1 or array.shape[0] != int(ends[-1]):
                raise ValueError(f"{key} is not a frame-aligned array")
            array_meta = json.loads((dataset_path / array.path / ".zarray").read_text())
            separator = array_meta.get("dimension_separator", ".")
            target = target_data.create_dataset(
                key,
                shape=(high - low, *array.shape[1:]),
                chunks=array.chunks,
                dtype=array.dtype,
                compressor=array.compressor,
                filters=array.filters,
                fill_value=array.fill_value,
                order=array.order,
                dimension_separator=separator,
            )
            target.attrs.update(dict(array.attrs))
            # Converted RGB uses one complete image per chunk. Copy encoded bytes
            # without decoding/re-encoding so cached local RGB can be reused by rsync.
            if array.chunks[0] == 1 and tuple(array.chunks[1:]) == tuple(array.shape[1:]):
                for offset, frame in enumerate(range(low, high)):
                    src_key = separator.join(map(str, (frame, *([0] * (array.ndim - 1)))))
                    dst_key = separator.join(map(str, (offset, *([0] * (array.ndim - 1)))))
                    original = dataset_path / array.path / src_key
                    if original.exists():
                        copied = destination / target.path / dst_key
                        copied.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, copied)
            else:
                for start in range(0, high - low, array.chunks[0]):
                    stop = min(high - low, start + array.chunks[0])
                    target[start:stop] = array[low + start : low + stop]
        target_meta.attrs.update(
            {
                key: value
                for key, value in meta.attrs.items()
                if key not in ("segments", "quality_report", "phase_markers")
            }
        )
        rebased = []
        for i in segments:
            item = dict(described[i])
            for key in ("output_start", "output_end"):
                if key in item:
                    item[key] -= low
            rebased.append(item)
        target_meta.attrs["segments"] = rebased
        target_meta.array("episode_ends", np.array([high - low], dtype=np.int64))
        target_meta.array("segment_ends", segment_ends[segments] - low)
        provenance["complete"] = True
        root.attrs["episode_export"] = provenance
        outputs.append({"path": str(destination), "frames": high - low, **provenance})
        print(json.dumps(outputs[-1], ensure_ascii=False), flush=True)
    manifest = output_dir / "episodes_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help="source_episode names, or zero-based original episode indices",
    )
    parser.add_argument("--prefix")
    args = parser.parse_args()
    export_episodes(args.dataset, args.output, args.episodes, args.prefix)


if __name__ == "__main__":
    main()
