import struct

import numpy as np

from rail_pose.bag import parse_camera_info, parse_image, parse_tf_string, _image_array


def _align(position: int) -> int:
    return (position + 3) & ~3


def _put_str(parts: list[bytes], text: str) -> None:
    raw = text.encode("ascii") + b"\x00"
    parts.append(struct.pack("<I", len(raw)))
    parts.append(raw)
    parts.append(b"\x00" * (_align(sum(map(len, parts))) - sum(map(len, parts))))


def test_parse_camera_info_and_bgr_image_round_trip():
    body = [b"\x00\x01\x00\x00", struct.pack("<II", 0, 1)]
    _put_str(body, "Color")
    body.append(struct.pack("<II", 2, 3))
    _put_str(body, "plumb_bob")
    body.append(struct.pack("<I", 5))
    body.append(struct.pack("<5d", -0.05, 0.06, 0.0, 0.0, -0.02))
    body.append(struct.pack("<9d", 100, 0, 1.5, 0, 100, 0.5, 0, 0, 1))
    body.append(struct.pack("<9d", *np.eye(3).ravel()))
    body.append(struct.pack("<12d", *([0.0] * 12)))
    info = parse_camera_info(b"".join(body))
    assert info["width"] == 3 and info["height"] == 2
    assert info["fx"] == 100 and info["ppx"] == 1.5

    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = [b"\x00\x01\x00\x00", struct.pack("<II", 0, 1)]
    _put_str(image, "Color")
    image.append(struct.pack("<II", 2, 3))
    raw = b"bgr8\x00"
    image.append(struct.pack("<I", len(raw)))
    image.append(raw)
    image.append(b"\x00")  # is_bigendian
    image.append(b"\x00" * (_align(sum(map(len, image))) - sum(map(len, image))))
    image.append(struct.pack("<I", 9))
    image.append(struct.pack("<I", 18))
    image.append(pixels.tobytes())
    header, payload = parse_image(b"".join(image))
    rgb = _image_array(payload, header)
    np.testing.assert_array_equal(rgb, pixels[..., ::-1])


def test_tf_string_transposes_column_major_rotation():
    text = "rotation=0,1,0,0,0,1,1,0,0;translation=0.01,0,0"
    blob = b"\x00\x01\x00\x00" + struct.pack("<I", len(text) + 1) + text.encode() + b"\x00"
    rotation, translation = parse_tf_string(blob)
    np.testing.assert_allclose(rotation, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(translation, [0.01, 0, 0])
