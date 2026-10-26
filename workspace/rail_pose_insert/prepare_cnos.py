"""Download official CNOS component weights and cache rail CAD templates; no devices."""
from pathlib import Path
import urllib.request

from rail_receive.logic import load_config
from rail_receive.cnos import CnosDetector, WEIGHTS


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1] / "third_party/cnos/pretrained"
    for relative, url in WEIGHTS.items():
        target = root / relative
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载官方权重：{target.name}", flush=True)
        temporary = target.with_suffix(".partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    detector = CnosDetector(load_config(here / "receive.yaml"))
    detector.close()
    print("CNOS 权重与导轨 CAD 模板已就绪；没有连接设备。")


if __name__ == "__main__":
    main()
