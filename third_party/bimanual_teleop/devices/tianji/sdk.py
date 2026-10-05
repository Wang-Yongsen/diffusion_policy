"""Load the unmodified official SDK and adapt its Python binding defects.

The vendor owns UDP transport. A command result is the host time at which its
API returned successfully, never a UDP receipt or a controller execution time.
"""

from __future__ import annotations

import argparse
import ctypes as ct
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import threading
import time


DEFAULT_SDK_ROOT = Path(__file__).resolve().parents[2] / "vendor/tianji"
DEFAULT_MODEL = DEFAULT_SDK_ROOT / "CommonConfig/ccs_m6_40.MvKDCfg"
SDK_COMMIT = "02440e886fb59095711eb9ec6dcbedd8be08922a"
SDK_VERSION = 100343014
_LOAD_LOCK = threading.RLock()
_OWNER_LOCK = threading.Lock()
_OWNER = None


def sdk_root(path=None) -> Path:
    return Path(path or DEFAULT_SDK_ROOT).expanduser().resolve()


@lru_cache(maxsize=8)
def _load(root: Path, component: str):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("The bundled Tianji SDK requires Linux x86_64")
    manifest = json.loads((DEFAULT_SDK_ROOT / "manifest.json").read_text())
    library = "libMarvinSDK.so" if component == "robot" else "libKine.so"
    files = (f"SDK_PYTHON/fx_{component}.py", f"SDK_PYTHON/{library}")
    for name in files:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Official Tianji SDK file missing: {path}; use --sdk-root SDK_DIRECTORY")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]:
            raise RuntimeError(f"Unsupported or modified Tianji SDK file: {path}; expected {SDK_COMMIT}")
    spec = importlib.util.spec_from_file_location(f"_tianji_{component}_{hash(root)}", root / files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layouts = ((module.StateCtr, 12), (module.RT_IN, 368), (module.RT_OUT, 312), (module.DCSS, 1428)) \
        if component == "robot" else ((module.FX_InvKineSolvePara, 992),)
    if ct.sizeof(ct.c_long) != 8 or any(ct.sizeof(kind) != size for kind, size in layouts):
        raise RuntimeError("Unsupported official Tianji SDK structure layout")
    return module


def load_sdk(component: str, root=None):
    if component not in ("robot", "kine"):
        raise ValueError("SDK component must be robot or kine")
    with _LOAD_LOCK:
        return _load(sdk_root(root), component)


class _RemovedLibrary(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error("--library has been removed; use --sdk-root for an official SDK directory, "
                     "or omit it to use the bundled SDK")


def add_sdk_argument(parser):
    parser.add_argument("--sdk-root", type=Path, help="official SDK directory; default: bundled SDK")
    parser.add_argument("--library", action=_RemovedLibrary, help=argparse.SUPPRESS)


@dataclass
class FeedbackSnapshot:
    """Unrounded SDK observations, in vendor units; not a wire structure."""

    received_ns: int = 0
    packet_index: int = 0  # Local observation index, not a UDP packet counter.
    sequence: list = field(default_factory=lambda: [0] * 2)
    input_sequence: list = field(default_factory=lambda: [0] * 2)
    state: list = field(default_factory=lambda: [0] * 2)
    commanded_state: list = field(default_factory=lambda: [0] * 2)
    error: list = field(default_factory=lambda: [0] * 2)
    impedance_type: list = field(default_factory=lambda: [0] * 2)
    low_speed: list = field(default_factory=lambda: [0] * 2)
    q: list = field(default_factory=lambda: [0.] * 14)
    dq: list = field(default_factory=lambda: [0.] * 14)
    current: list = field(default_factory=lambda: [0.] * 14)
    torque: list = field(default_factory=lambda: [0.] * 14)
    external_torque: list = field(default_factory=lambda: [0.] * 14)
    target: list = field(default_factory=lambda: [0.] * 14)
    cart_k: list = field(default_factory=lambda: [0.] * 14)
    cart_d: list = field(default_factory=lambda: [0.] * 14)
    tool_pose: list = field(default_factory=lambda: [0.] * 12)
    tool_dynamics: list = field(default_factory=lambda: [0.] * 20)
    velocity_ratio: list = field(default_factory=lambda: [0] * 2)
    acceleration_ratio: list = field(default_factory=lambda: [0] * 2)
    force_type: list = field(default_factory=lambda: [0] * 2)
    impedance_rotation: list = field(default_factory=lambda: [0.] * 14)
    force_tag: list = field(default_factory=lambda: [0.] * 2)
    wrench_raw: list = field(default_factory=lambda: [0.] * 12)

    @classmethod
    def from_dcss(cls, dcss, observed_ns, index):
        result = cls(received_ns=observed_ns, packet_index=index)
        for arm in range(2):
            state, inp, out = dcss.m_State[arm], dcss.m_In[arm], dcss.m_Out[arm]
            for name, value in (("sequence", out.m_OutFrameSerial), ("input_sequence", inp.m_InFrameSerial),
                                ("state", state.m_CurState), ("commanded_state", state.m_CmdState),
                                ("error", state.m_ERRCode), ("impedance_type", inp.m_ImpType),
                                ("low_speed", out.m_LowSpdFlag[0]), ("force_tag", out.m_EST_Joint_Firc[0]),
                                ("velocity_ratio", inp.m_Joint_Vel_Ratio),
                                ("acceleration_ratio", inp.m_Joint_Acc_Ratio), ("force_type", inp.m_Force_Type)):
                getattr(result, name)[arm] = value
            for name, values in (("q", out.m_FB_Joint_Pos), ("dq", out.m_FB_Joint_Vel),
                                 ("target", out.m_FB_Joint_Cmd), ("current", out.m_FB_Joint_CToq),
                                 ("torque", out.m_FB_Joint_SToq), ("external_torque", out.m_EST_Joint_Force),
                                 ("cart_k", [*inp.m_Cart_K, inp.m_Cart_KN]),
                                 ("cart_d", [*inp.m_Cart_D, inp.m_Cart_DN]),
                                 ("tool_pose", inp.m_ToolKine), ("tool_dynamics", inp.m_ToolDyn),
                                 ("impedance_rotation", inp.m_Force_PIDUL),
                                 ("wrench_raw", out.m_EST_Joint_Firc_Dot[:6])):
                width = len(values)
                getattr(result, name)[arm * width:(arm + 1) * width] = values
        return result


class ControlSDK:
    """One official robot connection with serialized, bounded command building."""

    def __init__(self, root=None):
        self.root = sdk_root(root)
        self.module = load_sdk("robot", self.root)
        self.robot = self.module.Marvin_Robot()
        self._lock = threading.RLock()
        self._opened = False
        self._sequence = (0, 0)
        self._index = 0
        self.cancelled = lambda: False
        lib = self.robot.robot
        # Only fix types on the official library object; never patch vendor files.
        signatures = {
            "OnGetBuf": ([ct.POINTER(self.module.DCSS)], ct.c_bool),
            "OnGetIntPara": ([ct.c_char_p, ct.POINTER(ct.c_long)], ct.c_long),
            "OnSetIntPara": ([ct.c_char_p, ct.c_long], ct.c_long),
            "OnLinkTo": ([ct.c_ubyte] * 4, ct.c_bool), "OnRelease": ([], ct.c_bool),
            "OnGetSDKVersion": ([], ct.c_long), "OnLocalLogOff": ([], None),
            "OnClearSet": ([], ct.c_bool), "OnSetSend": ([], ct.c_bool),
        }
        for arm in ("A", "B"):
            for name, args in (("OnSetJointCmdPos", [ct.POINTER(ct.c_double)]),
                               ("OnSetTargetState", [ct.c_int]), ("OnSetImpType", [ct.c_int]),
                               ("OnSetJointLmt", [ct.c_int, ct.c_int]),
                               ("OnSetTool", [ct.POINTER(ct.c_double)] * 2),
                               ("OnSetCartKD", [ct.POINTER(ct.c_double)] * 2 + [ct.c_int]),
                               ("OnSetEefRot", [ct.c_int, ct.POINTER(ct.c_double)]),
                               ("OnSetUserSpcfData", [ct.c_long])):
                signatures[f"{name}_{arm}"] = (args, ct.c_bool)
        for name, (args, result) in signatures.items():
            try:
                function = getattr(lib, name)
            except AttributeError as error:
                raise RuntimeError(f"Official Tianji SDK lacks {name}") from error
            function.argtypes, function.restype = args, result
        if self.robot.SDK_version() != SDK_VERSION or self.robot.check_sdk_type_compat()[0] < 0:
            raise RuntimeError("Incompatible official Tianji control SDK")
        self.robot.local_log_switch("0")

    def open(self, ip):
        global _OWNER
        with _OWNER_LOCK:
            if _OWNER is not None:
                raise RuntimeError("The Tianji SDK already has an owner in this process")
            _OWNER = self
        try:
            if not self.robot.connect(ip):
                raise RuntimeError("Official Tianji SDK connection failed (check address and UDP port 4730)")
            self._opened = True
        except BaseException:
            try:
                self.robot.release_robot()
            finally:
                with _OWNER_LOCK:
                    _OWNER = None
            raise

    def close(self):
        global _OWNER
        with self._lock:
            if not self._opened:
                return
            try:
                if not self.robot.release_robot():
                    raise RuntimeError("Official Tianji SDK release failed")
            finally:
                self._opened = False
                with _OWNER_LOCK:
                    _OWNER = None

    def _require_open(self):
        if not self._opened:
            raise RuntimeError("Tianji SDK is not connected")

    def read(self):
        self._require_open()
        value = self.module.DCSS()
        if not self.robot.robot.OnGetBuf(ct.byref(value)):
            raise RuntimeError("Official Tianji SDK feedback unavailable")
        return value

    def poll_feedback(self):
        if not self._opened:
            return None
        value = self.read()
        observed = time.monotonic_ns()
        sequence = tuple(out.m_OutFrameSerial for out in value.m_Out)
        if sequence == self._sequence:
            return None
        self._sequence = sequence
        self._index += 1
        return FeedbackSnapshot.from_dcss(value, observed, self._index)

    def get_int(self, name):
        with self._lock:
            self._require_open()
            value = ct.c_long()
            code = self.robot.robot.OnGetIntPara(self._param_name(name), ct.byref(value))
            if code != 0:
                raise RuntimeError(f"{name} query returned {code}")
            return value.value

    @staticmethod
    def _param_name(name):
        encoded = name.encode("ascii")
        if not encoded or len(encoded) >= 30 or b"\0" in encoded:
            raise ValueError("Tianji parameter name must contain 1–29 ASCII bytes")
        return encoded.ljust(30, b"\0")

    def versions(self):
        controller = self.get_int("VERSION")
        if not controller:
            raise RuntimeError("Controller VERSION query returned zero")
        return SDK_VERSION, controller

    def _set_int(self, name):
        with self._lock:
            self._require_open()
            code = self.robot.robot.OnSetIntPara(self._param_name(name), 0)
            if code != 0:
                raise RuntimeError(f"{name} returned {code}; inspect feedback")
            return time.monotonic_ns()

    @staticmethod
    def _arms(mask):
        if mask not in (1, 2, 3):
            raise ValueError("Select one or both Tianji arms")
        return [(index, arm) for index, arm in enumerate(("A", "B")) if mask & (1 << index)]

    def _command(self, build, expiry=None, *, disabling=False):
        with self._lock:
            self._require_open()
            slot_deadline = time.monotonic_ns() + 5_000_000
            if expiry is not None:
                slot_deadline = min(slot_deadline, expiry)

            def check():
                if not disabling and self.cancelled():
                    raise RuntimeError("Tianji command cancelled")
                if expiry is not None and time.monotonic_ns() >= expiry:
                    raise RuntimeError("Target expired before SDK submission")

            while True:
                check()
                if self.robot.clear_set():
                    break
                if time.monotonic_ns() >= slot_deadline:
                    raise RuntimeError("Tianji SDK command pending; send slot timeout")
                time.sleep(.0005)
            try:
                build()
                check()
                if not self.robot.send_cmd():
                    raise RuntimeError("Official Tianji SDK rejected command submission")
                return time.monotonic_ns()
            except BaseException:
                # Discard only an unsubmitted build. The vendor cannot cancel a
                # packet already accepted by send_cmd or intercept its deadline.
                self.robot.clear_set()
                raise

    @staticmethod
    def _require(result, operation):
        if not result:
            raise RuntimeError(f"Official Tianji SDK rejected {operation}")

    def configure(self, mask, profiles):
        def build():
            for index, arm in self._arms(mask):
                p = profiles[index]
                self._require(self.robot.set_tool(arm, [0.] * 6, list(p.tool_dyn10)), "tool")
                self._require(self.robot.set_vel_acc(arm, p.velocity_ratio, p.acceleration_ratio), "limits")
                self._require(self.robot.set_cart_kd_params(arm, [*p.stiffness, p.nullspace_stiffness],
                                                          [*p.damping, p.nullspace_damping]), "Cartesian KD")
                self._require(self.robot.set_EefCart_control_params(arm, 1, [0.] * 7), "Cartesian axes")
                self._require(self.robot.set_PD_vel_est_step(arm, 0), "velocity estimator")
        return self._command(build)

    def submit(self, mask, targets, expiry, *, engage=False):
        def build():
            for index, arm in self._arms(mask):
                self._require(self.robot.set_joint_cmd_pose(arm, list(targets[index*7:index*7+7])), "target")
                if engage:
                    self._require(self.robot.set_impedance_type(arm, 2), "impedance type")
                    self._require(self.robot.set_state(arm, 3), "Cartesian mode")
        return self._command(build, expiry)

    def engage(self, mask, targets, expiry):
        return self.submit(mask, targets, expiry, engage=True)

    def move_joints(self, index, target, velocity, acceleration, expiry):
        arm = self._arms(1 << index)[0][1]
        def build():
            self._require(self.robot.set_vel_acc(arm, velocity, acceleration), "limits")
            self._require(self.robot.set_joint_cmd_pose(arm, list(target)), "position seed")
            self._require(self.robot.set_state(arm, 1), "position mode")
        return self._command(build, expiry)

    def clear_errors(self, index):
        self._arms(1 << index)
        if self.cancelled():
            raise RuntimeError("Tianji clear errors cancelled")
        return self._set_int(f"RESET{index}")

    def hold(self, mask):
        self._arms(mask)
        return self._set_int({1: "RSTA0", 2: "RSTA1", 3: "RSTA01"}[mask])

    def disable(self, mask):
        def build():
            for _, arm in self._arms(mask):
                self._require(self.robot.set_state(arm, 0), "disable")
        return self._command(build, disabling=True)
