"""What this machine can actually run.

The product must not assume the developer's card. A user with 6 GB of VRAM is a
first-class user: they get smaller models, heavier quantization and CPU offload,
and the application stays usable rather than refusing to start.

Everything here is read-only observation, uses no extra dependencies, and degrades
to an honest "unknown" rather than guessing. A wrong number here would produce a
confident recommendation that cannot load.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from cvnator.observability import get_logger

log = get_logger(__name__)

#: Tiers the recommendation engine speaks in. The boundaries are deliberately
#: conservative: a card on the edge of a tier gets the lighter configuration,
#: because an out-of-memory failure costs far more than a slightly smaller model.
TIER_CPU_ONLY = "cpu_only"
TIER_LOW = "low"           # < 7 GB usable VRAM
TIER_STANDARD = "standard"  # 7-10 GB
TIER_RECOMMENDED = "recommended"  # 10-15 GB
TIER_HIGH = "high"          # >= 15 GB


@dataclass(slots=True)
class GPUInfo:
    name: str = ""
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    driver: str = ""
    vendor: str = ""

    @property
    def detected(self) -> bool:
        return bool(self.name) and self.vram_total_mb > 0


@dataclass(slots=True)
class HardwareProfile:
    """A snapshot of the machine, as far as it can be observed without new deps."""

    gpu: GPUInfo = field(default_factory=GPUInfo)
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    cpu_name: str = ""
    cpu_cores: int = 0
    os_name: str = ""
    architecture: str = ""
    backends: list[str] = field(default_factory=list)
    #: Set when a value could not be read, so the UI can say "unknown" instead of 0.
    unknown: list[str] = field(default_factory=list)

    @property
    def tier(self) -> str:
        if not self.gpu.detected:
            return TIER_CPU_ONLY
        gb = self.gpu.vram_total_mb / 1024
        if gb < 7:
            return TIER_LOW
        if gb < 10:
            return TIER_STANDARD
        if gb < 15:
            return TIER_RECOMMENDED
        return TIER_HIGH

    @property
    def low_vram(self) -> bool:
        return self.tier in {TIER_CPU_ONLY, TIER_LOW}

    def usable_vram_mb(self, headroom_fraction: float = 0.15) -> int:
        """VRAM a model may plan to use.

        Leaves headroom because the desktop compositor, the browser and the CUDA
        context all want some of the card, and because ``free`` at detection time is
        not ``free`` at load time.
        """
        if not self.gpu.detected:
            return 0
        return max(0, int(self.gpu.vram_total_mb * (1.0 - headroom_fraction)))

    def as_dict(self) -> dict:
        data = asdict(self)
        data["tier"] = self.tier
        data["low_vram"] = self.low_vram
        data["usable_vram_mb"] = self.usable_vram_mb()
        return data


def _run(command: list[str], timeout: float = 5.0) -> str:
    """Run a probe command, returning "" on any failure.

    Detection must never raise: a machine without nvidia-smi is a supported machine.
    """
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("Hardware probe %s failed: %s", command[0], exc)
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detect_gpu() -> GPUInfo:
    """Read the NVIDIA card through nvidia-smi, if it is there."""
    if shutil.which("nvidia-smi") is None:
        return GPUInfo()

    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return GPUInfo()

    # Only the first GPU is used; multi-GPU scheduling is not something this
    # application does, and pretending otherwise would mislead the estimator.
    first = output.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return GPUInfo()
    try:
        return GPUInfo(
            name=parts[0],
            vram_total_mb=int(float(parts[1])),
            vram_used_mb=int(float(parts[2])),
            vram_free_mb=int(float(parts[3])),
            driver=parts[4] if len(parts) > 4 else "",
            vendor="NVIDIA",
        )
    except ValueError:
        return GPUInfo(name=parts[0], vendor="NVIDIA")


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def detect_ram() -> tuple[int, int]:
    """Total and available RAM in MB. ``(0, 0)`` when it cannot be read."""
    system = platform.system()
    if system == "Windows":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return (
                    int(status.ullTotalPhys // (1024 * 1024)),
                    int(status.ullAvailPhys // (1024 * 1024)),
                )
        except (AttributeError, OSError) as exc:  # pragma: no cover - platform specific
            log.debug("GlobalMemoryStatusEx failed: %s", exc)
        return (0, 0)

    meminfo = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                meminfo[key.strip()] = rest.strip()
    except OSError:
        return (0, 0)

    def kb(key: str) -> int:
        raw = meminfo.get(key, "").split()
        return int(raw[0]) // 1024 if raw else 0

    return (kb("MemTotal"), kb("MemAvailable") or kb("MemFree"))


def detect_cpu() -> tuple[str, int]:
    name = platform.processor() or platform.machine() or ""
    if platform.system() == "Linux" and not name.strip():
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.partition(":")[2].strip()
                        break
        except OSError:
            pass
    return name, os.cpu_count() or 0


def detect_backends(ollama_base_url: str = "", openai_base_url: str = "") -> list[str]:
    """Which inference backends are *installed*, not which are currently running.

    Reachability is a separate question answered by the gateway's health check;
    conflating them would make a stopped service look like a missing one.
    """
    backends: list[str] = []
    if shutil.which("ollama") or ollama_base_url:
        backends.append("ollama")
    if shutil.which("llama-server") or shutil.which("llama-cli"):
        backends.append("llama.cpp")
    if openai_base_url:
        backends.append("openai-compatible")
    return backends


def detect_hardware(
    *, ollama_base_url: str = "", openai_base_url: str = ""
) -> HardwareProfile:
    """Observe the machine. Never raises; unreadable values are named, not invented."""
    gpu = detect_gpu()
    ram_total, ram_available = detect_ram()
    cpu_name, cpu_cores = detect_cpu()

    unknown: list[str] = []
    if not gpu.detected:
        unknown.append("gpu")
    if not ram_total:
        unknown.append("ram")
    if not cpu_name:
        unknown.append("cpu_name")

    profile = HardwareProfile(
        gpu=gpu,
        ram_total_mb=ram_total,
        ram_available_mb=ram_available,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        os_name=f"{platform.system()} {platform.release()}".strip(),
        architecture=platform.machine(),
        backends=detect_backends(ollama_base_url, openai_base_url),
        unknown=unknown,
    )
    log.info(
        "Hardware: %s, %s MB VRAM, %s MB RAM, %s cores, tier=%s",
        gpu.name or "no GPU detected",
        gpu.vram_total_mb,
        ram_total,
        cpu_cores,
        profile.tier,
    )
    return profile
