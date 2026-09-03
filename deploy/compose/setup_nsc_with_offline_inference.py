# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich>=13"]
# ///
"""Automated setup for Nsight Copilot Offline on DGX Spark.

Installs Docker, NVIDIA Container Toolkit, configures NGC authentication,
and starts the inference services. Idempotent — safe to re-run.

Usage:
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sudo -E sh
    sudo -E $(which uv) run setup_nsc_with_offline_inference.py
"""

import configparser
from dataclasses import dataclass
from pathlib import Path
import grp
import json
import os
import pwd
import platform
import re
import signal
import subprocess
import sys
import time
import webbrowser

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

NGC_SETUP_URL = "https://ngc.nvidia.com/setup"
NVCR_REGISTRY = "nvcr.io"
TOTAL_STEPS = 8

COMPOSE_FILES = ["-f", "compose.yaml"]

ARCH_RE = re.compile(r"^[a-z0-9]+$")
CODENAME_RE = re.compile(r"^[a-z]+$")

console = Console()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def print_step(step_num: int, message: str) -> None:
    console.print(f"\n[bold]\\[{step_num}/{TOTAL_STEPS}][/bold] {message}")


def _status_line(badge: str, detail: str = "") -> str:
    suffix = f"  [dim]{detail}[/dim]" if detail != "" else ""
    return f"  {badge}{suffix}"


def print_ok(detail: str = "") -> None:
    console.print(_status_line("[bold green]OK[/bold green]", detail))


def print_skipped(detail: str = "") -> None:
    console.print(_status_line("[bold yellow]SKIPPED[/bold yellow]", detail))


def print_failed(detail: str = "") -> None:
    console.print(_status_line("[bold red]FAILED[/bold red]", detail))


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
    stdin_data: str | bytes | None = None,
    status_msg: str = "",
) -> subprocess.CompletedProcess:
    """Run a command with an optional spinner.

    Shows a rich spinner with status_msg while the command runs.
    Output is captured during spinner display to avoid visual collision,
    then printed on failure. When status_msg is empty, no spinner is shown
    and output streams directly to the terminal.

    Returns:
        CompletedProcess from subprocess.run.

    Raises:
        FileNotFoundError: If the command binary is not found and check=False.
    """
    merged_env = {**os.environ, **(env or {})}
    use_spinner = bool(status_msg)

    kwargs: dict = {
        "check": check,
        "env": merged_env,
    }

    # With spinner: capture everything to keep display clean.
    # Without spinner: capture only stderr for error reporting, let stdout stream.
    if use_spinner or capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    else:
        kwargs["stderr"] = subprocess.PIPE

    if stdin_data is not None:
        kwargs["input"] = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data

    try:
        if use_spinner:
            with console.status(f"[yellow]{status_msg}[/yellow]", spinner="dots"):
                return subprocess.run(cmd, **kwargs)
        return subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as exc:
        stderr_text = ""
        if exc.stderr:
            stderr_text = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        console.print(f"  [red]Command failed:[/red] [dim]{' '.join(cmd)}[/dim]")
        if stderr_text != "":
            for line in stderr_text.strip().splitlines()[-10:]:
                console.print(f"  [red dim]{line}[/red dim]")
        sys.exit(1)
    except FileNotFoundError:
        if check:
            console.print(f"  [red]Command not found:[/red] [dim]{cmd[0]}[/dim]")
            sys.exit(1)
        raise


def run_streaming(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with stdout/stderr connected to the terminal."""
    merged_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(cmd, env=merged_env, check=False)
    except FileNotFoundError:
        if check:
            console.print(f"  [red]Command not found:[/red] [dim]{cmd[0]}[/dim]")
            sys.exit(1)
        raise

    if check and result.returncode != 0:
        console.print(f"  [red]Command failed:[/red] [dim]{' '.join(cmd)}[/dim]")
        sys.exit(1)

    return result


def run_silent(cmd: list[str]) -> bool:
    """Run a command silently.

    Returns:
        True if exit code 0, False otherwise.
    """
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_invoking_user() -> tuple[str, int, int, Path]:
    """Return (username, uid, gid, home) of the real user (handles sudo)."""
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user is not None:
        if sudo_user == "root":
            console.print("[red]SUDO_USER is 'root'; cannot determine the real invoking user.[/red]")
            console.print("[red]Run as a regular user with: sudo uv run setup_nsc_with_offline_inference.py[/red]")
            sys.exit(1)
        pw = pwd.getpwnam(sudo_user)
        return sudo_user, pw.pw_uid, pw.pw_gid, Path(pw.pw_dir)
    uid = os.getuid()
    pw = pwd.getpwuid(uid)
    return pw.pw_name, pw.pw_uid, pw.pw_gid, Path(pw.pw_dir)


def ensure_root() -> None:
    if os.geteuid() != 0:
        console.print(f"[red]This script must be run as root. Re-run with: sudo uv run {sys.argv[0]}[/red]")
        sys.exit(1)


def runtime_group_ids(username: str, gid: int) -> list[int]:
    group_ids = set(os.getgrouplist(username, gid))
    group_ids.add(gid)
    group_ids.add(grp.getgrnam("docker").gr_gid)
    return sorted(group_ids)


def drop_privileges(username: str, uid: int, gid: int) -> None:
    """Drop root privileges to the invoking user.

    Must be called after all root-requiring steps are complete.
    Preserves the user's supplementary groups and includes the docker group
    so subsequent docker commands work without sudo.
    """
    os.setgroups(runtime_group_ids(username, gid))
    os.setgid(gid)
    os.setuid(uid)


def resolve_docker_config_dir(home: Path) -> Path:
    """Resolve the docker config dir: explicit DOCKER_CONFIG, else ~/.docker."""
    docker_config = os.environ.get("DOCKER_CONFIG")
    if docker_config is not None:
        return Path(docker_config)
    return home / ".docker"


def configure_user_environment(username: str, home: Path) -> Path:
    """Point HOME/USER/LOGNAME/DOCKER_CONFIG at the invoking user for subprocesses.

    Call after drop_privileges(). Returns the resolved docker config dir.
    """
    docker_config_dir = resolve_docker_config_dir(home)
    os.environ["HOME"] = str(home)
    os.environ["USER"] = username
    os.environ["LOGNAME"] = username
    os.environ["DOCKER_CONFIG"] = str(docker_config_dir)
    return docker_config_dir


def fix_docker_config_ownership(home: Path, uid: int, gid: int) -> bool:
    """Repair a root-owned docker config dir left by an earlier root login.

    A previous ``sudo docker login`` can leave ~/.docker as ``700 root:root``,
    which the invoking user can't traverse — breaking config loading and
    compose-plugin discovery, so docker appears to still require sudo.

    Acts only when the config dir exists and is not already owned by the target
    user: the healthy case costs a single stat and writes nothing. When a repair
    is needed the whole tree is reassigned, since a root login created all of it.
    Must run while still root, before drop_privileges().

    Returns:
        True if ownership was repaired, False if nothing needed doing.
    """
    root = resolve_docker_config_dir(home)
    if not root.exists() or root.stat().st_uid == uid:
        return False
    for path in [root, *root.rglob("*")]:
        os.chown(path, uid, gid)
    return True


def prompt_input(message: str) -> str:
    """Read a line from stdin.

    Returns:
        The stripped input, or empty string if non-interactive.
    """
    if not sys.stdin.isatty():
        return ""
    try:
        return console.input(message).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def _has_display() -> bool:
    """Return True if a graphical display is likely available for a browser.

    This script is Linux-only (apt-get/systemctl/nvidia-ctk), so an X11 or
    Wayland display variable is the relevant signal. On a headless host (SSH,
    remote provisioning) neither is set, so we skip the browser launch and rely
    on the printed URL + prompt.
    """
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _read_os_release() -> dict[str, str]:
    """Parse /etc/os-release into a dict.

    Returns:
        Dict mapping keys to values from /etc/os-release.
    """
    result: dict[str, str] = {}
    with open("/etc/os-release") as f:
        for line in f:
            if "=" in line:
                k, _, v = line.strip().partition("=")
                result[k] = v.strip('"')
    return result


# ---------------------------------------------------------------------------
# Step 1: Install Docker Engine
# ---------------------------------------------------------------------------


def step_install_docker() -> None:
    print_step(1, "Install Docker Engine")

    if run_silent(["docker", "--version"]) and run_silent(["docker", "compose", "version"]):
        run(["systemctl", "start", "docker"], check=False, status_msg="Ensuring Docker daemon is running...")
        print_skipped("already installed")
        return

    run(["apt-get", "update", "-qq"], status_msg="Updating apt cache...")
    run(
        ["apt-get", "install", "-y", "-qq", "ca-certificates", "curl"],
        status_msg="Installing prerequisites...",
    )

    # Add Docker GPG key
    keyring_dir = "/etc/apt/keyrings"
    gpg_path = os.path.join(keyring_dir, "docker.asc")
    if not os.path.isfile(gpg_path):
        os.makedirs(keyring_dir, mode=0o755, exist_ok=True)
        run(
            ["curl", "-fsSL", "https://download.docker.com/linux/ubuntu/gpg", "-o", gpg_path],
            status_msg="Downloading Docker GPG key...",
        )
        os.chmod(gpg_path, 0o644)

    # Add Docker apt repository
    sources_path = "/etc/apt/sources.list.d/docker.list"
    if not os.path.isfile(sources_path):
        result = run(["dpkg", "--print-architecture"], capture=True)
        arch = result.stdout.decode().strip()
        if not ARCH_RE.match(arch):
            console.print(f"  [red]Unexpected architecture value:[/red] {arch}")
            sys.exit(1)

        os_release = _read_os_release()
        codename = os_release.get("VERSION_CODENAME", "")
        if not CODENAME_RE.match(codename):
            console.print(f"  [red]Unexpected VERSION_CODENAME value:[/red] {codename}")
            sys.exit(1)

        repo_line = f"deb [arch={arch} signed-by={gpg_path}] https://download.docker.com/linux/ubuntu {codename} stable"
        with open(sources_path, "w") as f:
            f.write(repo_line + "\n")

    run(["apt-get", "update", "-qq"], status_msg="Updating apt cache...")
    run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "docker-ce",
            "docker-ce-cli",
            "containerd.io",
            "docker-buildx-plugin",
            "docker-compose-plugin",
        ],
        status_msg="Installing Docker packages...",
    )

    print_ok()


# ---------------------------------------------------------------------------
# Step 2: Configure Docker without sudo
# ---------------------------------------------------------------------------


def step_configure_docker_no_sudo(username: str, home: Path, uid: int, gid: int) -> None:
    print_step(2, "Configure Docker for non-root usage")

    # Repair a root-owned ~/.docker before the group check — this must run on
    # every invocation, including re-runs where the user is already in the
    # docker group but a prior root login left the config dir unreadable.
    if fix_docker_config_ownership(home, uid, gid):
        console.print(f"  [dim]Reclaimed root-owned {resolve_docker_config_dir(home)} for {username}.[/dim]")

    try:
        docker_group = grp.getgrnam("docker")
        if username in docker_group.gr_mem:
            print_skipped(f"user '{username}' already in docker group")
            return
    except KeyError:
        pass  # group doesn't exist yet

    run(["groupadd", "-f", "docker"], status_msg="Creating docker group...")
    run(["usermod", "-aG", "docker", username], status_msg=f"Adding {username} to docker group...")

    print_ok()
    console.print("  [yellow]Note: Setup continues with Docker access enabled. Existing shells may need[/yellow]")
    console.print("  [yellow]'newgrp docker' or a fresh login before manual docker commands work without sudo.[/yellow]")


# ---------------------------------------------------------------------------
# Step 3: Install NVIDIA GPU driver
# ---------------------------------------------------------------------------


_NVIDIA_SERVER_OPEN_DRIVER_RE = re.compile(r"^nvidia-driver-(\d+)-server-open$")


def _select_latest_nvidia_server_open_driver(package_names: str) -> str | None:
    """Return the highest-branch server/open driver package in apt output."""
    candidates: list[tuple[int, str]] = []
    for package_name in package_names.splitlines():
        package_name = package_name.strip()
        match = _NVIDIA_SERVER_OPEN_DRIVER_RE.fullmatch(package_name)
        if match is not None:
            candidates.append((int(match.group(1)), package_name))
    return max(candidates)[1] if candidates else None


def _find_latest_nvidia_server_open_driver() -> str | None:
    """Find the newest server/open NVIDIA driver offered by configured apt repos."""
    result = run(["apt-cache", "pkgnames", "nvidia-driver-"], check=False, capture=True)
    if result.returncode != 0:
        return None
    stdout = result.stdout.decode(errors="replace") if isinstance(result.stdout, bytes) else result.stdout
    return _select_latest_nvidia_server_open_driver(stdout or "")


def _install_nvidia_driver_fallback_if_needed() -> bool:
    """Install the newest server/open driver when auto-detection installed none."""
    if run_silent(["modinfo", "nvidia"]):
        return True

    fallback_package = _find_latest_nvidia_server_open_driver()
    if fallback_package is None:
        console.print(
            "  [red]ubuntu-drivers did not install a driver, and no "
            "nvidia-driver-<branch>-server-open package is available.[/red]"
        )
        return False

    console.print(
        "  [yellow]ubuntu-drivers selected no driver; falling back to "
        f"{fallback_package}.[/yellow]"
    )
    run(
        ["apt-get", "install", "-y", fallback_package],
        status_msg=f"Installing {fallback_package}...",
    )
    return True


def _is_kernel_module_loaded(module_name: str) -> bool:
    return (Path("/sys/module") / module_name).is_dir()


def _load_nvidia_driver() -> bool:
    """Ensure the NVIDIA module exists and load it with at most one retry."""
    if not _install_nvidia_driver_fallback_if_needed():
        return False

    if run_streaming(["modprobe", "nvidia"], check=False).returncode == 0:
        return True

    # Unloading nouveau is the only condition where retrying can help.
    if not _is_kernel_module_loaded("nouveau"):
        return False
    if run_streaming(["modprobe", "-r", "nouveau"], check=False).returncode != 0:
        return False
    return run_streaming(["modprobe", "nvidia"], check=False).returncode == 0


def step_install_nvidia_driver() -> None:
    print_step(3, "Install NVIDIA GPU driver")

    if run_silent(["nvidia-smi"]):
        print_skipped("driver already loaded")
        return

    run(
        ["apt-get", "install", "-y", "-qq", "ubuntu-drivers-common"],
        status_msg="Installing ubuntu-drivers-common...",
    )
    # ubuntu-drivers exits 0 even when it finds no matching driver.
    run_streaming(["ubuntu-drivers", "autoinstall"])

    if not _load_nvidia_driver():
        console.print("  [red]NVIDIA driver module could not be loaded.[/red]")
        console.print(
            "  [red]Review the error above. Reboot only if it indicates a driver/kernel "
            "version mismatch or a module that requires activation after boot.[/red]"
        )
        sys.exit(1)

    if run_streaming(["nvidia-smi"], check=False).returncode == 0:
        print_ok("driver loaded without reboot")
        return

    console.print("  [yellow]NVIDIA driver module loaded, but nvidia-smi failed.[/yellow]")
    console.print(
        "  [yellow]A reboot may be required (for example, after a driver/library version "
        "mismatch). Review the nvidia-smi error above before rebooting.[/yellow]"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 4: Install NVIDIA Container Toolkit
# ---------------------------------------------------------------------------


def step_install_nvidia_ctk() -> None:
    print_step(4, "Install NVIDIA Container Toolkit")

    if run_silent(["nvidia-ctk", "--version"]):
        print_skipped("already installed")
        return

    # GPG key — download and dearmor without shell pipeline
    keyring_path = Path("/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
    if not keyring_path.is_file():
        gpg_result = run(
            ["curl", "-fsSL", "https://nvidia.github.io/libnvidia-container/gpgkey"],
            capture=True,
            status_msg="Downloading NVIDIA CTK GPG key...",
        )
        dearmor_result = run(["gpg", "--dearmor"], stdin_data=gpg_result.stdout, capture=True)
        with open(keyring_path, "wb") as f:
            f.write(dearmor_result.stdout)

    # Apt repo — download and substitute without shell pipeline
    sources_path = Path("/etc/apt/sources.list.d/nvidia-container-toolkit.list")
    if not sources_path.is_file():
        list_result = run(
            [
                "curl",
                "-fsSL",
                "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list",
            ],
            capture=True,
            status_msg="Downloading NVIDIA CTK apt source list...",
        )
        content = list_result.stdout.decode().replace(
            "deb https://",
            f"deb [signed-by={keyring_path}] https://",
        )
        with open(sources_path, "w") as f:
            f.write(content)

    run(["apt-get", "update", "-qq"], status_msg="Updating apt cache...")
    run(
        ["apt-get", "install", "-y", "-qq", "nvidia-container-toolkit"],
        status_msg="Installing NVIDIA Container Toolkit...",
    )

    print_ok()


# ---------------------------------------------------------------------------
# Step 5: Configure NVIDIA CTK for Docker
# ---------------------------------------------------------------------------


def step_configure_nvidia_ctk() -> None:
    print_step(5, "Configure NVIDIA Container Toolkit for Docker")

    daemon_json_path = "/etc/docker/daemon.json"
    try:
        with open(daemon_json_path) as f:
            config = json.load(f)
        if "nvidia" in config.get("runtimes", {}):
            print_skipped("NVIDIA runtime already configured")
            return
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # not configured yet

    run(
        ["nvidia-ctk", "runtime", "configure", "--runtime=docker"],
        status_msg="Configuring NVIDIA runtime...",
    )
    run(["systemctl", "restart", "docker"], status_msg="Restarting Docker daemon...")

    print_ok()


# ---------------------------------------------------------------------------
# GPU detection and placement
# ---------------------------------------------------------------------------
#
# compose.yaml pins each GPU service to one GPU via device_ids (default "0" — the
# single-GPU / DGX Spark layout). On a multi-GPU host we dedicate the largest GPU
# to the LLM and the next to the small models, published as env vars the compose
# interpolates; the user configures nothing. A multi-GPU host needs one GPU ≥80 GB
# for the LLM plus a second for the small models; a single-GPU host needs one GPU
# big enough for everything (LLM + small models). Hosts that don't qualify are
# reported unsupported and setup aborts early — instead of crashing at serve time —
# since tensor-parallel serving across smaller GPUs isn't wired yet (it would need a
# device_ids *list*, which an env var can't express, and NVIDIA_VISIBLE_DEVICES is
# ignored under a device reservation).

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.total,name,compute_cap",
    "--format=csv,noheader,nounits",
]

# Cap nvidia-smi so a wedged driver can't hang setup; on timeout detection returns
# empty, which select_gpu_profile treats as "no GPU" and the caller aborts.
NVIDIA_SMI_TIMEOUT_SECONDS = 15

# Per-GPU floor for gpt-oss-120b on its own GPU. The NIM manifest tags the TP1
# profile ">=96 GB/gpu" for the default (huge) context, but compose caps the context
# (NIM_MAX_MODEL_LEN=32768), which bounds the KV cache — it then runs comfortably on
# an 80 GB H100 (~75 GB used, validated). NIM compares decimal GB. An 80 GB H100
# clears this; a 48 GB L40S does not.
LLM_MIN_GPU_GB = 80

# The small models (autocomplete ~7B, bge, rerank) share one GPU; below this their
# memory fractions don't leave room for the 7B. A single-GPU host must hold the LLM
# AND these together, so its floor is LLM_MIN_GPU_GB + SMALL_MODELS_MIN_GB.
SMALL_MODELS_MIN_GB = 24

# gpt-oss-120b is MXFP4, which needs Hopper or newer (compute capability >= 9.0);
# Ampere (8.x) can't run it regardless of memory. Gating on this turns an 80 GB
# A100 from a late NIM crash into an early, clear abort.
MIN_COMPUTE_CAPABILITY = 9.0

# gpt-oss NIM image tag is host-dependent. NIM 2.0.x has a memory runaway on GB10
# unified memory (DGX Spark): the KV allocation ignores every limiter (util,
# absolute blocks, enforce-eager) and consumes the whole 128 GB pool → the host
# hangs. 1.6.1 runs there. On discrete GPUs (H100/GH200) 2.0.x works and 1.6.1
# does not. So use 1.6.1 on unified-memory hosts and 2.0.7 everywhere else.
# (Drop the split once NVIDIA ships a UMA-aware 2.0.x.)
GPT_OSS_IMAGE_TAG_DISCRETE = "2.0.7"
GPT_OSS_IMAGE_TAG_UNIFIED = "1.6.1"

_BYTES_PER_MIB = 1024 * 1024
_BYTES_PER_GB = 1_000_000_000


def _gpu_meets(memory_mib: int | None, min_gb: int) -> bool:
    """True if a GPU's VRAM clears a per-GPU floor (NIM compares decimal GB).

    Unknown memory (``None``, e.g. GB10's unified memory) can't be proven to clear
    any floor, so it never passes.
    """
    if memory_mib is None:
        return False
    return memory_mib * _BYTES_PER_MIB >= min_gb * _BYTES_PER_GB


def _can_run_llm(gpu: "GPUInfo") -> bool:
    """False only when the GPU is positively known too old for the LLM (MXFP4 needs
    cc >= 9.0). Unknown/unparseable compute capability is allowed — let NIM judge.
    """
    if gpu.compute_capability is None:
        return True
    try:
        return float(gpu.compute_capability) >= MIN_COMPUTE_CAPABILITY
    except ValueError:
        return True


@dataclass(frozen=True)
class GPUInfo:
    index: int
    memory_mib: int | None  # None when nvidia-smi can't report it (e.g. GB10 unified memory)
    name: str
    compute_capability: str | None = None  # e.g. "9.0" (Hopper), "12.1" (GB10); None if unreported


def parse_nvidia_smi(output: str) -> list[GPUInfo]:
    """Parse nvidia-smi CSV (``<memory_mib>, <name>, <compute_cap>`` per line).

    A bracketed nvidia-smi sentinel (e.g. ``[N/A]`` — DGX Spark's GB10 reports it for
    its unified memory) becomes ``None`` rather than dropping the GPU: it exists, we
    just can't read that field. Lines with unparseable memory are still skipped.

    Returns:
        One GPUInfo per parsed GPU, indexed in nvidia-smi enumeration order.
    """
    gpus: list[GPUInfo] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        # name sits between memory (first) and compute_cap (last); join the middle
        # so a comma in a name doesn't misalign the fixed first/last fields.
        mem_str, cc_str = parts[0], parts[-1]
        name = ",".join(parts[1:-1]).strip()
        mem: int | None
        if mem_str.startswith("["):
            mem = None  # nvidia-smi sentinel: [N/A] / [Not Supported] / ...
        else:
            try:
                mem = int(float(mem_str))
            except ValueError:
                continue
        cc = None if cc_str.startswith("[") else cc_str
        gpus.append(GPUInfo(index=len(gpus), memory_mib=mem, name=name, compute_capability=cc))
    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Query nvidia-smi for installed GPUs.

    Returns:
        Detected GPUs, or an empty list if nvidia-smi is missing, errors, or times
        out (select_gpu_profile treats empty as "no GPU" and the caller aborts).
    """
    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY, capture_output=True, check=False, timeout=NVIDIA_SMI_TIMEOUT_SECONDS
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_nvidia_smi(result.stdout.decode(errors="replace"))


def _placement_env(llm_gpu: GPUInfo, small_gpu: GPUInfo, kvcache_percent: str) -> dict[str, str]:
    """Build the env overrides for a dedicated LLM GPU + a small-model GPU."""
    return {
        "GPT_OSS_GPU": str(llm_gpu.index),
        "GPT_OSS_KVCACHE_PERCENT": kvcache_percent,
        "AUTOCOMPLETE_GPU": str(small_gpu.index),
        "AUTOCOMPLETE_GPU_MEM_UTIL": "0.3",
        "BGE_GPU": str(small_gpu.index),
        "BGE_GPU_MEM_UTIL": "0.05",
        "RERANK_GPU": str(small_gpu.index),
    }


def select_gpu_profile(gpus: list[GPUInfo]) -> tuple[str, dict[str, str]]:
    """Choose GPU-placement env overrides from detected hardware.

    Returns ``(profile_name, env_overrides)``:
      * ``single-gpu`` (empty overrides) — use compose defaults; one adequate GPU
        (e.g. DGX Spark).
      * ``multi-gpu`` — dedicate a GPU to the LLM (the largest that can run it), a
        second to the small models.
      * ``no-gpu`` — nvidia-smi reported nothing usable; the caller should abort.
      * ``unsupported`` — GPUs present but too small or too old for the LLM; abort.
    """
    if len(gpus) == 0:
        # No usable GPU (nvidia-smi missing/errored/timed out, or zero GPUs). The
        # whole stack is GPU-bound and the driver step already ran, so an empty
        # result here is a real problem — fail fast rather than crash at serve time.
        return "no-gpu", {}

    # Sort by memory, largest first; unknown memory (None) ranks last.
    ranked = sorted(gpus, key=lambda g: g.memory_mib if g.memory_mib is not None else -1, reverse=True)

    if len(gpus) == 1:
        # One GPU runs everything, so it must be new enough for the LLM and hold the
        # LLM + small models together. If its memory is unknown (e.g. GB10 unified
        # memory reports [N/A]), assume the Spark single-GPU layout and use compose
        # defaults; otherwise require it to fit the full stack, else abort early.
        only = gpus[0]
        fits = only.memory_mib is None or _gpu_meets(only.memory_mib, LLM_MIN_GPU_GB + SMALL_MODELS_MIN_GB)
        if _can_run_llm(only) and fits:
            return "single-gpu", {}
        return "unsupported", {}

    # Multi-GPU: dedicate one GPU to the LLM and put the small models on another.
    # Prefer the largest GPU for the LLM, but it must be new enough (MXFP4 needs
    # Hopper+) and meet the LLM floor — so if the largest is too old/small, fall
    # through to a lower-ranked eligible GPU rather than vetoing the layout. The small
    # models go on any other GPU that meets their floor (any arch is fine); an extra
    # tiny GPU that meets neither just sits idle.
    chosen_llm = next(
        (g for g in ranked if _can_run_llm(g) and _gpu_meets(g.memory_mib, LLM_MIN_GPU_GB)),
        None,
    )
    if chosen_llm is not None:
        chosen_small = next(
            (g for g in ranked if g is not chosen_llm and _gpu_meets(g.memory_mib, SMALL_MODELS_MIN_GB)),
            None,
        )
        if chosen_small is not None:
            return "multi-gpu", _placement_env(chosen_llm, chosen_small, "0.9")
    return "unsupported", {}


def select_gpt_oss_image_tag(gpus: list[GPUInfo]) -> str:
    """Pick the gpt-oss NIM image tag for the host.

    Unified-memory GPUs (GB10 / DGX Spark — the ones reporting memory as [N/A], i.e.
    ``memory_mib is None``) hit NIM 2.0.x's memory runaway that hangs the host, so
    pin 1.6.1 there. Discrete GPUs use 2.0.7 (2.0.x works, 1.6.1 doesn't).
    """
    if any(g.memory_mib is None for g in gpus):
        return GPT_OSS_IMAGE_TAG_UNIFIED
    return GPT_OSS_IMAGE_TAG_DISCRETE


# ---------------------------------------------------------------------------
# GH200-specific rerank model profile
# ---------------------------------------------------------------------------
#
# llama-nemotron-rerank-1b-v2:1.10.0 ships only an x86_64-built TensorRT plan for
# Hopper (compute capability 9.0). On an aarch64 cc-9.0 host (GH200 Grace Hopper)
# that prebuilt engine fails to deserialize ("platform tag mismatch"), yet NIM's
# tag-based selector matches on cc 9.0 alone — it can't tell x86_64 from aarch64 —
# and auto-selects the unrunnable plan. Pin the portable ONNX profile there.
# Gate on compute capability, not just aarch64: other aarch64 hosts are Blackwell
# (DGX Spark GB10 cc 12.x, GB200 cc 10.x), where the cc-9.0 plan isn't a candidate
# and NIM auto-selects a loadable profile — pinning would be wrong. cc 9.0 is the
# exact condition (and future-proof vs matching the "GH200" name).

RERANK_AARCH64_PROFILE = "f7391ddbcb95b2406853526b8e489fedf20083a2420563ca3e65358ff417b10f"
HOPPER_COMPUTE_CAPABILITY = "9.0"


def select_rerank_profile(machine: str, gpus: list[GPUInfo]) -> dict[str, str]:
    """Pin the rerank NIM to an ISA-compatible model profile on aarch64 Hopper.

    Args:
        machine: The host CPU architecture, as from ``platform.machine()``.
        gpus: Detected GPUs (their compute capability tells Hopper from Blackwell).

    Returns:
        ``{"NIM_MODEL_PROFILE": <onnx>}`` on an aarch64 cc-9.0 host (GH200), where the
        only cc-9.0 TensorRT plan is x86_64-built and won't deserialize; an empty dict
        otherwise (x86_64, and aarch64 non-Hopper like DGX Spark, auto-select fine).
    """
    if machine == "aarch64" and any(
        g.compute_capability == HOPPER_COMPUTE_CAPABILITY for g in gpus
    ):
        return {"NIM_MODEL_PROFILE": RERANK_AARCH64_PROFILE}
    return {}


def _report_gpu_profile(gpus: list[GPUInfo], profile_name: str, overrides: dict[str, str]) -> None:
    if gpus:
        names = ", ".join(
            f"{g.name} ({g.memory_mib} MiB)" if g.memory_mib is not None else f"{g.name} (memory N/A)"
            for g in gpus
        )
        console.print(f"  Detected [cyan]{len(gpus)}[/cyan] GPU(s): [dim]{names}[/dim]")
    else:
        console.print("  [yellow]No GPUs detected via nvidia-smi.[/yellow]")

    console.print(f"  GPU placement profile = [cyan]{profile_name}[/cyan]")

    if profile_name == "unsupported":
        console.print(
            "  [yellow]This host's GPUs can't run the models (too small, or too old "
            "for the LLM — needs Hopper+).[/yellow]"
        )
    elif overrides:
        console.print(
            f"  [dim]LLM dedicated to GPU {overrides['GPT_OSS_GPU']}; "
            f"small models on GPU {overrides['AUTOCOMPLETE_GPU']}.[/dim]"
        )
        idle = len(gpus) - 2  # LLM GPU + the small-model GPU
        if idle > 0:
            console.print(f"  [dim]{idle} additional GPU(s) left idle.[/dim]")

    for key, value in overrides.items():
        console.print(f"    [dim]{key}={value}[/dim]")


# ---------------------------------------------------------------------------
# Step 6: Resolve environment variables
# ---------------------------------------------------------------------------


def _resolve_ngc_api_key(home_dir: Path) -> str:
    """NGC_API_KEY resolution waterfall: env -> ~/.ngc/config -> interactive prompt.

    Returns:
        The resolved NGC API key string.
    """
    # 1. Environment variable
    key = os.environ.get("NGC_API_KEY", "")
    if key.startswith("nvapi-"):
        console.print("  [dim]NGC_API_KEY found in environment.[/dim]")
        return key

    # 2. NGC CLI config
    ngc_config_path = home_dir / ".ngc" / "config"
    try:
        config = configparser.ConfigParser()
        config.read(ngc_config_path)
        key = config.get("CURRENT", "apikey", fallback="")
        if key.startswith("nvapi-"):
            console.print(f"  [dim]NGC_API_KEY found in {ngc_config_path}.[/dim]")
            return key
    except (configparser.Error, OSError):
        pass

    # 3. Interactive prompt. Always print the URL so a headless/SSH user can open
    # it on their own machine. Only try to launch a local browser when a display
    # is actually present — on a remote host webbrowser.open() typically returns
    # False (or worse, hangs shelling out to xdg-open) and never helps.
    console.print("  [yellow]NGC_API_KEY not found.[/yellow]")
    # Best-effort browser launch as a convenience; the printed link below is the
    # source of truth. Gated on _has_display() — an unconditional webbrowser.open()
    # on a headless host can launch a terminal browser (lynx/w3m) that takes over
    # this terminal and blocks setup until it's quit.
    opened = _has_display() and webbrowser.open(NGC_SETUP_URL)
    if opened:
        console.print("  Opened the NGC setup page in your browser.")
    console.print(f"  If it didn't open, go to [link]{NGC_SETUP_URL}[/link]")
    console.print("  Generate an API key (starts with 'nvapi-') and paste it below.")

    if not sys.stdin.isatty():
        console.print("  [red]No NGC_API_KEY and no interactive terminal to prompt.[/red]")
        console.print("  [red]Set NGC_API_KEY in the environment or ~/.ngc/config and re-run.[/red]")
        sys.exit(1)

    for _ in range(3):
        key = prompt_input("  NGC API Key: ")
        if key.startswith("nvapi-"):
            return key
        if key != "":
            console.print("  [yellow]Invalid key (must start with 'nvapi-'). Try again.[/yellow]")
        else:
            console.print("  [yellow]No input received.[/yellow]")

    console.print("  [red]Failed to obtain NGC_API_KEY after 3 attempts.[/red]")
    sys.exit(1)


def step_setup_env_vars(uid: int, gid: int, home: Path) -> dict[str, str]:
    print_step(6, "Resolve environment variables")

    ngc_api_key = _resolve_ngc_api_key(home)

    offline_dir = os.environ.get("OFFLINE_INFERENCE_DIR")
    if offline_dir is None:
        offline_dir = home / ".cache" / "offline_inference"
        console.print(f"  [dim]OFFLINE_INFERENCE_DIR not set, using default: {offline_dir}[/dim]")

    offline_dir = Path(offline_dir)
    os.makedirs(offline_dir, exist_ok=True)
    os.chown(offline_dir, uid, gid)

    nim_cache_dir = offline_dir / "nim"
    os.makedirs(nim_cache_dir, exist_ok=True)
    os.chown(nim_cache_dir, uid, gid)

    user_id = str(uid)

    gpus = detect_gpus()
    profile_name, gpu_overrides = select_gpu_profile(gpus)
    rerank_overrides = select_rerank_profile(platform.machine(), gpus)
    gpt_oss_image_tag = select_gpt_oss_image_tag(gpus)

    env_vars = {
        "NGC_API_KEY": ngc_api_key,
        "OFFLINE_INFERENCE_DIR": str(offline_dir),
        "USER_ID": user_id,
        "GPT_OSS_IMAGE_TAG": gpt_oss_image_tag,
        **gpu_overrides,
        **rerank_overrides,
    }

    console.print(f"  OFFLINE_INFERENCE_DIR = [cyan]{offline_dir}[/cyan]")
    console.print(f"  USER_ID = [cyan]{user_id}[/cyan]")
    console.print(f"  gpt-oss NIM image tag = [cyan]{gpt_oss_image_tag}[/cyan]")
    _report_gpu_profile(gpus, profile_name, gpu_overrides)

    # Abort early instead of launching a layout that can't work and crashing deep
    # in startup (after the image pull and model download).
    if profile_name == "no-gpu":
        console.print(
            "  [red]No GPU detected. This deployment requires an NVIDIA GPU — confirm "
            "the driver is loaded (run `nvidia-smi`), then re-run.[/red]"
        )
        sys.exit(1)
    if profile_name == "unsupported":
        console.print(
            "  [red]This host's GPUs can't run the models (too small, or too old for "
            "the LLM — gpt-oss-120b needs Hopper+). See the hardware requirements in "
            "the README. Use suitable hardware, or set up tensor-parallel serving "
            "across GPUs in compose.yaml, then re-run.[/red]"
        )
        sys.exit(1)

    if rerank_overrides:
        console.print(
            f"  [dim]GH200 host: pinning rerank to ONNX profile "
            f"(NIM_MODEL_PROFILE={RERANK_AARCH64_PROFILE[:12]}…) — the x86_64 "
            f"TensorRT plan can't load here.[/dim]"
        )
    print_ok()
    return env_vars


# ---------------------------------------------------------------------------
# Step 7: Docker login to nvcr.io
# ---------------------------------------------------------------------------


def step_docker_login(ngc_api_key: str, docker_config_dir: Path) -> None:
    print_step(7, "Authenticate Docker with NGC registry")

    # Check if already logged in — inspect docker config file for nvcr.io auth entry.
    # Note: if Docker uses a credential helper (credStore), the auths dict may be empty
    # and this check will fall through to a redundant but harmless docker login.
    config_path = docker_config_dir / "config.json"
    try:
        with config_path.open() as f:
            docker_config = json.load(f)
        if NVCR_REGISTRY in docker_config.get("auths", {}):
            print_skipped(f"already logged in to {NVCR_REGISTRY}")
            return
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    run(
        ["docker", "login", NVCR_REGISTRY, "-u", "$oauthtoken", "--password-stdin"],
        stdin_data=ngc_api_key,
        status_msg=f"Logging in to {NVCR_REGISTRY}...",
    )

    print_ok()


# ---------------------------------------------------------------------------
# Step 8: Start services
# ---------------------------------------------------------------------------


KEY_SERVICES = {"gpt-oss-nim", "autocomplete-nim", "vllm-bge", "rerank-nim", "litellm"}
HEALTH_POLL_INTERVAL_SECONDS = 15


def _as_compose_ps_entries(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _parse_compose_ps_entries(output: bytes | str) -> list[dict[str, object]]:
    text = output.decode(errors="replace") if isinstance(output, bytes) else output
    text = text.strip()
    if text == "":
        return []

    try:
        return _as_compose_ps_entries(json.loads(text))
    except json.JSONDecodeError:
        pass

    entries: list[dict[str, object]] = []
    for line in text.splitlines():
        try:
            entries.extend(_as_compose_ps_entries(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return entries


def _get_compose_services(
    compose_cmd: list[str],
    env: dict[str, str],
    *,
    all_services: bool = False,
) -> list[dict[str, object]]:
    args = ["ps"]
    if all_services:
        args.append("-a")
    args.extend(["--format", "json"])

    result = run(compose_cmd + args, check=False, capture=True, env=env)
    if result.returncode != 0 or result.stdout is None:
        return []
    return _parse_compose_ps_entries(result.stdout)


def _get_running_services(compose_cmd: list[str], env: dict[str, str]) -> set[str]:
    """Query docker compose for currently running services.

    Returns:
        Set of service names that are in "running" state.
    """
    running: set[str] = set()
    for svc in _get_compose_services(compose_cmd, env):
        service = svc.get("Service")
        if svc.get("State") == "running" and isinstance(service, str) and service != "":
            running.add(service)
    return running


def _all_compose_services_stopped(services: list[dict[str, object]]) -> bool:
    if not services:
        return False
    active_states = {"created", "running", "restarting"}
    return all(svc.get("State") not in active_states for svc in services)


def _start_compose_logs(compose_cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        compose_cmd + ["logs", "-f"],
        env=env,
    )


def _start_and_stream_logs(compose_cmd: list[str], env: dict[str, str]) -> bool:
    """Start services detached, then stream all logs until healthy or failed.

    Returns:
        True if all key services are running, False otherwise.
    """
    merged_env = {**os.environ, **env}

    # Start detached in background — `up -d` blocks on depends_on conditions
    # (model-prep, healthchecks) so we can't wait for it. Run it as a background
    # process and stream logs in parallel.
    up_proc = subprocess.Popen(
        compose_cmd + ["up", "-d", "--remove-orphans"],
        env=merged_env,
        stdout=subprocess.DEVNULL,
        # Compose renders container create/wait progress on stderr. Suppress it
        # while logs are streaming so the two live views do not interleave.
        stderr=subprocess.DEVNULL,
    )

    # Give compose a moment to create containers before tailing logs.
    # `docker compose logs -f` can exit immediately when no containers exist yet.
    time.sleep(2)

    logs_proc: subprocess.Popen | None = None
    if _get_compose_services(compose_cmd, env, all_services=True):
        logs_proc = _start_compose_logs(compose_cmd, merged_env)
    else:
        console.print("  [dim]Waiting for compose to create containers before streaming logs...[/dim]")

    healthy = False
    try:
        while True:
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
            services = _get_compose_services(compose_cmd, env, all_services=True)

            if logs_proc is None and services:
                logs_proc = _start_compose_logs(compose_cmd, merged_env)
            elif logs_proc is not None and logs_proc.poll() is not None and up_proc.poll() is None:
                logs_proc = _start_compose_logs(compose_cmd, merged_env)

            running = {
                svc.get("Service")
                for svc in services
                if svc.get("State") == "running" and isinstance(svc.get("Service"), str)
            }
            if KEY_SERVICES.issubset(running):
                healthy = True
                break

            up_returncode = up_proc.poll()
            if up_returncode is not None and up_returncode != 0:
                break
            if up_returncode is not None and not services:
                break
            if up_returncode is not None and _all_compose_services_stopped(services):
                break
    except KeyboardInterrupt:
        console.print("\n  [yellow]Interrupted — services stay running in background.[/yellow]")
    finally:
        procs = [proc for proc in (logs_proc, up_proc) if proc is not None]
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    return healthy


def step_start_services(env_vars: dict[str, str]) -> None:
    print_step(8, "Start inference services")

    compose_cmd = ["docker", "compose"] + COMPOSE_FILES

    # Check if services are already running
    if KEY_SERVICES.issubset(_get_running_services(compose_cmd, env_vars)):
        print_skipped("all key services already running")
        return

    # Clean up stale containers from prior interrupted runs
    run(
        compose_cmd + ["down", "--remove-orphans"],
        env=env_vars,
        status_msg="Cleaning up previous containers...",
    )

    console.print("  [dim]Pulling container images...[/dim]\n")
    pull_result = run_streaming(compose_cmd + ["pull"], check=False, env=env_vars)
    if pull_result.returncode != 0:
        print_failed("could not pull all container images")
        console.print(
            "  [yellow]Docker keeps completed image layers. "
            "Re-run setup after fixing network or registry access.[/yellow]"
        )
        sys.exit(1)

    console.print("  [dim]Streaming all service logs (Ctrl+C to stop early)...[/dim]\n")

    healthy = _start_and_stream_logs(compose_cmd, env_vars)

    if healthy:
        print_ok("all key services running")
    else:
        print_failed("not all services came up healthy")

        # Show container status so the user can see which services failed
        merged_env = {**os.environ, **env_vars}
        console.print("\n  [bold]Container status:[/bold]")
        subprocess.run(
            compose_cmd + ["ps", "-a"],
            env=merged_env,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        # Show recent logs from non-running services to surface the actual errors
        result = subprocess.run(
            compose_cmd + ["ps", "-a", "--format", "json"],
            env=merged_env,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            failed_services: list[str] = []
            for svc in _parse_compose_ps_entries(result.stdout):
                if svc.get("State") not in ("running",):
                    name = svc.get("Service", "")
                    if isinstance(name, str) and name:
                        failed_services.append(name)
            if failed_services:
                console.print(f"\n  [bold]Recent logs from failed services ({', '.join(failed_services)}):[/bold]")
                subprocess.run(
                    compose_cmd + ["logs", "--tail=30"] + failed_services,
                    env=merged_env,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_root()

    # This installer is apt-based (Ubuntu/Debian/DGX OS). Fail fast on anything
    # else rather than dying mid-run with a cryptic "apt-get not found".
    if not run_silent(["apt-get", "--version"]):
        console.print("[red]This script supports apt-based distributions (Ubuntu/Debian/DGX OS).[/red]")
        console.print("[red]apt-get not found — your platform isn't supported. Install Docker, the[/red]")
        console.print("[red]NVIDIA Container Toolkit, and run 'docker compose up' manually.[/red]")
        sys.exit(1)

    username, uid, gid, home = get_invoking_user()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    # Verify required files exist
    required_files = ["compose.yaml", "litellm-config.yaml", "model-prep.sh"]
    for required_file in required_files:
        _path = script_dir / required_file
        if not _path.is_file():
            console.print(f"[red]Error: Required file '{_path.name}' not found in {script_dir}[/red]")
            sys.exit(1)

    console.print(
        Panel(
            Text.assemble(
                ("Nsight Copilot Offline Setup\n", "bold"),
                (f"Target user: {username} (uid={uid})\n", ""),
                (f"Working directory: {script_dir}", "dim"),
            ),
            border_style="blue",
        )
    )

    step_install_docker()
    step_configure_docker_no_sudo(username, home, uid, gid)
    step_install_nvidia_driver()
    step_install_nvidia_ctk()
    step_configure_nvidia_ctk()
    env_vars = step_setup_env_vars(uid, gid, home)
    drop_privileges(username, uid, gid)
    docker_config_dir = configure_user_environment(username, home)
    step_docker_login(env_vars["NGC_API_KEY"], docker_config_dir)
    step_start_services(env_vars)

    console.print()
    console.print(
        Panel(
            "[bold green]Setup complete![/bold green]\n"
            "The Nsight Copilot API will be available at [link]http://localhost:8080[/link] "
            "once all services are healthy.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
