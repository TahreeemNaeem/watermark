#!/usr/bin/env python3
"""Environment diagnostics tailored for the watermark removal project."""

from __future__ import annotations

import os
import platform
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:  # Python 3.8+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - older Python fallback
    import importlib_metadata  # type: ignore


@dataclass
class PackageSpec:
    import_name: str
    label: str
    required: bool = True
    dist_name: Optional[str] = None
    note: Optional[str] = None


@dataclass
class PackageStatus:
    spec: PackageSpec
    ok: bool
    version: Optional[str] = None
    error: Optional[str] = None


SECTION_LINE = "=" * 60


def print_section(title: str) -> None:
    print(f"\n{SECTION_LINE}\n{title}\n{SECTION_LINE}")


def detect_virtualenv() -> Tuple[bool, Optional[str]]:
    prefix = sys.prefix
    if getattr(sys, "real_prefix", None):
        return True, prefix
    base_prefix = getattr(sys, "base_prefix", prefix)
    if base_prefix != prefix:
        return True, prefix
    return False, None


def parse_requirements(path: str) -> List[str]:
    packages: List[str] = []
    if not os.path.exists(path):
        return packages
    with open(path, "r", encoding="utf8") as req_file:
        for raw_line in req_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split(";")[0]
            token = token.split(" ")[0]
            for sep in ("==", ">=", "<=", "~=", "!="):
                token = token.split(sep)[0]
            if token:
                packages.append(token)
    return packages


def check_package(spec: PackageSpec) -> PackageStatus:
    try:
        module = __import__(spec.import_name)
    except Exception as exc:  # ImportError or runtime error on import
        return PackageStatus(spec=spec, ok=False, error=str(exc))

    version = getattr(module, "__version__", None)
    if not version and spec.dist_name:
        try:
            version = importlib_metadata.version(spec.dist_name)
        except importlib_metadata.PackageNotFoundError:
            version = None

    return PackageStatus(spec=spec, ok=True, version=version)


def summarize_packages(statuses: Iterable[PackageStatus]) -> Dict[str, List[PackageStatus]]:
    buckets: Dict[str, List[PackageStatus]] = {"ok": [], "missing": []}
    for status in statuses:
        bucket = "ok" if status.ok else "missing"
        buckets[bucket].append(status)
    return buckets


def emit_package_table(group: str, statuses: Iterable[PackageStatus]) -> None:
    rows = list(statuses)
    if not rows:
        return
    print_section(group)
    label_width = max(len(row.spec.label) for row in rows)
    for row in rows:
        label = row.spec.label.ljust(label_width)
        if row.ok:
            version = row.version or "version unknown"
            note = f" - {version}"
            if row.spec.note:
                note += f" ({row.spec.note})"
            print(f"[ OK ] {label}{note}")
        else:
            print(f"[MISS] {label} - {row.error or 'not installed'}")
            if row.spec.note:
                wrapper = textwrap.TextWrapper(width=72, initial_indent="        ", subsequent_indent="        ")
                print(wrapper.fill(f"Note: {row.spec.note}"))


def print_python_info() -> None:
    print_section("Python Interpreter")
    print(f"Executable : {sys.executable}")
    print(f"Version    : {sys.version.split()[0]}")
    print(f"Platform   : {platform.platform()}")
    in_venv, path = detect_virtualenv()
    if in_venv:
        print(f"Virtualenv: {path}")
    else:
        print("Virtualenv: not detected (for reproducibility it is recommended)")


def python_version_ok(min_version: Tuple[int, int] = (3, 9)) -> bool:
    current = sys.version_info[:2]
    return current >= min_version


def check_torch_capabilities() -> None:
    try:
        import torch
    except Exception:
        return

    print_section("PyTorch Capabilities")
    print(f"Torch version: {torch.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available : {'yes' if cuda_available else 'no'}")
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            device_name = f"unavailable ({exc})"
        print(f"CUDA device   : {device_name}")
        print(f"CUDA toolkit  : {torch.version.cuda or 'unknown'}")
    try:
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps_available = False
    print(f"Apple MPS     : {'yes' if mps_available else 'no'}")


def check_plugin_status() -> None:
    print_section("Plugin Status")
    plugin_name = os.environ.get("WMR_LAMA_PLUGIN", "plugins.lama_plugin")
    try:
        module = __import__(plugin_name, fromlist=["dummy"])
    except ModuleNotFoundError:
        print(f"[WARN] Plugin module '{plugin_name}' not found. The fast OpenCV path will be used.")
        return
    except Exception as exc:
        print(f"[WARN] Plugin '{plugin_name}' import failed: {exc}")
        return

    has_inpaint = hasattr(module, "inpaint")
    if has_inpaint:
        print(f"[ OK ] '{plugin_name}' loaded and exposes inpaint()")
    else:
        print(f"[WARN] '{plugin_name}' loaded but inpaint() is missing")


def check_hf_cache() -> None:
    print_section("Model Cache")
    env_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    default_home = os.path.expanduser("~/.cache/huggingface")
    cache_path = env_home or default_home
    exists = os.path.isdir(cache_path)
    status = "found" if exists else "missing"
    print(f"HuggingFace cache: {cache_path} ({status})")
    if not exists:
        print("You may need network access to download pretrained weights on first run.")


def main() -> None:
    print("Environment diagnostics for Watermark Removal")
    print("Run inside your active Python environment.")

    print_python_info()

    if not python_version_ok():
        print("\n[WARN] Python 3.9 or newer is recommended for this project.")

    requirement_specs = [
        PackageSpec("numpy", "NumPy", dist_name="numpy"),
        PackageSpec("PIL", "Pillow", dist_name="Pillow"),
        PackageSpec("cv2", "OpenCV", dist_name="opencv-python-headless"),
        PackageSpec("scipy", "SciPy", dist_name="scipy"),
        PackageSpec("tqdm", "tqdm"),
        PackageSpec("matplotlib", "Matplotlib", dist_name="matplotlib"),
        PackageSpec("sklearn", "scikit-learn", dist_name="scikit-learn"),
    ]

    optional_specs = [
        PackageSpec("torch", "PyTorch", required=False, dist_name="torch", note="Needed for AI engine and SD plugin"),
        PackageSpec("torchvision", "torchvision", required=False, dist_name="torchvision"),
        PackageSpec("diffusers", "diffusers", required=False, dist_name="diffusers", note="Required for Stable Diffusion plugin"),
        PackageSpec("transformers", "transformers", required=False, dist_name="transformers"),
        PackageSpec("accelerate", "accelerate", required=False, dist_name="accelerate"),
        PackageSpec("safetensors", "safetensors", required=False, dist_name="safetensors"),
        PackageSpec("gradio", "gradio", required=False, dist_name="gradio"),
        PackageSpec("fastapi", "fastapi", required=False, dist_name="fastapi"),
        PackageSpec("uvicorn", "uvicorn", required=False, dist_name="uvicorn"),
    ]

    core_statuses = [check_package(spec) for spec in requirement_specs]
    optional_statuses = [check_package(spec) for spec in optional_specs]

    emit_package_table("Core Dependencies", core_statuses)
    emit_package_table("Optional / Plugin Dependencies", optional_statuses)

    missing_core = [st for st in core_statuses if not st.ok]
    missing_optional = [st for st in optional_statuses if not st.ok]

    if missing_core or missing_optional:
        print_section("Installation Summary")
        if missing_core:
            print("Core packages missing:")
            for st in missing_core:
                print(f"  - {st.spec.label} (pip install {st.spec.dist_name or st.spec.import_name})")
        if missing_optional:
            print("Optional packages missing (install if you need the related features):")
            for st in missing_optional:
                pkg = st.spec.dist_name or st.spec.import_name
                feature = st.spec.note or ""
                suffix = f" - {feature}" if feature else ""
                print(f"  - {st.spec.label} (pip install {pkg}){suffix}")
    else:
        print_section("Installation Summary")
        print("All listed packages are available. You are ready to run all engines.")

    reqs = parse_requirements("requirements.txt")
    if reqs:
        print_section("requirements.txt Summary")
        print(f"Found requirements.txt with {len(reqs)} entries.")
        missing_from_reqs: List[str] = []
        for dist_name in reqs:
            try:
                importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                missing_from_reqs.append(dist_name)
        if missing_from_reqs:
            print("Packages from requirements.txt not installed:")
            for name in missing_from_reqs:
                print(f"  - {name}")
        else:
            print("All packages listed in requirements.txt are installed.")
    else:
        print_section("requirements.txt Summary")
        print("No requirements.txt file found next to this script.")

    check_torch_capabilities()
    check_plugin_status()
    check_hf_cache()

    print(f"\nWorking directory : {os.getcwd()}")
    print(f"PYTHONPATH       : {os.environ.get('PYTHONPATH', 'not set')}")
    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()

