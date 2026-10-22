#!/usr/bin/env python3
"""Install VideoFake deps on current Google Colab (Python 3.12+/3.13, CUDA 12/13)."""

from __future__ import annotations

import glob
import pathlib
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def pip_install(*args: str) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def patch_setup_get_version(setup_py: pathlib.Path) -> bool:
    """Fix Python 3.13: exec() no longer populates locals() in get_version()."""
    text = setup_py.read_text(encoding="utf-8")
    old = (
        "def get_version():\n"
        "    with open(version_file, 'r') as f:\n"
        "        exec(compile(f.read(), version_file, 'exec'))\n"
        "    return locals()['__version__']"
    )
    new = (
        "def get_version():\n"
        "    with open(version_file, 'r') as f:\n"
        "        _d = {}\n"
        "        exec(compile(f.read(), version_file, 'exec'), _d)\n"
        "    return _d['__version__']"
    )
    if old in text:
        setup_py.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched get_version in {setup_py}")
        return True
    if "return locals()['__version__']" in text:
        setup_py.write_text(
            text.replace("return locals()['__version__']", "return _d['__version__']")
            .replace(
                "exec(compile(f.read(), version_file, 'exec'))",
                "_d = {}\n        exec(compile(f.read(), version_file, 'exec'), _d)",
            ),
            encoding="utf-8",
        )
        print(f"Patched get_version (fallback) in {setup_py}")
        return True
    print(f"No get_version patch needed in {setup_py}")
    return False


def install_git_package(url: str, dest: pathlib.Path) -> None:
    if not dest.exists():
        run(["git", "clone", "--depth", "1", url, str(dest)])
    patch_setup_get_version(dest / "setup.py")
    pip_install("-e", str(dest), "--no-build-isolation")


def patch_torchvision_basicsr() -> None:
    old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new = "from torchvision.transforms.functional import rgb_to_grayscale"
    patterns = [
        "/usr/local/lib/python*/dist-packages/basicsr/data/degradations.py",
        "/content/BasicSR/basicsr/data/degradations.py",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = pathlib.Path(path)
            text = p.read_text(encoding="utf-8")
            if old in text:
                p.write_text(text.replace(old, new), encoding="utf-8")
                print("Patched torchvision import:", path)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    print("Python:", sys.version)

    # 1) Core packages first — must succeed even if enhancer deps fail
    pip_install("-r", str(root / "requirements-colab.txt"))

    # 2) basicsr / facexlib / gfpgan need Python 3.13 setup.py fixes
    install_git_package("https://github.com/XPixelGroup/BasicSR.git", pathlib.Path("/content/BasicSR"))
    install_git_package("https://github.com/xinntao/facexlib.git", pathlib.Path("/content/facexlib"))
    # gfpgan pulls basicsr again; install without deps then ensure import works
    pip_install("--no-deps", "gfpgan==1.3.8")
    pip_install("yapf", "lmdb", "tb-nightly", "addict", "future", "scipy")

    patch_torchvision_basicsr()

    import onnxruntime as ort

    print("onnxruntime:", ort.__version__)
    print("providers:", ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("WARNING: CUDAExecutionProvider missing — face swap may fall back to CPU.")

    try:
        import gfpgan  # noqa: F401
        import basicsr  # noqa: F401

        print("gfpgan/basicsr: OK")
    except Exception as exc:
        print("WARNING: face_enhancer deps incomplete:", exc)
        print("You can still run with Processor = face_swapper only.")


if __name__ == "__main__":
    main()
