"""Check that the Python / PyTorch / CUDA stack works on the RTX 5060 Ti (Blackwell, sm_120).

Exits with code 1 (and prints the install command) if the GPU is not usable.
"""
from __future__ import annotations

import platform
import sys

import torch
import torchvision

REQUIRED_ARCH = "sm_120"
INSTALL_CMD = (
    "pip install --force-reinstall torch torchvision "
    "--index-url https://download.pytorch.org/whl/cu128   # or cu129 / cu130"
)


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}")
    print("Install a CUDA 12.8+ build of PyTorch with Blackwell support:")
    print(f"  {INSTALL_CMD}")
    sys.exit(1)


def gib(n_bytes: int) -> str:
    return f"{n_bytes / 2**30:.2f} GiB"


def train_step(dtype: torch.dtype, device: torch.device) -> float:
    """Tiny MLP forward/backward under autocast; returns the loss value."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 10)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    x = torch.randn(64, 256, device=device)
    y = torch.randint(0, 10, (64,), device=device)
    with torch.autocast("cuda", dtype=dtype):
        out = model(x)
        assert out.dtype == dtype, f"autocast produced {out.dtype}, expected {dtype}"
        loss = torch.nn.functional.cross_entropy(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    grads_ok = all(torch.isfinite(p.grad).all() for p in model.parameters())
    if not grads_ok:
        raise RuntimeError(f"non-finite gradients in {dtype}")
    return loss.item()


def main() -> None:
    print(f"Python       : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"torch        : {torch.__version__}")
    print(f"torchvision  : {torchvision.__version__}")
    print(f"CUDA (build) : {torch.version.cuda}")
    print(f"cuDNN        : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    archs = torch.cuda.get_arch_list()
    print(f"GPU          : {name}, compute capability sm_{cap[0]}{cap[1]}")
    print(f"Arch list    : {' '.join(archs)}")
    if f"sm_{cap[0]}{cap[1]}" not in archs:
        fail(f"this torch build has no kernels for sm_{cap[0]}{cap[1]}")
    if REQUIRED_ARCH not in archs:
        fail(f"{REQUIRED_ARCH} missing from torch arch list")

    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    ref = (a.double() @ b.double()).float()
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        c = (a.to(dt) @ b.to(dt)).float()
        rel = ((c - ref).norm() / ref.norm()).item()
        print(f"matmul {str(dt):15s}: rel. error vs fp64 = {rel:.2e}")
    torch.cuda.synchronize()

    for dt in (torch.float16, torch.bfloat16):
        print(f"train step {str(dt):15s}: loss = {train_step(dt, device):.4f} (grads finite)")

    # torchvision CUDA ops (needed later for DETR post-processing / eval)
    try:
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device=device)
        torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
        print("torchvision.ops.nms on CUDA: OK")
    except (RuntimeError, NotImplementedError) as e:
        print(f"[WARN] torchvision CUDA ops unavailable ({type(e).__name__}).")
        print("  torchvision is probably a CPU-only build. Fix with:")
        print(f"  pip install --force-reinstall --no-deps torchvision=="
              f"{torchvision.__version__.split('+')[0]} "
              f"--index-url https://download.pytorch.org/whl/cu{torch.version.cuda.replace('.', '')}")

    # Third-party native extensions: Windows Smart App Control can block unsigned DLLs.
    import importlib
    for mod in ("transformers", "transformers.models.detr.modeling_detr", "timm", "peft",
                "albumentations", "cv2", "pycocotools.cocoeval", "scipy.optimize",
                "torch.utils.tensorboard", "grpc"):
        try:
            m = importlib.import_module(mod)
            print(f"import {mod:40s}: OK {getattr(m, '__version__', '')}")
        except ImportError as e:
            print(f"[WARN] import {mod} failed: {e}")

    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU memory   : total {gib(total)}, free {gib(free)}, "
          f"allocated by torch {gib(torch.cuda.memory_allocated(device))}, "
          f"peak {gib(torch.cuda.max_memory_allocated(device))}")
    print("\n[OK] environment ready")


if __name__ == "__main__":
    main()
