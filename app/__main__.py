"""Run SmearScan with one command: builds the frontend if needed, then serves API + UI.

conda run -n medthesis python -m app                  # http://127.0.0.1:8000 (UI), /docs (API), CPU
conda run -n medthesis python -m app --device cuda    # GPU (only when no training needs it)
conda run -n medthesis python -m app --dev            # API only; run `npm run dev` in app/frontend
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

FRONTEND = Path(__file__).with_name("frontend")


def build_frontend(force: bool) -> None:
    dist = FRONTEND / "dist" / "index.html"
    if dist.is_file() and not force:
        return
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit(
            "npm not found: install Node.js 20+ (https://nodejs.org) to build the frontend, "
            "or run with --dev to serve the API only"
        )
    if not (FRONTEND / "node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="SmearScan web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dev", action="store_true", help="API only (frontend via the Vite dev server)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the frontend even if dist/ exists")
    args = ap.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # '' would unset it and leave the GPU visible
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.dev:
        build_frontend(args.rebuild)

    import uvicorn

    from app.main import create_app

    app = create_app(device=args.device, static_dir=None if args.dev else FRONTEND / "dist")
    print(f"SmearScan: http://{args.host}:{args.port}  (API docs: /docs, device: {args.device})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
