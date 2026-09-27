"""Launch the local simulation dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from nr_pusch.web import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local NR PUSCH simulation dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: local machine only)")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"), help="TOML profile directory")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/web"), help="Persistent run output directory")
    args = parser.parse_args()
    serve(args.host, args.port, args.config_dir, args.runs_dir)


if __name__ == "__main__":
    main()
