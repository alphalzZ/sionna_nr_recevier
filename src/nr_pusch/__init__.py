"""Configuration-driven Sionna NR PUSCH tools."""

import os
from pathlib import Path
import tempfile

# Sionna's optional plotting imports may initialize Matplotlib. Keep its cache
# out of a possibly read-only ~/.config and outside the repository checkout.
if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / f"nr-pusch-matplotlib-{os.getuid()}"
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

__version__ = "0.1.0"
