 from __future__ import annotations

 import logging
 import os
 from datetime import datetime
 from pathlib import Path
 from typing import Optional

 from rich.logging import RichHandler


 def _resolve_log_dir(output_dir: str | os.PathLike[str]) -> Path:
     path = Path(output_dir).expanduser().resolve()
     path.mkdir(parents=True, exist_ok=True)
     return path


 def configure_logging(
     output_dir: str | os.PathLike[str],
     experiment_name: str,
     level: int = logging.INFO,
     filename: Optional[str] = None,
 ) -> Path:
     """Configure Python logging with both Rich console and file handlers."""
     log_dir = _resolve_log_dir(output_dir) / "logs"
     log_dir.mkdir(parents=True, exist_ok=True)

     timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
     log_filename = filename or f"{experiment_name}-{timestamp}.log"
     log_path = log_dir / log_filename

     logging.basicConfig(
         level=level,
         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
         datefmt="%Y-%m-%d %H:%M:%S",
         handlers=[
             RichHandler(rich_tracebacks=True, markup=True),
             logging.FileHandler(log_path, encoding="utf-8"),
         ],
     )
     logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
     logging.info("Logging initialized. Output directory: %s", log_path.parent)
     return log_path

