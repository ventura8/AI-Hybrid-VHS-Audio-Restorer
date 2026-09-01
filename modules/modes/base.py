"""Base interface and common primitives for audio restoration processing modes."""

import abc
from pathlib import Path
from typing import Any, Dict, Optional


class BaseRestorationMode(abc.ABC):
    """Abstract base class establishing standard pipeline lifecycle for restoration modes."""

    mode_name: str = "base"
    display_name: str = "Base Mode"

    @abc.abstractmethod
    def execute(
        self,
        work_dir: Path,
        original_wav: Path,
        video_path: Path,
        final_output_video: Path,
        video_dur: float,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Executes the complete restoration mode pipeline."""
        raise NotImplementedError

    @staticmethod
    def resolve_strategy_val(strategy: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
        """Extracts a configuration value from an optional acoustic strategy dict."""
        if not strategy or not isinstance(strategy, dict):
            return default
        return strategy.get(key, default)
