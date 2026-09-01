"""Processing modes package for AI Hybrid VHS Audio Restorer.

Exports individual restoration mode classes and the central mode registry.
"""

from .arnndn import ArnndnSpeechMode
from .auto_pure import AutoPureMode
from .auto_pure_linear import AutoPureLinearMode
from .base import BaseRestorationMode
from .cathar import CatharMode
from .denoise_only import DenoiseOnlyMode
from .hybrid import HybridMode
from .multipass import MultiPassMode
from .native_dsp import AutoFFmpegNativeMode, FFmpegNativeMode
from .registry import MODE_CLASS_MAP, get_mode_instance

__all__ = [
    "BaseRestorationMode",
    "AutoPureLinearMode",
    "CatharMode",
    "HybridMode",
    "MultiPassMode",
    "AutoPureMode",
    "DenoiseOnlyMode",
    "FFmpegNativeMode",
    "AutoFFmpegNativeMode",
    "ArnndnSpeechMode",
    "MODE_CLASS_MAP",
    "get_mode_instance",
]
