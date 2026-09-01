"""Registry and factory resolving processing mode instances."""

from typing import Dict, Type

from .arnndn import ArnndnSpeechMode
from .auto_pure import AutoPureMode
from .auto_pure_linear import AutoPureLinearMode
from .base import BaseRestorationMode
from .cathar import CatharMode
from .denoise_only import DenoiseOnlyMode
from .hybrid import HybridMode
from .multipass import MultiPassMode
from .native_dsp import AutoFFmpegNativeMode, FFmpegNativeMode

MODE_CLASS_MAP: Dict[str, Type[BaseRestorationMode]] = {
    "auto_pure_linear": AutoPureLinearMode,
    "cathar": CatharMode,
    "cathar_vhs": CatharMode,
    "hybrid": HybridMode,
    "multipass_auto": MultiPassMode,
    "multipass": MultiPassMode,
    "auto_pure": AutoPureMode,
    "pure": AutoPureMode,
    "denoise_only": DenoiseOnlyMode,
    "ffmpeg_native": FFmpegNativeMode,
    "vhs_native": FFmpegNativeMode,
    "auto_ffmpeg_native": AutoFFmpegNativeMode,
    "auto_vhs_native": AutoFFmpegNativeMode,
    "arnndn_speech": ArnndnSpeechMode,
}


def get_mode_instance(target_mode: str) -> BaseRestorationMode:
    """Returns an instantiated restoration mode handler for the given mode identifier."""
    mode_cls = MODE_CLASS_MAP.get(target_mode, HybridMode)
    return mode_cls()
