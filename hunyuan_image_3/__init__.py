# HunyuanImage-3.0: A Powerful Native Multimodal Model for T2T, T2I, TI2T, TI2I
"""
HunyuanImage-3.0 Package

This package provides the implementation of HunyuanImage-3.0, a unified multimodal model
that supports multiple generation tasks:
- Text-to-Text: Text generation from text prompts
- Text-to-Image: Image generation from text prompts
- Text & Image to Text: Text generation from text and image inputs
- Text & Image to Image: Image generation from text and image inputs

The package uses lazy loading to optimize import performance and reduce memory usage.
"""

from typing import TYPE_CHECKING

from utils import _LazyModule
from utils.import_utils import define_import_structure

# TYPE_CHECKING is used to provide type hints for static type checkers without
# actually importing the modules at runtime, which helps with circular imports
# and reduces startup time.
if TYPE_CHECKING:
    from .configuration_hunyuan_image_3 import *
    from .modeling_hunyuan_image_3 import *
    from .autoencoder_kl_3d import *
    from .image_processor import *
    from .siglip2 import *
    from .tokenization_hunyuan_image_3 import *
else:
    # At runtime, use lazy loading to defer module imports until they are actually accessed.
    # This improves startup time and reduces memory footprint.
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
