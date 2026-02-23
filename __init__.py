# __init__.py - ComfyUI EMAG Custom Node Package
"""
EMAG (Exponential Moving Average Guidance) Custom Node for ComfyUI

This custom node implements the EMAG guidance technique from the paper:
"EMAG: Exponential Moving Average Guidance for Diffusion Models"

EMAG perturbs attention maps using exponential moving averages to create
"hard negatives" that improve image quality and prompt adherence without
additional training.

Installation:
    1. Place this folder in ComfyUI/custom_nodes/
    2. Restart ComfyUI
    3. Find "EMAG Guider" in sampling/custom_sampling/guiders category

Usage:
    Use as a drop-in replacement for CFGGuider with SamplerCustomAdvanced:
    - Connect model, positive, negative conditioning
    - Set CFG scale (standard) and EMAG scale (1.75 recommended for conditional)
    - Adjust EMA decay (0.9 default) and schedule percentages

For more details, see the paper at: https://arxiv.org/abs/2512.17303
"""

from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "EMAG Implementation for ComfyUI"
__description__ = "Exponential Moving Average Guidance for improved diffusion sampling"

print("[EMAG] Loaded EMAG Guider node successfully")
print(f"[EMAG] Version {__version__}")
print("[EMAG] EMAG perturbs attention with EMA to create hard negatives")
print("[EMAG] Recommended settings: cfg=7.0, emag_scale=1.75 (conditional), ema_decay=0.9")