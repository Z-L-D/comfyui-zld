"""
Block-level Spectrum hook for Wan 2.1 / 2.2 video models.

Wan architecture in ComfyUI:
  diffusion_model.blocks – main transformer blocks

Each block typically takes (x, context, timestep_emb, ...) and returns
the updated hidden state. We cache and forecast block outputs.
"""

import torch
import logging
from .spectrum_core import SpectrumState

log = logging.getLogger("ComfyUI-Spectrum")


def apply_spectrum_wan(model, state: SpectrumState):
    """
    Patch Wan transformer blocks for Spectrum acceleration.
    """
    diff = model.model.diffusion_model

    n_patched = 0

    # Wan models in ComfyUI expose blocks via .blocks
    if hasattr(diff, "blocks"):
        for idx, blk in enumerate(diff.blocks):
            _patch_wan_block(blk, f"wan_{idx}", state)
            n_patched += 1

    # Some Wan implementations use .transformer_blocks
    elif hasattr(diff, "transformer_blocks"):
        for idx, blk in enumerate(diff.transformer_blocks):
            _patch_wan_block(blk, f"wan_{idx}", state)
            n_patched += 1

    if n_patched == 0:
        log.warning(
            "Spectrum/Wan: could not find transformer blocks. "
            "Looked for .blocks and .transformer_blocks on the diffusion model. "
            "Block-level acceleration disabled; model-output mode still works."
        )
    else:
        log.info(f"Spectrum/Wan: patched {n_patched} blocks")


def _patch_wan_block(block, block_id: str, state: SpectrumState):
    """
    Patch a single Wan transformer block.  The approach is identical
    to the Flux hook's generic _patch_block – cache on full-compute
    steps, forecast on skip steps.
    """
    orig = block.forward
    meta = {"shape": None}

    def patched(*args, **kwargs):
        t = state.t_norm()
        fc = state.get_forecaster(block_id)

        if state.is_full:
            result = orig(*args, **kwargs)
            # Wan blocks typically return a single tensor
            if isinstance(result, tuple):
                # Some blocks return (output, extras) – cache only main output
                meta["shape"] = result[0].shape
                fc.push(t, result[0].reshape(-1))
            else:
                meta["shape"] = result.shape
                fc.push(t, result.reshape(-1))
            return result

        # Skip step
        predicted_flat = fc.forecast(t)

        if predicted_flat is None:
            result = orig(*args, **kwargs)
            if isinstance(result, tuple):
                meta["shape"] = result[0].shape
                fc.push(t, result[0].reshape(-1))
            else:
                meta["shape"] = result.shape
                fc.push(t, result.reshape(-1))
            return result

        if meta["shape"] is not None:
            predicted = predicted_flat.reshape(meta["shape"])
        else:
            predicted = predicted_flat

        return predicted

    block.forward = patched
    block._spectrum_orig = orig


def remove_spectrum_wan(model):
    """Restore original forward methods."""
    diff = model.model.diffusion_model
    for attr in ("blocks", "transformer_blocks"):
        if hasattr(diff, attr):
            for blk in getattr(diff, attr):
                if hasattr(blk, "_spectrum_orig"):
                    blk.forward = blk._spectrum_orig
                    del blk._spectrum_orig
