"""
Block-level Spectrum hook for FLUX models.

FLUX architecture (from diffusers/ComfyUI):
  transformer_blocks       – double-stream joint attention (img + txt)
  single_transformer_blocks – single-stream post-merge blocks

Each block's forward is patched to:
  • Full-compute step → run normally, cache output
  • Skip step → return Chebyshev-forecasted output
"""

import torch
import logging
from .spectrum_core import SpectrumState

log = logging.getLogger("ComfyUI-Spectrum")


def apply_spectrum_flux(model, state: SpectrumState):
    """
    Walk the Flux diffusion model and patch every transformer block.
    """
    diff = model.model.diffusion_model

    n_double = 0
    if hasattr(diff, "transformer_blocks"):
        for idx, blk in enumerate(diff.transformer_blocks):
            _patch_block(blk, f"dbl_{idx}", state)
            n_double += 1

    n_single = 0
    if hasattr(diff, "single_transformer_blocks"):
        for idx, blk in enumerate(diff.single_transformer_blocks):
            _patch_block(blk, f"sgl_{idx}", state)
            n_single += 1

    log.info(f"Spectrum/Flux: patched {n_double} double + {n_single} single blocks")


def _patch_block(block, block_id: str, state: SpectrumState):
    """
    Generic block-level patch.  Works regardless of whether the block
    returns a single tensor or a tuple (double-stream returns two tensors).

    We serialise tuples into a flat cache tensor and reconstruct on forecast.
    """
    orig = block.forward

    # Metadata for shape reconstruction (populated on first full-compute call)
    meta = {"shapes": None, "is_tuple": False}

    def patched(*args, **kwargs):
        t = state.t_norm()
        fc = state.get_forecaster(block_id)

        if state.is_full:
            result = orig(*args, **kwargs)

            # Cache
            if isinstance(result, tuple):
                meta["is_tuple"] = True
                meta["shapes"] = [r.shape for r in result]
                flat = torch.cat([r.reshape(-1) for r in result])
                fc.push(t, flat)
            else:
                meta["is_tuple"] = False
                meta["shapes"] = [result.shape]
                fc.push(t, result.reshape(-1))

            return result

        # ---- Skip step ----
        predicted_flat = fc.forecast(t)

        if predicted_flat is None:
            # Not enough data yet, fall through to real compute
            result = orig(*args, **kwargs)
            if isinstance(result, tuple):
                meta["is_tuple"] = True
                meta["shapes"] = [r.shape for r in result]
                flat = torch.cat([r.reshape(-1) for r in result])
                fc.push(t, flat)
            else:
                meta["is_tuple"] = False
                meta["shapes"] = [result.shape]
                fc.push(t, result.reshape(-1))
            return result

        # Reconstruct from flat prediction
        if meta["is_tuple"] and meta["shapes"] is not None:
            tensors = []
            offset = 0
            for shp in meta["shapes"]:
                numel = 1
                for s in shp:
                    numel *= s
                tensors.append(predicted_flat[offset:offset + numel].reshape(shp))
                offset += numel
            return tuple(tensors)
        elif meta["shapes"] is not None:
            return predicted_flat.reshape(meta["shapes"][0])
        else:
            return predicted_flat

    block.forward = patched
    block._spectrum_orig = orig


def remove_spectrum_flux(model):
    """Restore original forward methods."""
    diff = model.model.diffusion_model
    for attr in ("transformer_blocks", "single_transformer_blocks"):
        if hasattr(diff, attr):
            for blk in getattr(diff, attr):
                if hasattr(blk, "_spectrum_orig"):
                    blk.forward = blk._spectrum_orig
                    del blk._spectrum_orig
