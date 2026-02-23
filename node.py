# node.py

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.clip_vision
import comfy.model_management as mm
import numpy as np
import folder_paths
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Any, List
from collections import deque
import logging

try:
    import comfy.ldm.modules.attention as comfy_attention
except ImportError:
    comfy_attention = None

from .deconv_engine import (
    make_disk_psf,
    make_gaussian_psf,
    make_motion_psf,
    make_motion_psf_tapered,
    estimate_blur_width,
    estimate_motion_params,
    deconvolve_image,
    deconvolve_channel,
)


class EMAGGuider:
    """
    EMAG (Exponential Moving Average Guidance) Guider Node
    A drop-in replacement for CFGGuider that applies EMA-based attention perturbation
    to create hard negatives for improved guidance.
    
    Based on: "EMAG: Exponential Moving Average Guidance for Diffusion Models"
    Paper equations implemented: Eq. 12 (EMA), Eq. 15 (EMAG update), Eq. 16 (CFG with EMAG)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {
                    "default": 7.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.1, 
                    "round": 0.01,
                    "tooltip": "Classifier-Free Guidance scale (standard CFG)"
                }),
                "emag_scale": ("FLOAT", {
                    "default": 1.75, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "EMAG guidance scale (w_e in paper). Recommended: 1.75 for conditional, 5.125 for unconditional"
                }),
                "ema_decay": ("FLOAT", {
                    "default": 0.9, 
                    "min": 0.0, 
                    "max": 0.999, 
                    "step": 0.001,
                    "tooltip": "EMA decay factor (lambda in Eq. 12). Higher = more smoothing"
                }),
                "start_percent": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Start applying EMAG at this percent of total steps (1.0 = beginning)"
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Stop applying EMAG at this percent of total steps (0.2 = last 20%)"
                }),
            },
            "optional": {
                "adaptive_layers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use adaptive layer selection (recommended, see paper Section 4.3)"
                }),
                "perturb_img_to_text": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "EMAG (Full)", 
                    "label_off": "EMAG-I (Img Only)",
                    "tooltip": "EMAG perturbs both Image->Image and Image->Text. EMAG-I only perturbs Image->Image"
                }),
            }
        }
    
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"
    DESCRIPTION = "EMAG: Exponential Moving Average Guidance - perturbs attention with EMA to create hard negatives for improved image quality and prompt adherence"
    
    def get_guider(self, model, positive, negative, cfg, emag_scale, ema_decay, 
                   start_percent, end_percent, adaptive_layers=True, perturb_img_to_text=True):
        """Create and return the EMAG guider instance"""
        guider = EMAGGuiderImpl(
            model_patcher=model,
            cfg=cfg,
            emag_scale=emag_scale,
            ema_decay=ema_decay,
            start_percent=start_percent,
            end_percent=end_percent,
            adaptive_layers=adaptive_layers,
            perturb_img_to_text=perturb_img_to_text
        )
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)


class EMAGGuiderImpl(comfy.samplers.CFGGuider):
    """
    Implementation of EMAG guidance logic - inherits from CFGGuider
    to ensure compatibility with SamplerCustomAdvanced.
    
    EMA state is persisted on the model_patcher object so it survives
    guider recreation across separate sampler calls (e.g. scheduled CFG
    workflows that run multiple 1-step samplers).
    """
    
    # Class-level key for storing EMA state on the model_patcher
    _EMA_STATE_KEY = '_emag_ema_state'
    _EMA_STEP_KEY = '_emag_step_counter'
    _EMA_FIRST_TIMESTEP_KEY = '_emag_first_timestep'
    
    def __init__(self, model_patcher, cfg, emag_scale, ema_decay, 
                 start_percent, end_percent, adaptive_layers, perturb_img_to_text):
        # Call parent init to set up model_patcher and other required attributes
        super().__init__(model_patcher)
        
        # Store EMAG specific parameters
        self.emag_scale = emag_scale
        self.ema_decay = ema_decay
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.adaptive_layers = adaptive_layers
        self.perturb_img_to_text = perturb_img_to_text
        
        # Hook handles for cleanup (these are transient per-call)
        self._hook_handles = []
        
        # Total steps tracking (will be inferred or set)
        self.total_steps = None
        
    def _get_persistent_ema(self):
        """
        Get or create the persistent EMA state dictionary stored on the model_patcher.
        This survives guider recreation across separate sampler calls.
        """
        if not hasattr(self.model_patcher, self._EMA_STATE_KEY):
            self.model_patcher._emag_ema_state = {}
        return self.model_patcher._emag_ema_state
    
    def _get_persistent_step(self):
        """Get the persistent step counter from the model_patcher."""
        if not hasattr(self.model_patcher, self._EMA_STEP_KEY):
            self.model_patcher._emag_step_counter = 0
        return self.model_patcher._emag_step_counter
    
    def _set_persistent_step(self, step):
        """Set the persistent step counter on the model_patcher."""
        self.model_patcher._emag_step_counter = step
        
    def _detect_new_generation(self, timestep):
        """
        Detect if this is a new generation (not just a new step in the same generation).
        
        Heuristic: if the current timestep is >= the last first-timestep we saw,
        this is likely a brand new generation and we should reset EMA state.
        Also resets if no prior state exists.
        """
        timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else float(timestep)
        
        if not hasattr(self.model_patcher, self._EMA_FIRST_TIMESTEP_KEY):
            # First ever call - record the starting timestep
            self.model_patcher._emag_first_timestep = timestep_val
            self.model_patcher._emag_last_timestep = timestep_val
            return True
        
        last_timestep = getattr(self.model_patcher, '_emag_last_timestep', 0.0)
        
        # In diffusion, timesteps decrease over the course of a generation.
        # If current timestep is significantly higher than the last one we saw,
        # we've started a new generation.
        if timestep_val > last_timestep + 0.01:
            # New generation detected - reset
            print(f"[EMAG] New generation detected (timestep {timestep_val:.4f} > last {last_timestep:.4f}). Resetting EMA state.")
            self.model_patcher._emag_first_timestep = timestep_val
            self.model_patcher._emag_last_timestep = timestep_val
            return True
        
        # Same generation, advancing forward
        self.model_patcher._emag_last_timestep = timestep_val
        return False
        
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """
        Override predict_noise to implement EMAG guidance.
        Implements Eq. 15 and 16 from the EMAG paper:
        
        Eq. 15: ε̃(x_t, c)' = ε'_θ(x_t, c) + w_e · (ε_θ(x_t, c) - ε'_θ(x_t, c))
        Eq. 16: ε̃(x_t, c) = ε_θ(x_t, ∅) + w · (ε̃(x_t, c)' - ε_θ(x_t, ∅))
        
        where ε'_θ is the perturbed (EMA) prediction and w_e is emag_scale
        """
        # Detect new generation and reset if needed
        is_new_gen = self._detect_new_generation(timestep)
        if is_new_gen:
            self._get_persistent_ema().clear()
            self._set_persistent_step(0)
        
        # Get persistent state
        ema_attention = self._get_persistent_ema()
        current_step = self._get_persistent_step()
        
        # Infer total steps if not set
        if self.total_steps is None and 'sigmas' in model_options:
            self.total_steps = len(model_options['sigmas']) - 1
        
        # Check if EMAG should be applied at this step
        apply_emag = self._should_apply_emag(current_step)
        
        print(f"[EMAG] step={current_step}, timestep={timestep.item():.4f}, apply={apply_emag}, ema_keys={len(ema_attention)}")
        
        if not apply_emag:
            # Standard CFG without EMAG - use parent's implementation
            result = super().predict_noise(x, timestep, model_options, seed)
            self._set_persistent_step(current_step + 1)
            return result
        
        # Get conditioning
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        
        # Register EMA hooks before forward pass
        self._register_emag_hooks(ema_attention)
        
        try:
            # Get predictions with EMA perturbation active (hooks registered)
            # This gives us ε'_θ(x_t, c) - the perturbed conditional prediction
            out_perturbed = comfy.samplers.calc_cond_batch(
                self.inner_model, 
                [negative_cond, positive_cond], 
                x, 
                timestep, 
                model_options
            )
            cond_pred_perturbed = out_perturbed[1]  # positive with EMA
            uncond_pred = out_perturbed[0]  # negative (unconditional)
            
        finally:
            # Remove hooks before standard pass
            self._remove_emag_hooks()
        
        # Get standard conditional prediction without EMA
        # This gives us ε_θ(x_t, c) - the standard conditional prediction
        out_standard = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],  # Only need positive
            x,
            timestep,
            model_options
        )
        cond_pred_standard = out_standard[0]
        
        # Check if perturbation is actually happening
        diff_norm = (cond_pred_standard - cond_pred_perturbed).norm().item()
        print(f"[EMAG] Perturbation magnitude: {diff_norm:.6f}")
        
        # Eq. 15: First step of EMAG guidance
        # ε̃(x_t, c)' = ε'_θ(x_t, c) + w_e · (ε_θ(x_t, c) - ε'_θ(x_t, c))
        emag_guidance = cond_pred_standard + self.emag_scale * (cond_pred_standard - cond_pred_perturbed)
        
        # Eq. 16: Standard CFG with EMAG-enhanced conditioning
        # ε̃(x_t, c) = ε_θ(x_t, ∅) + w · (ε̃(x_t, c)' - ε_θ(x_t, ∅))
        noise_pred = uncond_pred + self.cfg * (emag_guidance - uncond_pred)
        
        # Advance persistent step counter
        self._set_persistent_step(current_step + 1)
        return noise_pred
    
    def _should_apply_emag(self, current_step):
        """
        Check if EMAG should be applied based on current step and schedule.
        Paper uses "tail schedule": apply at high noise levels (early timesteps).
        """
        if self.total_steps is None or self.total_steps == 0:
            return True  # Default to applying if we don't know step count
        
        # Convert percentages to step indices
        start_step = int(self.start_percent * self.total_steps)
        end_step = int(self.end_percent * self.total_steps)
        
        # Apply EMAG between end_step and start_step
        # For diffusion: step 0 = highest noise, so start_percent=1.0 means beginning
        return end_step <= current_step <= start_step
    
    def _register_emag_hooks(self, ema_attention):
        """
        Register forward hooks to apply EMA perturbation to attention layers.
        Implements Eq. 12: eA_t = (1 - λ) * A_t + λ * E_t
        where eA_t is the EMA attention used as perturbation.
        
        Args:
            ema_attention: The persistent EMA state dictionary
        """
        self._remove_emag_hooks()  # Clear any existing hooks first
        
        try:
            # Access the model's diffusion_model
            model = self.model_patcher.model
            
            # Find transformer blocks - handles different architectures
            blocks = self._find_transformer_blocks(model)
            
            if blocks is None:
                print("[EMAG] Warning: Could not find transformer blocks in model")
                return
            
            # Select which layers to perturb
            if self.adaptive_layers:
                layers_to_perturb = self._select_layers_adaptive(blocks)
            else:
                layers_to_perturb = list(range(len(blocks)))
            
            # Register hooks on selected layers
            for idx in layers_to_perturb:
                if idx >= len(blocks):
                    continue
                
                block = blocks[idx]
                self._hook_attention_modules(block, idx, ema_attention)
            
            print(f"[EMAG] Registered {len(self._hook_handles)} hooks on layers {layers_to_perturb}")
                        
        except Exception as e:
            print(f"[EMAG] Warning: Could not register hooks: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_transformer_blocks(self, model):
        """
        Find transformer blocks in the model, handling different architectures.
        Supports: SD1.5/SDXL (UNet), SD3/Flux (DiT), LTX2/LTXAV.
        """
        # Try through diffusion_model first
        if hasattr(model, 'diffusion_model'):
            dm = model.diffusion_model
            for attr in ('transformer_blocks', 'joint_blocks', 'blocks'):
                if hasattr(dm, attr):
                    return getattr(dm, attr)
        
        # Try directly on model
        for attr in ('transformer_blocks', 'joint_blocks', 'blocks'):
            if hasattr(model, attr):
                return getattr(model, attr)
        
        return None
    
    def _hook_attention_modules(self, block, layer_idx, ema_attention):
        """
        Find and hook attention modules on a transformer block.
        Handles multiple naming conventions across architectures:
        - SD1.5/SDXL: attn1 (self), attn2 (cross)
        - Flux/SD3: attn (combined)
        - LTX2/LTXAV: attn1 (self), attn2 (cross), plus audio variants
        """
        # Self-attention (Image->Image) - try multiple attribute names
        for attr_name in ('attn1', 'attn', 'self_attn'):
            if hasattr(block, attr_name):
                module = getattr(block, attr_name)
                handle = module.register_forward_hook(
                    self._make_emag_hook(layer_idx, 'self', ema_attention)
                )
                self._hook_handles.append(handle)
                break
        
        # Cross-attention (Image->Text) if enabled
        if self.perturb_img_to_text:
            for attr_name in ('attn2', 'cross_attn'):
                if hasattr(block, attr_name):
                    module = getattr(block, attr_name)
                    handle = module.register_forward_hook(
                        self._make_emag_hook(layer_idx, 'cross', ema_attention)
                    )
                    self._hook_handles.append(handle)
                    break
    
    def _remove_emag_hooks(self):
        """Remove all registered forward hooks"""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except:
                pass
        self._hook_handles.clear()
    
    def _make_emag_hook(self, layer_idx, attn_type, ema_attention):
        """
        Create a forward hook function for EMA perturbation.
        
        Implements Eq. 12 from the paper:
        eA_new = decay * eA_old + (1-decay) * A_current
        
        On first encounter: stores current attention as EMA, returns original.
        On subsequent calls: returns OLD EMA (the perturbation), then updates EMA.
        
        Args:
            layer_idx: Which transformer block layer
            attn_type: 'self' or 'cross'
            ema_attention: The persistent EMA state dictionary (stored on model_patcher)
        """
        decay = self.ema_decay
        
        def hook(module, input, output):
            # Extract attention output (can be tuple or tensor)
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None
            
            # Create unique key for this layer/type
            key = f"layer_{layer_idx}_{attn_type}"
            
            if key in ema_attention:
                ema_attn = ema_attention[key]
                
                # Ensure shape compatibility (batch size can change between steps)
                if ema_attn.shape == attn_output.shape:
                    # Store the old EMA before updating - this is what we return as perturbation
                    old_ema = ema_attn
                    
                    # Update EMA: eA_new = decay * eA_old + (1-decay) * A_current
                    # This is Eq. 12 where λ = decay
                    new_ema = (decay * ema_attn + (1.0 - decay) * attn_output.detach())
                    ema_attention[key] = new_ema
                    
                    # Return the OLD EMA as the perturbation
                    # This creates the "hard negative" by using temporally smoothed attention
                    if rest is not None:
                        return (old_ema,) + rest
                    else:
                        return old_ema
                else:
                    # Shape mismatch (resolution change, etc.) - reinitialize
                    print(f"[EMAG] Shape mismatch for {key}: stored {ema_attn.shape} vs current {attn_output.shape}. Reinitializing.")
                    ema_attention[key] = attn_output.detach().clone()
                    return output
            
            # Initialize EMA storage on first encounter
            ema_attention[key] = attn_output.detach().clone()
            
            # Return original output (no perturbation on first call - 
            # perturbation will activate starting from the next step)
            return output
        
        return hook
    
    def _select_layers_adaptive(self, blocks):
        """
        Adaptive layer selection strategy from paper Section 4.3.
        
        The paper found that perturbing middle-to-deep layers works best:
        - Small models (<=12 layers): layers 6-8
        - Large models (>12 layers): layers 12-15
        
        This balances FID and HPS metrics.
        """
        n_layers = len(blocks)
        
        if n_layers <= 12:
            # For smaller models, perturb middle layers
            start = max(0, n_layers // 2 - 1)
            end = min(n_layers, n_layers // 2 + 2)
            return list(range(start, end))
        else:
            # For larger models (SD3, DiT-XL, Flux, LTX2), perturb layers 12-15
            return list(range(12, min(n_layers, 16)))


# class LTXDualPromptEncoder:
#     """
#     Dual prompt encoder for LTX-2 that accepts a pre-loaded Gemma text encoder
#     and creates separate video/audio conditionings that get merged for the dual-stream model.
#     """
    
#     def __init__(self):
#         self.device = mm.get_torch_device()
        
#     @classmethod
#     def INPUT_TYPES(cls) -> Dict[str, Any]:
#         return {
#             "required": {
#                 "text_encoder": ("CLIP", {
#                     "tooltip": "Connect from LTXV Gemma CLIP Loader or similar text encoder loader"
#                 }),
#                 "video_prompt": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Visual description: cinematography, lighting, camera movement, scene composition...",
#                     "tooltip": "Prompt describing visual elements for the video stream"
#                 }),
#                 "audio_prompt": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Audio description: dialogue, sound effects, music, ambience...",
#                     "tooltip": "Prompt describing auditory elements for the audio stream"
#                 }),
#                 "merge_strategy": (["concatenate", "interleave", "video_weighted", "audio_weighted", "separate_tokens"], {
#                     "default": "concatenate",
#                     "tooltip": "How to merge video and audio embeddings for dual-stream conditioning"
#                 }),
#                 "video_weight": ("FLOAT", {
#                     "default": 1.0,
#                     "min": 0.0,
#                     "max": 2.0,
#                     "step": 0.1,
#                     "tooltip": "Weight for video prompt when using weighted strategies"
#                 }),
#                 "audio_weight": ("FLOAT", {
#                     "default": 1.0,
#                     "min": 0.0,
#                     "max": 2.0,
#                     "step": 0.1,
#                     "tooltip": "Weight for audio prompt when using weighted strategies"
#                 }),
#             },
#             "optional": {
#                 "sync_prompt": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Shared context for AV synchronization (optional)",
#                     "tooltip": "Optional shared context to ensure audio-video alignment"
#                 }),
#             }
#         }
    
#     RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
#     RETURN_NAMES = ("conditioning", "combined_prompt", "token_info")
#     FUNCTION = "encode_dual_prompts"
#     CATEGORY = "LTXVideo/dual_prompt"
#     DESCRIPTION = "Encodes separate video and audio prompts using a wired text encoder, merges them for LTX-2 dual-stream generation"
    
#     def encode_with_encoder(self, text_encoder, prompt: str):
#         """
#         Use the provided text encoder (Gemma) to encode the prompt.
#         Returns embeddings and attention mask.
#         """
#         if not prompt.strip():
#             # Return minimal embeddings for empty prompts
#             return torch.zeros((1, 1, 4096), device=self.device), torch.zeros((1, 1), device=self.device)
        
#         # Tokenize using the encoder's tokenizer
#         tokens = text_encoder.tokenize(prompt)
        
#         # Encode - this returns (cond, pooled) or similar depending on encoder type
#         encode_result = text_encoder.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        
#         # Extract embeddings - handle different return formats
#         if isinstance(encode_result, dict):
#             embeddings = encode_result.get("cond", encode_result.get("hidden_states", None))
#             # pooled = encode_result.get("pooled_output", None)
#         elif isinstance(encode_result, (tuple, list)):
#             embeddings = encode_result[0]
#         else:
#             embeddings = encode_result
        
#         # Ensure embeddings are on the correct device
#         if embeddings is not None:
#             embeddings = embeddings.to(self.device)
        
#         # Create attention mask based on actual token count
#         # Count non-padding tokens (assuming pad_token_id is 0 or tokenizer knows)
#         if hasattr(tokens, 'input_ids'):
#             token_ids = tokens.input_ids
#         elif isinstance(tokens, dict) and 'input_ids' in tokens:
#             token_ids = tokens['input_ids']
#         elif isinstance(tokens, torch.Tensor):
#             token_ids = tokens
#         else:
#             # Fallback: assume all tokens are real
#             token_ids = None
        
#         if token_ids is not None:
#             if len(token_ids.shape) == 1:
#                 token_ids = token_ids.unsqueeze(0)
#             # Assume padding tokens are 0 or use attention mask if available
#             mask = (token_ids != 0).long().to(self.device)
#         else:
#             # Create mask based on embedding length
#             seq_len = embeddings.size(1) if embeddings is not None else 1
#             mask = torch.ones((1, seq_len), device=self.device)
        
#         return embeddings, mask
    
#     def merge_embeddings(
#         self,
#         video_emb: torch.Tensor,
#         video_mask: torch.Tensor,
#         audio_emb: torch.Tensor,
#         audio_mask: torch.Tensor,
#         strategy: str,
#         v_weight: float,
#         a_weight: float,
#         sync_emb: torch.Tensor = None,
#         sync_mask: torch.Tensor = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Merge video and audio embeddings according to strategy.
#         CRITICAL: All tensors must be on the same device!
#         """
        
#         # Ensure all inputs are on the same device
#         target_device = video_emb.device
        
#         # Move everything to the same device (the video embeddings' device)
#         video_emb = video_emb.to(target_device)
#         video_mask = video_mask.to(target_device)
#         audio_emb = audio_emb.to(target_device)
#         audio_mask = audio_mask.to(target_device)
        
#         if sync_emb is not None:
#             sync_emb = sync_emb.to(target_device)
#             sync_mask = sync_mask.to(target_device)
        
#         if strategy == "concatenate":
#             # [video_emb | audio_emb]
#             merged_emb = torch.cat([video_emb, audio_emb], dim=1)
#             merged_mask = torch.cat([video_mask, audio_mask], dim=1)
            
#             if sync_emb is not None:
#                 merged_emb = torch.cat([sync_emb, merged_emb], dim=1)
#                 merged_mask = torch.cat([sync_mask, merged_mask], dim=1)
                
#         elif strategy == "interleave":
#             min_len = min(video_emb.size(1), audio_emb.size(1))
#             interleaved = []
#             mask_interleaved = []
            
#             for i in range(min_len):
#                 interleaved.extend([
#                     video_emb[:, i:i+1] * v_weight, 
#                     audio_emb[:, i:i+1] * a_weight
#                 ])
#                 mask_interleaved.extend([
#                     video_mask[:, i:i+1], 
#                     audio_mask[:, i:i+1]
#                 ])
            
#             # Handle remaining tokens
#             if video_emb.size(1) > min_len:
#                 interleaved.append(video_emb[:, min_len:] * v_weight)
#                 mask_interleaved.append(video_mask[:, min_len:])
#             elif audio_emb.size(1) > min_len:
#                 interleaved.append(audio_emb[:, min_len:] * a_weight)
#                 mask_interleaved.append(audio_mask[:, min_len:])
            
#             merged_emb = torch.cat(interleaved, dim=1)
#             merged_mask = torch.cat(mask_interleaved, dim=1)
            
#             if sync_emb is not None:
#                 merged_emb = torch.cat([sync_emb, merged_emb], dim=1)
#                 merged_mask = torch.cat([sync_mask, merged_mask], dim=1)
                
#         elif strategy == "video_weighted":
#             weighted_video = video_emb * v_weight
#             weighted_audio = audio_emb * a_weight * 0.5
#             merged_emb = torch.cat([weighted_video, weighted_audio], dim=1)
#             merged_mask = torch.cat([video_mask, audio_mask], dim=1)
            
#         elif strategy == "audio_weighted":
#             weighted_video = video_emb * v_weight * 0.5
#             weighted_audio = audio_emb * a_weight
#             merged_emb = torch.cat([weighted_audio, weighted_video], dim=1)
#             merged_mask = torch.cat([audio_mask, video_mask], dim=1)
            
#         elif strategy == "separate_tokens":
#             # Add separator token (zero vector)
#             sep_emb = torch.zeros((1, 1, video_emb.size(-1)), device=target_device, dtype=video_emb.dtype)
#             sep_mask = torch.ones((1, 1), device=target_device, dtype=video_mask.dtype)
            
#             merged_emb = torch.cat([
#                 video_emb * v_weight, 
#                 sep_emb, 
#                 audio_emb * a_weight
#             ], dim=1)
#             merged_mask = torch.cat([video_mask, sep_mask, audio_mask], dim=1)
            
#             if sync_emb is not None:
#                 merged_emb = torch.cat([sync_emb, sep_emb, merged_emb], dim=1)
#                 merged_mask = torch.cat([sync_mask, sep_mask, merged_mask], dim=1)
        
#         return merged_emb, merged_mask
    
#     def encode_dual_prompts(
#         self,
#         text_encoder,
#         video_prompt: str,
#         audio_prompt: str,
#         merge_strategy: str,
#         video_weight: float,
#         audio_weight: float,
#         sync_prompt: str = ""
#     ) -> Tuple[List, str, str]:
        
#         # Ensure text_encoder model is loaded and on correct device
#         if hasattr(text_encoder, 'load_model'):
#             text_encoder.load_model()
        
#         # Get the actual model device from the encoder if possible
#         if hasattr(text_encoder, 'cond_stage_model'):
#             model_device = next(text_encoder.cond_stage_model.parameters()).device
#             self.device = model_device
#         elif hasattr(text_encoder, 'device'):
#             self.device = text_encoder.device
        
#         # Encode video prompt
#         video_emb, video_mask = self.encode_with_encoder(text_encoder, video_prompt)
        
#         # Encode audio prompt
#         audio_emb, audio_mask = self.encode_with_encoder(text_encoder, audio_prompt)
        
#         # Encode optional sync prompt
#         sync_emb, sync_mask = None, None
#         if sync_prompt.strip():
#             sync_emb, sync_mask = self.encode_with_encoder(text_encoder, sync_prompt)
        
#         # Merge embeddings (device handling happens inside)
#         merged_emb, merged_mask = self.merge_embeddings(
#             video_emb, video_mask,
#             audio_emb, audio_mask,
#             merge_strategy,
#             video_weight,
#             audio_weight,
#             sync_emb, sync_mask
#         )
        
#         # Create standard ComfyUI CONDITIONING format
#         conditioning = [[merged_emb, {
#             "pooled_output": merged_emb.mean(dim=1, keepdim=True),
#             "attention_mask": merged_mask,
#             "video_prompt": video_prompt,
#             "audio_prompt": audio_prompt,
#             "merge_strategy": merge_strategy,
#             "video_weight": video_weight,
#             "audio_weight": audio_weight,
#             "is_dual_ltx_conditioning": True,
#             "original_length": merged_emb.size(1)
#         }]]
        
#         # Combine prompts for display
#         combined_prompt = f"[VIDEO] {video_prompt}\n[AUDIO] {audio_prompt}"
#         if sync_prompt:
#             combined_prompt = f"[SYNC] {sync_prompt}\n{combined_prompt}"
        
#         token_info = (
#             f"Video tokens: {video_emb.size(1)}, "
#             f"Audio tokens: {audio_emb.size(1)}, "
#             f"Total merged: {merged_emb.size(1)}"
#         )
        
#         return (conditioning, combined_prompt, token_info)


# class LTXDualPromptEncoderAdvanced(LTXDualPromptEncoder):
#     """
#     Advanced version with separate positive/negative prompts for both streams.
#     """
    
#     @classmethod
#     def INPUT_TYPES(cls) -> Dict[str, Any]:
#         return {
#             "required": {
#                 "text_encoder": ("CLIP", {
#                     "tooltip": "Connect from LTXV Gemma CLIP Loader or similar"
#                 }),
#                 "positive_video": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Positive visual description...",
#                 }),
#                 "positive_audio": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Positive audio description...",
#                 }),
#                 "negative_video": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Negative visual description...",
#                 }),
#                 "negative_audio": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                     "placeholder": "Negative audio description...",
#                 }),
#                 "merge_strategy": (["concatenate", "interleave", "video_weighted", "audio_weighted", "separate_tokens"], {
#                     "default": "concatenate"
#                 }),
#                 "video_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
#                 "audio_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
#             },
#             "optional": {
#                 "sync_prompt": ("STRING", {
#                     "multiline": True,
#                     "default": "",
#                 }),
#             }
#         }
    
#     RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
#     RETURN_NAMES = ("positive_conditioning", "negative_conditioning", "info")
#     FUNCTION = "encode_advanced"
#     CATEGORY = "LTXVideo/dual_prompt"
#     DESCRIPTION = "Advanced dual prompt encoder with positive/negative prompts for both streams using wired text encoder"
    
#     def encode_advanced(
#         self,
#         text_encoder,
#         positive_video: str,
#         positive_audio: str,
#         negative_video: str,
#         negative_audio: str,
#         merge_strategy: str,
#         video_weight: float,
#         audio_weight: float,
#         sync_prompt: str = ""
#     ) -> Tuple[List, List, str]:
        
#         # Encode positive
#         pos_cond, pos_combined, pos_info = self.encode_dual_prompts(
#             text_encoder,
#             positive_video,
#             positive_audio,
#             merge_strategy,
#             video_weight,
#             audio_weight,
#             sync_prompt
#         )
        
#         # Encode negative
#         neg_cond, neg_combined, neg_info = self.encode_dual_prompts(
#             text_encoder,
#             negative_video if negative_video.strip() else "low quality, blurry, distorted",
#             negative_audio if negative_audio.strip() else "noise, static, distortion, silence",
#             merge_strategy,
#             video_weight * 0.8,
#             audio_weight * 0.8,
#             ""  # No sync for negative
#         )
        
#         info = f"POS: {pos_info}\nNEG: {neg_info}"
        
#         return (pos_cond, neg_cond, info)


class LTX2ScheduledEnhanceVideo:
    """
    Scheduled Enhance-A-Video for LTX-2 with proper attention signature handling.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "schedule": ("STRING", {
                    "default": "3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5",
                    "multiline": False,
                }),
                "expected_steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
            },
            "optional": {
                "temporal_only": ("BOOLEAN", {"default": True}),
                "print_schedule": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_schedule"
    CATEGORY = "LTXVideo/advanced"
    
    def apply_schedule(self, model, schedule, expected_steps, temporal_only=True, print_schedule=True):
        # Parse and validate
        values = [float(x.strip()) for x in schedule.split(',')]
        
        if len(values) != expected_steps:
            raise ValueError(
                f"SCHEDULE ERROR: Expected {expected_steps} values for {expected_steps} steps, "
                f"but got {len(values)} values.\n"
                f"Your input: {schedule}\n"
                f"Parsed: {values}"
            )
        
        # Use ComfyUI's built-in attention patching system
        # This is the CORRECT way - don't replace module.forward
        self._setup_attention_patch(model, values, temporal_only, print_schedule)
        
        if print_schedule:
            print(f"[LTX2Enhance] Applied schedule: {values}")
        
        return (model,)
    
    def _setup_attention_patch(self, model, schedule_values, temporal_only, print_schedule):
        """
        Setup patching using ComfyUI's model patcher system.
        Uses set_model_attn1_patch which properly intercepts attention without breaking Linear layers.
        """
        
        step_counter = [0]  # Mutable counter
        
        def enhance_a_video_patch(q, k, v, extra_options):
            """
            Called by ComfyUI's attention system with pre-computed q, k, v tensors.
            Signature: (q, k, v, extra_options) where q,k,v are [batch, heads, seq, dim]
            """
            step = step_counter[0]
            
            # Get tau for this step
            tau = schedule_values[min(step, len(schedule_values) - 1)]
            
            # Compute attention scores
            heads = q.shape[1]
            dim_head = q.shape[3]
            scale = dim_head ** -0.5
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply Enhance-A-Video logic
            if tau != 1.0 and attn_scores.dim() >= 4:
                seq_len = attn_scores.shape[-1]
                
                if temporal_only:
                    # Diagonal mask for intra-frame attention
                    diag_mask = torch.eye(seq_len, device=attn_scores.device, dtype=torch.bool)
                    diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand_as(attn_scores)
                    
                    # Calculate CFI from off-diagonal elements
                    off_diag = attn_scores.masked_select(~diag_mask)
                    cfi = off_diag.mean() if off_diag.numel() > 0 else torch.tensor(1.0, device=attn_scores.device)
                    
                    # Estimate frames
                    frames = max(1, seq_len // 256)
                    
                    # Enhance: clip((tau + F) * CFI, 1)
                    enhanced_cfi = torch.clamp((tau + frames) * cfi, min=1.0)
                    attn_scores = attn_scores * enhanced_cfi
                    
                    if print_schedule and step % 2 == 0:
                        print(f"[LTX2Enhance] Step {step}/{len(schedule_values)} | τ={tau:.2f} | Enhanced={enhanced_cfi:.3f}")
                else:
                    attn_scores = attn_scores * tau
            
            # Softmax and compute output
            attn_weights = F.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            
            # Increment step counter
            step_counter[0] += 1
            
            return output
        
        # Register with ComfyUI - THIS IS THE KEY
        # Use set_model_attn1_patch for self-attention (attn1)
        if hasattr(model, 'set_model_attn1_patch'):
            model.set_model_attn1_patch(enhance_a_video_patch)
            if print_schedule:
                print("[LTX2Enhance] Registered via set_model_attn1_patch")
        else:
            # Fallback for older ComfyUI versions
            if not hasattr(model, 'model_options'):
                model.model_options = {}
            model.model_options['attn1_patch'] = enhance_a_video_patch
            if print_schedule:
                print("[LTX2Enhance] Registered via model_options")

class FreqDecompTemporalGuidance:
    """
    Frequency-Decomposed Temporal Guidance for video diffusion models.
    Applies wavelet decomposition to the model's velocity prediction,
    then applies differentiated guidance to low-vs-high frequency bands
    with optional temporal consistency enforcement on high-frequency content.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "hf_guidance_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Multiplier for guidance on high-frequency sub-bands"
                }),
                "temporal_consistency_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Strength of temporal smoothing on HF sub-bands (0=off)"
                }),
                "start_sigma": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "end_sigma": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "guidance"
    
    def haar_wavelet_2d(self, x):
        """Simple Haar wavelet decomposition on spatial dims of latent."""
        # x shape: [B, C, T, H, W] for video latents
        # Apply along H and W
        ll = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + 
              x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 4
        lh = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] - 
              x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 4
        hl = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] + 
              x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 4
        hh = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - 
              x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 4
        return ll, lh, hl, hh
    
    def haar_inverse_2d(self, ll, lh, hl, hh):
        """Inverse Haar wavelet - reconstruct from sub-bands."""
        B = ll.shape[0]
        C = ll.shape[1]
        T = ll.shape[2] if ll.dim() == 5 else 1
        H, W = ll.shape[-2] * 2, ll.shape[-1] * 2
        
        if ll.dim() == 5:
            out = torch.zeros(B, C, T, H, W, device=ll.device, dtype=ll.dtype)
        else:
            out = torch.zeros(B, C, H, W, device=ll.device, dtype=ll.dtype)
        
        out[..., 0::2, 0::2] = ll + lh + hl + hh
        out[..., 0::2, 1::2] = ll + lh - hl - hh
        out[..., 1::2, 0::2] = ll - lh + hl - hh
        out[..., 1::2, 1::2] = ll - lh - hl + hh
        return out
    
    def apply(self, model, hf_guidance_scale, temporal_consistency_strength,
              start_sigma, end_sigma):
        
        prev_hf = [None]  # Store previous frame HF for temporal consistency
        
        def fdtg_post_cfg_function(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            
            # Check if we're in the active sigma range
            s = sigma[0].item() if sigma.dim() > 0 else sigma.item()
            if s > start_sigma or s < end_sigma:
                return denoised
            
            # Compute schedule position (1 at start_sigma, 0 at end_sigma)
            t = (s - end_sigma) / (start_sigma - end_sigma + 1e-8)
            
            # Decompose into frequency bands
            ll, lh, hl, hh = self.haar_wavelet_2d(denoised)
            
            # Apply boosted guidance to HF bands
            # Scale decreases as we approach end_sigma (geometry should be set)
            current_hf_scale = 1.0 + (hf_guidance_scale - 1.0) * t
            lh = lh * current_hf_scale
            hl = hl * current_hf_scale
            hh = hh * current_hf_scale
            
            # Temporal consistency on HF bands
            if temporal_consistency_strength > 0 and prev_hf[0] is not None:
                tc = temporal_consistency_strength * t
                hf_combined = torch.cat([lh, hl, hh], dim=1)
                prev_combined = prev_hf[0]
                
                if hf_combined.shape == prev_combined.shape:
                    # Blend current HF with previous for temporal smoothness
                    hf_blended = (1 - tc) * hf_combined + tc * prev_combined
                    c = lh.shape[1]
                    lh = hf_blended[:, :c]
                    hl = hf_blended[:, c:2*c]
                    hh = hf_blended[:, 2*c:]
            
            # Store current HF for next step
            prev_hf[0] = torch.cat([lh, hl, hh], dim=1).detach()
            
            # Reconstruct
            result = self.haar_inverse_2d(ll, lh, hl, hh)
            return result
        
        m = model.clone()
        m.set_model_sampler_post_cfg_function(fdtg_post_cfg_function)
        return (m,)


"""
Entropy Rectifying Guidance (ERG) for ComfyUI
==============================================================================
Paper: "Entropy Rectifying Guidance for Diffusion and Flow Models"
  Berrada Ifriqi et al., NeurIPS 2025  |  arXiv:2504.13987

ERG improves quality, diversity, AND prompt consistency simultaneously by
manipulating the energy landscape of attention layers. Unlike CFG (which
trades these against each other), ERG creates a "weak" prediction via
temperature-scaled attention and extrapolates away from it.

Implementation uses ComfyUI's native attn1_patch / attn1_output_patch system
for maximum composability with other guidance methods (APG, CADS, EMAG,
Enhance-A-Video, CFGZeroStar, etc.).

Architecture support: Works with any transformer-based model that ComfyUI
routes through BasicTransformerBlock (LTX-2, Flux, SD3, SDXL, etc.).

Connection: MODEL → [ERG] → MODEL  (inline, before sampler)
==============================================================================
"""

logger = logging.getLogger("ComfyUI-ERG")


class EntropyRectifyingGuidance:
    """
    Full-featured ERG node with all paper parameters exposed.

    Mechanism (per-layer I-ERG):
      1. Capture Q, K, V before self-attention
      2. After attention, compute weak signal: out_weak = α · attn(τ·Q, K, V)
      3. Apply correction: result = out + γ · w · (out - out_weak)

    This pushes each layer's attention output AWAY from the temperature-degraded
    pattern and TOWARD the model's natural strong prediction, amplifying the
    rich attention associations that temperature-scaling destroys.

    The correction composes additively with the standard attention output,
    so it stacks cleanly with other model patches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "erg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": (
                        "Guidance strength (w). Controls how much to push away "
                        "from the weak prediction. 0 = disabled. "
                        "LTX-2: start with 0.3-0.5. Higher values = stronger effect "
                        "but risk artifacts if too aggressive."
                    )
                }),
                "temperature": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 100.0,
                    "step": 0.001,
                    "tooltip": (
                        "Attention temperature (τ). Scales the QK^T logits before "
                        "softmax in the weak prediction.\n"
                        "  τ < 1: Sharpens attention (peaked/one-hot) — paper default 0.01\n"
                        "  τ = 1: No change (ERG disabled)\n"
                        "  τ > 1: Smooths attention (uniform)\n"
                        "Both directions away from 1.0 create weaker predictions.\n"
                        "Paper uses τ=0.01 for denoiser, τ=3.0 for text encoder (C-ERG)."
                    )
                }),
            },
            "optional": {
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": (
                        "Pattern matching weight (α). Scales the weak attention output. "
                        "1.0 = standard. <1 = even weaker signal. "
                        "Paper Table 5: α=1.0 is the default."
                    )
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": (
                        "Step size (γ/η) for energy landscape optimization. "
                        "1.0 = standard. Paper Table 5 found 1.5 gives marginal "
                        "improvement in Coverage and VQAScore."
                    )
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Start applying ERG at this fraction of sampling progress. "
                        "0.0 = from the beginning. Paper uses a kickoff threshold (κ) "
                        "to avoid overly penalizing the weak signal in early steps. "
                        "For LTX-2 distilled (few steps), 0.0 is fine. "
                        "For non-distilled (40+ steps), try 0.2-0.4."
                    )
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Stop applying ERG at this fraction of sampling progress. "
                        "1.0 = apply until the end."
                    )
                }),
                "block_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "tooltip": (
                        "First transformer block index to apply ERG. "
                        "Paper applied to specific layers (Appendix D.4). "
                        "0 = start from first block."
                    )
                }),
                "block_end": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 200,
                    "tooltip": (
                        "Last transformer block index. -1 = all remaining blocks. "
                        "For LTX-2 (48 blocks), targeting middle blocks "
                        "(e.g., 12-36) can be effective."
                    )
                }),
                "block_skip": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": (
                        "Apply ERG every N blocks. 1 = every block, "
                        "2 = every other block. Higher values reduce compute cost."
                    )
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_erg"
    CATEGORY = "model_patches/guidance"
    DESCRIPTION = (
        "Entropy Rectifying Guidance (ERG) — arXiv:2504.13987, NeurIPS 2025\n\n"
        "Improves quality, diversity, AND prompt consistency simultaneously\n"
        "by temperature-scaling self-attention to create a weak prediction,\n"
        "then extrapolating away from it per-layer.\n\n"
        "No extra model forward pass — operates inside attention layers\n"
        "for minimal overhead. Composable with CFG, APG, CADS, EMAG,\n"
        "Enhance-A-Video, CFGZeroStar.\n\n"
        "LTX-2 recommended starting values:\n"
        "  erg_weight: 0.3-0.5  |  temperature: 0.01\n"
        "  start_percent: 0.0   |  gamma: 1.0-1.5\n\n"
        "Connect: MODEL → ERG → (your existing model chain) → Sampler"
    )

    def apply_erg(self, model, erg_weight, temperature,
                  alpha=1.0, gamma=1.0,
                  start_percent=0.0, end_percent=1.0,
                  block_start=0, block_end=-1, block_skip=1):

        if erg_weight == 0.0:
            return (model,)

        m = model.clone()

        # ── Shared mutable state ──────────────────────────────────
        # FIFO queue: input_patch pushes, output_patch pops (1:1 pairing)
        erg_cache = deque()

        # Per-step tracking (reset each time sigma changes)
        state = {
            "block_counter": 0,
            "last_sigma_data": None,   # Track sigma identity
            "step_active": True,
        }

        # ── Freeze parameters into closure ────────────────────────
        _w = erg_weight
        _tau = temperature
        _alpha = alpha
        _gamma = gamma
        _start_pct = start_percent
        _end_pct = end_percent
        _blk_start = block_start
        _blk_end = block_end
        _blk_skip = block_skip

        # ── Input patch: capture Q, K, V ──────────────────────────
        def attn1_input_patch(q, k, v, extra_options):
            """
            Called before each self-attention computation.
            Captures Q, K, V and determines block/step eligibility.
            
            extra_options contains:
              - n_heads: number of attention heads
              - attn_precision: precision for attention computation
              - transformer_options: dict with sigmas, cond_or_uncond, etc.
            """
            transformer_opts = extra_options

            # --- Detect new sampling step via sigma change ---
            sigmas = transformer_opts.get("sigmas", None)
            # Use the actual sigma value as identifier (tensor identity
            # changes per step in most samplers)
            sigma_key = None
            if sigmas is not None and sigmas.numel() > 0:
                sigma_key = (sigmas.data_ptr(), sigmas.shape[0])

            if sigma_key != state["last_sigma_data"]:
                state["block_counter"] = 0
                state["last_sigma_data"] = sigma_key
                state["step_active"] = True

                # Sigma-based scheduling
                if sigmas is not None and sigmas.numel() > 0:
                    current_sigma = sigmas[0].item()
                    sample_sigmas = transformer_opts.get("sample_sigmas", None)

                    if sample_sigmas is not None and sample_sigmas.numel() > 1:
                        s_max = sample_sigmas[0].item()
                        s_min = sample_sigmas[-1].item()
                        if s_max > s_min:
                            pct = 1.0 - (current_sigma - s_min) / (s_max - s_min)
                            pct = max(0.0, min(1.0, pct))
                            if pct < _start_pct or pct > _end_pct:
                                state["step_active"] = False

            # --- Block filtering ---
            blk_idx = state["block_counter"]
            state["block_counter"] += 1

            actual_end = _blk_end if _blk_end >= 0 else float("inf")
            should_apply = (
                state["step_active"]
                and blk_idx >= _blk_start
                and blk_idx <= actual_end
                and (blk_idx - _blk_start) % _blk_skip == 0
            )

            erg_cache.append({
                "q": q,
                "k": k,
                "v": v,
                "apply": should_apply,
                "n_heads": extra_options.get("n_heads", 1),
                "attn_precision": extra_options.get("attn_precision", None),
            })

            return q, k, v

        # ── Output patch: compute ERG correction ──────────────────
        def attn1_output_patch(out, extra_options):
            """
            Called after each self-attention computation.
            
            If this block is eligible:
              1. Compute weak attention: out_weak = α · attn(τ·Q, K, V)
              2. Apply correction: result = out + γ·w·(out - out_weak)
            
            The weak attention uses temperature-scaled Q to produce a
            degraded attention pattern. The correction extrapolates
            the output away from this degraded pattern, amplifying the
            information-rich attention associations.
            """
            if not erg_cache:
                return out

            cached = erg_cache.popleft()

            if not cached["apply"]:
                return out

            q = cached["q"]
            k = cached["k"]
            v = cached["v"]
            heads = cached["n_heads"]
            attn_prec = cached["attn_precision"]

            # ── Compute temperature-scaled attention ──────────
            #
            # Standard: softmax(Q·K^T / √d) @ V
            # Weak:     softmax(τ·Q·K^T / √d) @ V
            #
            # Achieved by scaling Q by τ before the dot product.
            # For τ=0.01: logits collapse → near-one-hot softmax
            #   → attention loses associative richness
            # For τ=100: logits explode → near-uniform softmax
            #   → attention loses all specificity
            q_weak = q * _tau

            try:
                out_weak = comfy_attention.optimized_attention(
                    q_weak, k, v, heads,
                    attn_precision=attn_prec,
                )
            except TypeError:
                # Older ComfyUI versions may not accept attn_precision
                try:
                    out_weak = comfy_attention.optimized_attention(
                        q_weak, k, v, heads,
                    )
                except Exception:
                    return out
            except Exception:
                return out

            # Apply alpha (Hopfield energy pattern matching weight)
            if _alpha != 1.0:
                out_weak = out_weak * _alpha

            # ── ERG correction ────────────────────────────────
            #
            # result = out + γ · w · (out - out_weak)
            #        = (1 + γw)·out - γw·out_weak
            #
            # Interpretation:
            #   (out - out_weak) = what temperature scaling destroyed
            #   Adding this back amplifies the model's natural
            #   attention patterns that carry information.
            #
            # The correction is ADDITIVE to the standard output,
            # so it composes with any other attention modifications.
            correction = _gamma * _w * (out - out_weak)
            return out + correction

        # ── Register with ComfyUI's native patch system ───────────
        m.set_model_attn1_patch(attn1_input_patch)
        m.set_model_attn1_output_patch(attn1_output_patch)

        return (m,)


class EntropyRectifyingGuidanceSimple:
    """
    Simplified ERG with just three knobs.
    Uses paper defaults for alpha, gamma, block selection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "erg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": (
                        "Guidance strength. 0.3-0.8 recommended for LTX-2. "
                        "0 = disabled. Higher = stronger quality/diversity boost."
                    )
                }),
                "temperature": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 100.0,
                    "step": 0.001,
                    "tooltip": (
                        "Attention temperature. Paper default: 0.01 (peaked). "
                        "Values far from 1.0 create weaker signals for guidance. "
                        "Try 0.001-0.1 (peaked) or 10-100 (smoothed)."
                    )
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Skip this fraction of early sampling steps. "
                        "0.0 = apply from start. For LTX-2 distilled, 0.0 is fine."
                    )
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_erg"
    CATEGORY = "model_patches/guidance"
    DESCRIPTION = (
        "ERG Simple — quick entropy rectifying guidance.\n"
        "Uses paper defaults for advanced parameters.\n"
        "Connect: MODEL → ERG Simple → (existing chain) → Sampler"
    )

    def apply_erg(self, model, erg_weight, temperature, start_percent=0.0):
        full_node = EntropyRectifyingGuidance()
        return full_node.apply_erg(
            model=model,
            erg_weight=erg_weight,
            temperature=temperature,
            alpha=1.0,
            gamma=1.0,
            start_percent=start_percent,
            end_percent=1.0,
            block_start=0,
            block_end=-1,
            block_skip=1,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ropebal_logger = logging.getLogger("ComfyUI-RoPEBalance")

def _compute_equalized_max_pos(h_latent, w_latent, ref_max_pos_w, max_pos_t,
                                vertical_boost=1.0):
    """Return [t, h, w] max_pos with H equalised to match W's angular span.

    Derivation
    ----------
    frac_h = ((h_latent-1) * 32) / max_pos_h
    frac_w = ((w_latent-1) * 32) / max_pos_w

    Equalize frac_h == frac_w:
        max_pos_h = max_pos_w * (h_latent - 1) / (w_latent - 1)

    Then apply vertical_boost (> 1 = tighter H range = MORE discrimination):
        max_pos_h /= vertical_boost          ← smaller denom → larger frac → wider span
    """
    if w_latent <= 1 or h_latent <= 1:
        return [max_pos_t, ref_max_pos_w, ref_max_pos_w]

    equalized_h = ref_max_pos_w * (h_latent - 1) / (w_latent - 1)

    # vertical_boost > 1 shrinks max_pos_h → widens fractional range → more
    # angular discrimination on the vertical axis.
    if vertical_boost > 0:
        adjusted_h = equalized_h / vertical_boost
    else:
        adjusted_h = equalized_h

    adjusted_h = max(1, int(round(adjusted_h)))
    return [max_pos_t, adjusted_h, ref_max_pos_w]


def _angular_span(latent_dim, max_pos, scale=32):
    """Fractional angular span for one axis (diagnostic helper)."""
    if max_pos <= 0 or latent_dim <= 1:
        return 0.0
    frac = ((latent_dim - 1) * scale) / max_pos
    return 2.0 * frac   # mapped span after (frac * 2 - 1)


# ---------------------------------------------------------------------------
# Node: LTX RoPE Axis Balance
# ---------------------------------------------------------------------------

class LTXRoPEAxisBalance:
    """
    Equalise or manually control the 3D RoPE positional embedding ranges
    for LTX-Video / LTX-2 models.

    Fixes directional asymmetry where horizontal camera motion is stable
    but vertical pans produce noise/artifacts, by adjusting the angular
    discrimination of the height axis to match the width axis.

    Zero computational cost — same operations, different normalisation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["auto_equalize", "manual"], {
                    "default": "auto_equalize",
                    "tooltip": (
                        "auto_equalize: compute max_pos_h from the actual "
                        "latent aspect ratio each step so H and W get equal "
                        "angular discrimination.  manual: use the explicit "
                        "max_pos values below."
                    ),
                }),
                "vertical_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": (
                        "Multiplier on vertical-axis discrimination.  "
                        "1.0 = equalised with horizontal.  "
                        ">1.0 = vertical gets MORE discrimination (try 1.2–1.5 "
                        "if squiggles persist).  "
                        "<1.0 = vertical gets less (closer to default bias)."
                    ),
                }),
            },
            "optional": {
                "max_pos_t": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Temporal axis max position (default 20). Used in both modes.",
                }),
                "max_pos_h": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "tooltip": "Manual mode only: height axis max position.",
                }),
                "max_pos_w": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "tooltip": (
                        "Width axis max position (default 2048).  "
                        "In auto mode this is the reference value for "
                        "computing the equalised height."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/guidance"
    TITLE = "LTX RoPE Axis Balance"

    def patch(self, model, mode, vertical_boost,
              max_pos_t=20, max_pos_h=2048, max_pos_w=2048):

        m = model.clone()

        # ----- Verify the target function exists ---------------------------
        try:
            import comfy.ldm.lightricks.model as ltx_module
            if not hasattr(ltx_module, "precompute_freqs_cis"):
                ropebal_logger.warning(
                    "[RoPE Axis Balance] precompute_freqs_cis not found in "
                    "comfy.ldm.lightricks.model — node will have no effect. "
                    "This may mean your ComfyUI version changed the function "
                    "location."
                )
                return (m,)
        except ImportError:
            ropebal_logger.warning(
                "[RoPE Axis Balance] comfy.ldm.lightricks.model not found — "
                "node will have no effect."
            )
            return (m,)

        # ----- Capture existing model_function_wrapper for chaining --------
        existing_wrapper = m.model_options.get("model_function_wrapper", None)

        # ----- Close over node settings ------------------------------------
        _mode = mode
        _vboost = vertical_boost
        _mpt = max_pos_t
        _mph = max_pos_h
        _mpw = max_pos_w

        # Track last-printed values so we log once per unique config,
        # not every single step.
        _last_logged = [None]

        def rope_balance_wrapper(apply_model, args):
            import comfy.ldm.lightricks.model as ltx_mod
            original_func = ltx_mod.precompute_freqs_cis

            # --- Compute custom max_pos --------------------------------
            if _mode == "auto_equalize":
                x = args.get("input")
                if x is not None and x.dim() == 5:
                    h_lat = x.shape[3]
                    w_lat = x.shape[4]
                    custom_max_pos = _compute_equalized_max_pos(
                        h_lat, w_lat, _mpw, _mpt, _vboost
                    )
                else:
                    custom_max_pos = [_mpt, _mph, _mpw]
            else:
                # Manual mode — apply vertical_boost as multiplier on max_pos_h
                # (boost > 1 shrinks max_pos → more discrimination)
                adjusted_h = max(1, int(round(_mph / _vboost)))
                custom_max_pos = [_mpt, adjusted_h, _mpw]

            # --- Log once per unique configuration ----------------------
            log_key = tuple(custom_max_pos)
            if log_key != tuple(_last_logged[0] or []):
                _last_logged[0] = list(custom_max_pos)
                if args.get("input") is not None and args["input"].dim() == 5:
                    x = args["input"]
                    h_lat, w_lat = x.shape[3], x.shape[4]
                    span_h = _angular_span(h_lat, custom_max_pos[1])
                    span_w = _angular_span(w_lat, custom_max_pos[2])
                    default_span_h = _angular_span(h_lat, 2048)
                    default_span_w = _angular_span(w_lat, 2048)
                    ropebal_logger.info(
                        f"[RoPE Axis Balance] mode={_mode} boost={_vboost:.2f}  "
                        f"latent={w_lat}×{h_lat}  "
                        f"max_pos=[{custom_max_pos[0]}, {custom_max_pos[1]}, {custom_max_pos[2]}]  "
                        f"span H={span_h:.3f} W={span_w:.3f}  "
                        f"(default: H={default_span_h:.3f} W={default_span_w:.3f})"
                    )
                else:
                    ropebal_logger.info(
                        f"[RoPE Axis Balance] max_pos="
                        f"[{custom_max_pos[0]}, {custom_max_pos[1]}, {custom_max_pos[2]}]"
                    )
            # Also print to stdout for ComfyUI console visibility
            if log_key != tuple((_last_logged[0] or [])[:3]):
                pass  # Already logged above, stdout via logger

            # --- Monkey-patch for this forward pass ---------------------
            def patched_precompute_freqs_cis(
                indices_grid, dim, out_dtype, theta=10000.0,
                max_pos=None  # swallow whatever the caller sends
            ):
                return original_func(
                    indices_grid, dim, out_dtype,
                    theta=theta, max_pos=custom_max_pos
                )

            ltx_mod.precompute_freqs_cis = patched_precompute_freqs_cis
            try:
                if existing_wrapper is not None:
                    return existing_wrapper(apply_model, args)
                else:
                    return apply_model(
                        args["input"], args["timestep"], **args["c"]
                    )
            finally:
                # Always restore regardless of exceptions
                ltx_mod.precompute_freqs_cis = original_func

        m.set_model_unet_function_wrapper(rope_balance_wrapper)
        return (m,)


# ---------------------------------------------------------------------------
# Node: LTX RoPE Axis Balance — Simple
# ---------------------------------------------------------------------------

class LTXRoPEAxisBalanceSimple:
    """
    Simplified one-knob version.

    Auto-equalises the vertical RoPE axis to match horizontal, with
    a single vertical_boost slider to go beyond parity if needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vertical_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": (
                        "1.0 = equalise H and W angular ranges (fixes "
                        "vertical pan artifacts).  >1.0 = boost vertical "
                        "further.  <1.0 = partial correction."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/guidance"
    TITLE = "LTX RoPE Axis Balance (Simple)"

    def patch(self, model, vertical_boost):
        # Delegate to the full node with auto mode
        full_node = LTXRoPEAxisBalance()
        return full_node.patch(
            model,
            mode="auto_equalize",
            vertical_boost=vertical_boost,
            max_pos_t=20,
            max_pos_h=2048,
            max_pos_w=2048,
        )


# ---------------------------------------------------------------------------
# Node: LTX RoPE Diagnostic
# ---------------------------------------------------------------------------

class LTXRoPEDiagnostic:
    """
    Diagnostic node — does NOT modify the model.

    Computes and displays the effective RoPE angular spans for a given
    resolution so you can see the asymmetry before/after patching.
    Connect to any MODEL output; prints info to console.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 32,
                    "tooltip": "Output pixel width"
                }),
                "height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 32,
                    "tooltip": "Output pixel height"
                }),
                "max_pos_h": ("INT", {
                    "default": 2048, "min": 1, "max": 16384,
                    "tooltip": "Current or proposed max_pos for height axis"
                }),
                "max_pos_w": ("INT", {
                    "default": 2048, "min": 1, "max": 16384,
                    "tooltip": "Current or proposed max_pos for width axis"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("model", "report",)
    FUNCTION = "diagnose"
    CATEGORY = "model_patches/guidance"
    TITLE = "LTX RoPE Diagnostic"
    OUTPUT_NODE = True

    def diagnose(self, model, width, height, max_pos_h, max_pos_w):
        # LTX VAE: 32× spatial downscale
        w_lat = width // 32
        h_lat = height // 32

        # With default max_pos
        default_span_h = _angular_span(h_lat, 2048)
        default_span_w = _angular_span(w_lat, 2048)
        default_ratio = default_span_w / default_span_h if default_span_h > 0 else float('inf')

        # With proposed max_pos
        proposed_span_h = _angular_span(h_lat, max_pos_h)
        proposed_span_w = _angular_span(w_lat, max_pos_w)
        proposed_ratio = proposed_span_w / proposed_span_h if proposed_span_h > 0 else float('inf')

        # Auto-equalized
        eq_max_pos = _compute_equalized_max_pos(h_lat, w_lat, 2048, 20, 1.0)
        eq_span_h = _angular_span(h_lat, eq_max_pos[1])
        eq_span_w = _angular_span(w_lat, eq_max_pos[2])

        lines = [
            f"═══ RoPE Axis Diagnostic for {width}×{height} ═══",
            f"Latent dimensions: {w_lat}×{h_lat} (W×H)",
            f"",
            f"── DEFAULT max_pos=[20, 2048, 2048] ──",
            f"  H angular span: {default_span_h:.4f}",
            f"  W angular span: {default_span_w:.4f}",
            f"  W/H ratio:      {default_ratio:.3f}×  ← W has {default_ratio:.1f}× more discrimination",
            f"",
            f"── PROPOSED max_pos=[20, {max_pos_h}, {max_pos_w}] ──",
            f"  H angular span: {proposed_span_h:.4f}",
            f"  W angular span: {proposed_span_w:.4f}",
            f"  W/H ratio:      {proposed_ratio:.3f}×",
            f"",
            f"── AUTO EQUALIZED max_pos=[20, {eq_max_pos[1]}, {eq_max_pos[2]}] ──",
            f"  H angular span: {eq_span_h:.4f}",
            f"  W angular span: {eq_span_w:.4f}",
            f"  Equalized ratio: {eq_span_w / eq_span_h if eq_span_h > 0 else 0:.3f}×",
        ]
        report = "\n".join(lines)

        print(report)
        ropebal_logger.info(report)

        return (model, report,)
    

"""
ComfyUI node definitions for Focus Magic-style deconvolution.

Provides three main nodes:
1. FocusDeconvDefocus - Out-of-focus blur correction (disk PSF)
2. FocusDeconvMotion - Motion blur correction (linear PSF)
3. FocusDeconvBlind - Automatic blind deconvolution (PSF auto-estimated)

Plus utility nodes:
4. FocusDeconvPSFPreview - Visualize the PSF being used
5. FocusDeconvAdvanced - Full control over all parameters
"""


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI image tensor (B,H,W,C) to numpy (H,W,C) float64 in [0,1]."""
    img = tensor[0].cpu().numpy().astype(np.float64)
    return np.clip(img, 0, 1)


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy (H,W,C) float64 to ComfyUI image tensor (B,H,W,C)."""
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)


# ─── Node 1: Out-of-Focus Deconvolution ────────────────────────────────────────

class FocusDeconvDefocus:
    """
    Correct out-of-focus blur using deconvolution with a circular disk PSF.
    
    This is the equivalent of Focus Magic's "Fix Out-of-Focus Blur" mode.
    The blur_width parameter corresponds to the diameter of the circle of
    confusion - measure this by finding point light sources in your image
    that have become circles, and measuring their diameter in pixels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_width": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.5,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Diameter of the defocus blur circle in pixels. "
                               "Find point sources in your image and measure how "
                               "large their circles are. (Focus Magic's 'Blur Width')"
                }),
                "method": (["wiener", "richardson_lucy", "hybrid"], {
                    "default": "hybrid",
                    "tooltip": "Deconvolution algorithm. Wiener is fast, RL is higher "
                               "quality but slower, Hybrid uses both."
                }),
                "noise_reduction": (["none", "low", "medium", "high", "very_high"], {
                    "default": "medium",
                    "tooltip": "Amount of noise reduction applied. Higher values reduce "
                               "artifacts but may lose fine detail."
                }),
            },
            "optional": {
                "psf_type": (["disk", "gaussian"], {
                    "default": "disk",
                    "tooltip": "PSF shape. Disk models lens defocus, Gaussian models "
                               "mild atmospheric/diffusion blur."
                }),
                "amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Deconvolution strength. 1.0 = full correction, "
                               "<1.0 blends with original, >1.0 for extra sharpening."
                }),
                "iterations": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Number of iterations for RL/Hybrid methods. "
                               "More = higher quality but slower. (Accuracy/Speed tradeoff)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("deconvolved",)
    FUNCTION = "execute"
    CATEGORY = "image/deconvolution"
    DESCRIPTION = ("Correct out-of-focus blur using Focus Magic-style deconvolution. "
                   "Set blur_width to the diameter of defocused point sources in your image.")

    def execute(self, image, blur_width, method, noise_reduction,
                psf_type="disk", amount=1.0, iterations=30):

        img_np = tensor_to_numpy(image)

        # Generate PSF
        if psf_type == "gaussian":
            # Convert diameter to sigma (diameter ≈ 4*sigma for Gaussian)
            sigma = blur_width / 4.0
            psf = make_gaussian_psf(sigma)
        else:
            psf = make_disk_psf(blur_width)

        # Map noise reduction setting to parameters
        noise_params = {
            "none":      (0.001, 0.0, 0.0),
            "low":       (0.005, 0.0, 0.3),
            "medium":    (0.01,  0.2, 0.5),
            "high":      (0.02,  0.4, 0.7),
            "very_high": (0.05,  0.6, 0.9),
        }
        noise_power, denoise_before, denoise_after = noise_params.get(
            noise_reduction, (0.01, 0.2, 0.5)
        )

        # Scale noise power with blur width (larger blur = more noise amplification)
        noise_power *= (1 + blur_width / 10.0)

        result = deconvolve_image(
            img_np, psf,
            method=method,
            noise_power=noise_power,
            regularization=noise_power * 0.1,
            iterations=iterations,
            tv_reg=noise_power * 0.1,
            denoise_before=denoise_before,
            denoise_after=denoise_after,
            amount=amount,
        )

        return (numpy_to_tensor(result),)


# ─── Node 2: Motion Blur Deconvolution ─────────────────────────────────────────

class FocusDeconvMotion:
    """
    Correct motion blur using deconvolution with a linear PSF.
    
    This is the equivalent of Focus Magic's "Fix Motion Blur" mode.
    You need to specify the direction of motion (angle) and the distance.
    
    To determine these: look for point sources in the image that have
    become streaks. The streak direction gives you the angle, and the
    streak length gives you the distance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_direction": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": "Direction of motion in degrees. 0°=right, 90°=up, "
                               "180°=left, 270°=down. (Focus Magic's 'Blur Direction')"
                }),
                "blur_distance": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Length of the motion blur streak in pixels. "
                               "(Focus Magic's 'Blur Distance')"
                }),
                "method": (["wiener", "richardson_lucy", "hybrid"], {
                    "default": "hybrid",
                }),
                "noise_reduction": (["none", "low", "medium", "high", "very_high"], {
                    "default": "medium",
                }),
            },
            "optional": {
                "tapered_motion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable for 'slow-fast-slow' motion blur (camera jerk). "
                               "Creates brighter endpoints in the PSF."
                }),
                "taper_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How much the motion endpoints are weighted "
                               "(only used with tapered_motion)."
                }),
                "amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                }),
                "iterations": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("deconvolved",)
    FUNCTION = "execute"
    CATEGORY = "image/deconvolution"
    DESCRIPTION = ("Correct motion blur (camera shake) using Focus Magic-style "
                   "deconvolution. Specify the direction and distance of the blur.")

    def execute(self, image, blur_direction, blur_distance, method, noise_reduction,
                tapered_motion=False, taper_strength=0.3, amount=1.0, iterations=30):

        img_np = tensor_to_numpy(image)

        # Generate motion PSF
        if tapered_motion:
            psf = make_motion_psf_tapered(blur_direction, blur_distance, taper_strength)
        else:
            psf = make_motion_psf(blur_direction, blur_distance)

        noise_params = {
            "none":      (0.001, 0.0, 0.0),
            "low":       (0.005, 0.0, 0.3),
            "medium":    (0.01,  0.2, 0.5),
            "high":      (0.02,  0.4, 0.7),
            "very_high": (0.05,  0.6, 0.9),
        }
        noise_power, denoise_before, denoise_after = noise_params.get(
            noise_reduction, (0.01, 0.2, 0.5)
        )

        noise_power *= (1 + blur_distance / 15.0)

        result = deconvolve_image(
            img_np, psf,
            method=method,
            noise_power=noise_power,
            regularization=noise_power * 0.1,
            iterations=iterations,
            tv_reg=noise_power * 0.1,
            denoise_before=denoise_before,
            denoise_after=denoise_after,
            amount=amount,
        )

        return (numpy_to_tensor(result),)


# ─── Node 3: Blind Deconvolution ───────────────────────────────────────────────

class FocusDeconvBlind:
    """
    Automatic blind deconvolution - estimates PSF parameters from the image.
    
    Analyzes the image's frequency spectrum to automatically determine:
    - Whether the blur is defocus or motion
    - The blur width/distance and direction
    
    Then applies the appropriate deconvolution. This is similar to
    Focus Magic's auto-detection feature, though Focus Magic's site notes
    their auto feature "is not truly blind."
    
    For best results, manually specifying parameters with the Defocus or
    Motion nodes will usually outperform this automatic mode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (["auto", "defocus", "motion"], {
                    "default": "auto",
                    "tooltip": "Type of blur to correct. 'auto' will attempt to "
                               "detect whether it's defocus or motion blur."
                }),
                "method": (["wiener", "richardson_lucy", "hybrid"], {
                    "default": "hybrid",
                }),
                "noise_reduction": (["none", "low", "medium", "high", "very_high"], {
                    "default": "medium",
                }),
            },
            "optional": {
                "amount": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Defaulting to 0.75 for blind mode since auto-estimated "
                               "parameters may not be perfect."
                }),
                "iterations": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("deconvolved", "estimated_params",)
    FUNCTION = "execute"
    CATEGORY = "image/deconvolution"
    DESCRIPTION = ("Automatic blind deconvolution. Estimates blur parameters from "
                   "the image and applies correction. For best results, use the "
                   "Defocus or Motion nodes with manually measured parameters.")

    def execute(self, image, blur_type, method, noise_reduction,
                amount=0.75, iterations=30):

        img_np = tensor_to_numpy(image)

        # Convert to grayscale for analysis
        gray = np.mean(img_np, axis=2)

        param_info = ""

        if blur_type == "auto" or blur_type == "defocus":
            blur_width = estimate_blur_width(gray)
            psf = make_disk_psf(blur_width)
            param_info = f"Detected: Defocus blur, width={blur_width:.1f}px"

            if blur_type == "auto":
                # Also check for motion blur
                motion_angle, motion_dist = estimate_motion_params(gray)
                # Simple heuristic: if motion distance is much larger than
                # estimated defocus width, it's likely motion blur
                if motion_dist > blur_width * 1.5 and motion_dist > 3:
                    psf = make_motion_psf(motion_angle, motion_dist)
                    param_info = (f"Detected: Motion blur, angle={motion_angle:.0f}°, "
                                  f"distance={motion_dist:.1f}px")
        else:  # motion
            motion_angle, motion_dist = estimate_motion_params(gray)
            psf = make_motion_psf(motion_angle, motion_dist)
            param_info = (f"Detected: Motion blur, angle={motion_angle:.0f}°, "
                          f"distance={motion_dist:.1f}px")

        noise_params = {
            "none":      (0.001, 0.0, 0.0),
            "low":       (0.005, 0.0, 0.3),
            "medium":    (0.01,  0.2, 0.5),
            "high":      (0.02,  0.4, 0.7),
            "very_high": (0.05,  0.6, 0.9),
        }
        noise_power, denoise_before, denoise_after = noise_params.get(
            noise_reduction, (0.01, 0.2, 0.5)
        )

        result = deconvolve_image(
            img_np, psf,
            method=method,
            noise_power=noise_power,
            regularization=noise_power * 0.1,
            iterations=iterations,
            tv_reg=noise_power * 0.1,
            denoise_before=denoise_before,
            denoise_after=denoise_after,
            amount=amount,
        )

        print(f"[FocusDeconv] {param_info}")

        return (numpy_to_tensor(result), param_info)


# ─── Node 4: PSF Preview ───────────────────────────────────────────────────────

class FocusDeconvPSFPreview:
    """
    Generate and visualize the Point Spread Function (PSF) being used.
    
    Useful for understanding what blur pattern is being modeled,
    and for verifying your parameters are correct before running
    the full deconvolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psf_type": (["disk", "gaussian", "motion", "motion_tapered"], {
                    "default": "disk",
                }),
                "size_param": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "For disk/gaussian: blur width. For motion: blur distance."
                }),
            },
            "optional": {
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": "Motion blur angle (only for motion PSF types)."
                }),
                "taper_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "preview_size": ("INT", {
                    "default": 128,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("psf_preview",)
    FUNCTION = "execute"
    CATEGORY = "image/deconvolution"
    DESCRIPTION = "Visualize the Point Spread Function (PSF) for debugging."

    def execute(self, psf_type, size_param, angle=0.0, taper_strength=0.3,
                preview_size=128):

        if psf_type == "disk":
            psf = make_disk_psf(size_param)
        elif psf_type == "gaussian":
            psf = make_gaussian_psf(size_param / 4.0)
        elif psf_type == "motion":
            psf = make_motion_psf(angle, size_param)
        elif psf_type == "motion_tapered":
            psf = make_motion_psf_tapered(angle, size_param, taper_strength)
        else:
            psf = make_disk_psf(size_param)

        # Scale PSF to preview size and normalize for visibility
        from scipy.ndimage import zoom
        h, w = psf.shape
        scale = preview_size / max(h, w)
        psf_resized = zoom(psf, scale, order=1)

        # Pad to square
        ph, pw = psf_resized.shape
        canvas = np.zeros((preview_size, preview_size), dtype=np.float64)
        y_off = (preview_size - ph) // 2
        x_off = (preview_size - pw) // 2
        canvas[y_off:y_off + ph, x_off:x_off + pw] = psf_resized

        # Normalize for display
        max_val = canvas.max()
        if max_val > 0:
            canvas = canvas / max_val

        # Convert to RGB
        rgb = np.stack([canvas, canvas, canvas], axis=2)

        return (numpy_to_tensor(rgb),)


# ─── Node 5: Advanced Full-Control ──────────────────────────────────────────────

class FocusDeconvAdvanced:
    """
    Advanced deconvolution node with full control over all parameters.
    
    For power users who want to fine-tune every aspect of the deconvolution
    process. Exposes all internal parameters including noise power,
    regularization, TV weight, pre/post noise reduction, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "psf_type": (["disk", "gaussian", "motion", "motion_tapered"], {
                    "default": "disk",
                }),
                "blur_size": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.5,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Blur width (defocus) or blur distance (motion)."
                }),
                "method": (["wiener", "richardson_lucy", "hybrid"], {
                    "default": "hybrid",
                }),
            },
            "optional": {
                "blur_angle": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Motion blur angle in degrees."
                }),
                "taper_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Motion endpoint weighting for tapered motion PSF."
                }),
                "noise_power": ("FLOAT", {
                    "default": 0.01, "min": 0.0001, "max": 0.5, "step": 0.001,
                    "tooltip": "Noise-to-signal ratio for Wiener filter. Higher = "
                               "more regularization, less noise but softer result."
                }),
                "regularization": ("FLOAT", {
                    "default": 0.001, "min": 0.0, "max": 0.1, "step": 0.0005,
                    "tooltip": "Tikhonov regularization strength. Penalizes high "
                               "frequency content to reduce ringing."
                }),
                "iterations": ("INT", {
                    "default": 30, "min": 1, "max": 200, "step": 1,
                    "tooltip": "RL/Hybrid iteration count. More = sharper but slower."
                }),
                "tv_regularization": ("FLOAT", {
                    "default": 0.001, "min": 0.0, "max": 0.05, "step": 0.0005,
                    "tooltip": "Total Variation regularization. Reduces noise while "
                               "preserving edges during RL iterations."
                }),
                "denoise_before": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Pre-deconvolution noise reduction strength."
                }),
                "denoise_after": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Post-deconvolution noise reduction strength."
                }),
                "amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and deconvolved (1). "
                               ">1.0 for aggressive sharpening."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("deconvolved",)
    FUNCTION = "execute"
    CATEGORY = "image/deconvolution"
    DESCRIPTION = ("Advanced deconvolution with full parameter control. "
                   "For power users who want to fine-tune every aspect.")

    def execute(self, image, psf_type, blur_size, method,
                blur_angle=0.0, taper_strength=0.3,
                noise_power=0.01, regularization=0.001,
                iterations=30, tv_regularization=0.001,
                denoise_before=0.0, denoise_after=0.5, amount=1.0):

        img_np = tensor_to_numpy(image)

        # Generate PSF
        if psf_type == "disk":
            psf = make_disk_psf(blur_size)
        elif psf_type == "gaussian":
            psf = make_gaussian_psf(blur_size / 4.0)
        elif psf_type == "motion":
            psf = make_motion_psf(blur_angle, blur_size)
        elif psf_type == "motion_tapered":
            psf = make_motion_psf_tapered(blur_angle, blur_size, taper_strength)
        else:
            psf = make_disk_psf(blur_size)

        result = deconvolve_image(
            img_np, psf,
            method=method,
            noise_power=noise_power,
            regularization=regularization,
            iterations=iterations,
            tv_reg=tv_regularization,
            denoise_before=denoise_before,
            denoise_after=denoise_after,
            amount=amount,
        )

        return (numpy_to_tensor(result),)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EMAGGuider": EMAGGuider,
    # "LTXDualPromptEncoder": LTXDualPromptEncoder,
    # "LTXDualPromptEncoderAdvanced": LTXDualPromptEncoderAdvanced,
    "LTX2ScheduledEnhanceVideo": LTX2ScheduledEnhanceVideo,
    "FreqDecompTemporalGuidance": FreqDecompTemporalGuidance,
    "EntropyRectifyingGuidance": EntropyRectifyingGuidance,
    "EntropyRectifyingGuidanceSimple": EntropyRectifyingGuidanceSimple,
    "LTXRoPEAxisBalance":       LTXRoPEAxisBalance,
    "LTXRoPEAxisBalanceSimple": LTXRoPEAxisBalanceSimple,
    "LTXRoPEDiagnostic":        LTXRoPEDiagnostic,
    "FocusDeconvDefocus": FocusDeconvDefocus,
    "FocusDeconvMotion": FocusDeconvMotion,
    "FocusDeconvBlind": FocusDeconvBlind,
    "FocusDeconvPSFPreview": FocusDeconvPSFPreview,
    "FocusDeconvAdvanced": FocusDeconvAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EMAGGuider": "EMAG Guider",
    # "LTXDualPromptEncoder": "LTX Dual Prompt Encoder",
    # "LTXDualPromptEncoderAdvanced": "LTX Dual Prompt Encoder (Advanced)",
    "LTX2ScheduledEnhanceVideo": "LTX-2 Scheduled Enhance-A-Video",
    "FreqDecompTemporalGuidance": "Frequency-Decomposed Temporal Guidance",
    "EntropyRectifyingGuidance": "ERG (Entropy Rectifying Guidance)",
    "EntropyRectifyingGuidanceSimple": "ERG Simple",
    "LTXRoPEAxisBalance":       "LTX RoPE Axis Balance",
    "LTXRoPEAxisBalanceSimple": "LTX RoPE Axis Balance (Simple)",
    "LTXRoPEDiagnostic":        "LTX RoPE Diagnostic",
    "FocusDeconvDefocus": "🔍 Focus Deconv - Defocus",
    "FocusDeconvMotion": "🔍 Focus Deconv - Motion Blur",
    "FocusDeconvBlind": "🔍 Focus Deconv - Blind (Auto)",
    "FocusDeconvPSFPreview": "🔍 Focus Deconv - PSF Preview",
    "FocusDeconvAdvanced": "🔍 Focus Deconv - Advanced",
}