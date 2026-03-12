# node.py

import torch
import torch.nn.functional as F
import comfy.samplers


from .emag import EMAGGuiderImpl
from .emasync import EMASyncGuiderImpl
from .sa_rf_solver import sa_rf_sample



# :: ==========================================================================
# :: ==========================================================================
# ::
# :: EMAG GUIDER
# :: 
# :: ==========================================================================
# :: ==========================================================================
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
    CATEGORY = "ZLD/sampling/guiders"
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



# :: ==========================================================================
# :: ==========================================================================
# ::
# :: EMASync GUIDER
# :: 
# :: ==========================================================================
# :: ==========================================================================

class EMASyncGuider:
    """
    EMAG + SyncCFG Hybrid Guider Node
    Combines Exponential Moving Average Guidance with Synchronization-Enhanced CFG
    for improved audio-video generation alignment.
    
    Modes:
    - EMAG_ONLY: Standard EMAG guidance (original behavior)
    - SYNCCFG_ONLY: Pure SyncCFG without EMA perturbation
    - HYBRID: EMAG perturbation + SyncCFG guidance structure
    
    Based on: 
    - "EMAG: Exponential Moving Average Guidance for Diffusion Models"
    - "Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["EMAG_ONLY", "SYNCCFG_ONLY", "HYBRID"], {
                    "default": "HYBRID",
                    "tooltip": "Guidance mode: EMAG_ONLY (original), SYNCCFG_ONLY (alignment-focused), HYBRID (both)"
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.1, 
                    "round": 0.01,
                    "tooltip": "Base CFG scale (w in standard CFG)"
                }),
                # EMAG parameters (used in EMAG_ONLY and HYBRID modes)
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
                # SyncCFG parameters (used in SYNCCFG_ONLY and HYBRID modes)
                "sync_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Synchronization guidance scale (s_sync in Harmony paper). Amplifies audio-video alignment"
                }),
                "video_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Video quality guidance scale (s_v). Lower than cfg to prevent over-amplification"
                }),
                "audio_scale": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Audio quality guidance scale (s_a). Lower than cfg to prevent over-amplification"
                }),
                # Scheduling
                "start_percent": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Start applying guidance at this percent of total steps (1.0 = beginning)"
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Stop applying guidance at this percent of total steps (0.2 = last 20%)"
                }),
            },
            "optional": {
                "adaptive_layers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use adaptive layer selection for EMAG (recommended, see paper Section 4.3)"
                }),
                "perturb_img_to_text": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "EMAG (Full)", 
                    "label_off": "EMAG-I (Img Only)",
                    "tooltip": "EMAG perturbs both Image->Image and Image->Text. EMAG-I only perturbs Image->Image"
                }),
                "separate_audio_video_cond": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable if you have separate audio and video conditionings for SyncCFG"
                }),
                "audio_positive": ("CONDITIONING", {
                    "default": None,
                    "tooltip": "Optional separate audio conditioning for SyncCFG mode"
                }),
                "video_positive": ("CONDITIONING", {
                    "default": None,
                    "tooltip": "Optional separate video conditioning for SyncCFG mode"
                }),
            }
        }
    
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "ZLD/sampling/guiders"
    DESCRIPTION = "EMAG + SyncCFG Hybrid: Combines EMA attention perturbation with decoupled audio-video synchronization guidance"
    
    def get_guider(self, model, positive, negative, mode, cfg, emag_scale, ema_decay,
                   sync_scale, video_scale, audio_scale,
                   start_percent, end_percent, adaptive_layers=True, 
                   perturb_img_to_text=True, separate_audio_video_cond=False,
                   audio_positive=None, video_positive=None):
        """Create and return the EMAG+SyncCFG hybrid guider instance"""
        
        # Validate conditioning inputs for SyncCFG modes
        if mode in ["SYNCCFG_ONLY", "HYBRID"] and separate_audio_video_cond:
            if audio_positive is None or video_positive is None:
                print("[EMASync] Warning: separate_audio_video_cond enabled but missing audio_positive or video_positive. Falling back to shared conditioning.")
                separate_audio_video_cond = False
        
        guider = EMASyncGuiderImpl(
            model_patcher=model,
            mode=mode,
            cfg=cfg,
            emag_scale=emag_scale,
            ema_decay=ema_decay,
            sync_scale=sync_scale,
            video_scale=video_scale,
            audio_scale=audio_scale,
            start_percent=start_percent,
            end_percent=end_percent,
            adaptive_layers=adaptive_layers,
            perturb_img_to_text=perturb_img_to_text,
            separate_audio_video_cond=separate_audio_video_cond
        )
        
        # Set conditionings
        guider.set_conds(positive, negative)
        
        # Store separate conditionings if provided
        if separate_audio_video_cond and audio_positive is not None and video_positive is not None:
            guider.audio_positive = audio_positive
            guider.video_positive = video_positive
        
        guider.set_cfg(cfg)
        return (guider,)



# :: ==========================================================================
# :: ==========================================================================
# ::
# :: LTX2 Scheduled Enhance-A-Video
# :: 
# :: ==========================================================================
# :: ==========================================================================

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
    CATEGORY = "ZLD/sampling/guiders"
    
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



# :: ==========================================================================
# :: ==========================================================================
# ::
# :: Frequency Decomposed Temporal Guidance
# :: 
# :: ==========================================================================
# :: ==========================================================================

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
    FUNCTION = "apply_guider"
    CATEGORY = "ZLD/sampling/guiders"
    
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
    


# :: ==========================================================================
# :: ==========================================================================
# ::
# :: SA RF Solver Sampler
# :: 
# :: ==========================================================================
# :: ==========================================================================

class SARFSolverSamplerNode:
    """
    SA-RF-Solver v2: Proper SDE sampler for Rectified Flow models.
    
    The key parameter is ETA:
      0.0 = Deterministic ODE (like Euler/RF-Solver)
      1.0 = Full ancestral SDE (fresh noise each step — best for LTX-2)
    
    The predictor controls how x̂₀ is estimated:
      euler = 1 NFE (matches SA-Solver predictor_order=1)
      rf_solver_2 = 2 NFE (better x̂₀ via second-order correction)
      ab2 = 1 NFE after warmup (reuses velocity history)
    
    Empirically, eta=1.0 + euler matches SA-Solver's best LTX-2 settings.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": (
                        "Stochasticity. 0.0 = deterministic ODE, "
                        "1.0 = full ancestral SDE (recommended for LTX-2). "
                        ">1.0 = extra noise (experimental)."
                    ),
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": (
                        "Noise multiplier on the stochastic component. "
                        "1.0 = standard. Only matters when eta > 0."
                    ),
                }),
                "predictor": (["euler", "rf_solver_2", "ab2"], {
                    "default": "euler",
                    "tooltip": (
                        "Data prediction method. "
                        "euler: 1 NFE/step (matches SA-Solver optimal). "
                        "rf_solver_2: 2 NFE/step (better x̂₀). "
                        "ab2: 1 NFE/step after warmup (multistep)."
                    ),
                }),
                "sde_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Normalized sigma below which SDE is active. "
                        "0.0 = SDE from the start."
                    ),
                }),
                "sde_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Normalized sigma above which SDE is active. "
                        "1.0 = SDE until the end."
                    ),
                }),
            },
        }
    
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "ZLD/sampling/solvers"
    
    DESCRIPTION = (
        "SA-RF-Solver v2: SDE sampler for Rectified Flow models (LTX-2, Flux, "
        "SD3, Wan, HunyuanVideo). DDIM-eta for RF's linear schedule. "
    )
    
    def get_sampler(self, eta, s_noise, predictor, sde_start, sde_end):
        sampler = comfy.samplers.KSAMPLER(
            sa_rf_sample,
            extra_options={
                "eta": eta,
                "s_noise": s_noise,
                "predictor": predictor,
                "sde_start": sde_start,
                "sde_end": sde_end,
            },
        )
        return (sampler,)



# :: ==========================================================================
# :: ==========================================================================
# ::
# :: RF Solver Sampler
# :: 
# :: ==========================================================================
# :: ==========================================================================
class RFSolverSamplerNode:
    """
    RF-Solver: Pure deterministic ODE sampler for rectified flow models.
    
    Use this when you need deterministic output (reproducibility, inversion).
    For best generation quality on LTX-2, use SA-RF-Solver with eta=1.0.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "order": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 2,
                    "step": 1,
                    "tooltip": (
                        "1 = Euler (1 NFE/step). "
                        "2 = RF-Solver-2 data prediction averaging (2 NFE/step)."
                    ),
                }),
            },
        }
    
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "ZLD/sampling/solvers"
    
    DESCRIPTION = (
        "RF-Solver: Deterministic ODE sampler for rectified flow models. "
    )
    
    def get_sampler(self, order):
        predictor = "euler" if order == 1 else "rf_solver_2"
        sampler = comfy.samplers.KSAMPLER(
            sa_rf_sample,
            extra_options={
                "eta": 0.0,
                "s_noise": 1.0,
                "predictor": predictor,
                "sde_start": 0.0,
                "sde_end": 1.0,
            },
        )
        return (sampler,)



# :: ==========================================================================
# :: ==========================================================================
# ::
# :: Node Registration
# :: 
# :: ==========================================================================
# :: ==========================================================================

NODE_CLASS_MAPPINGS = {
    "EMAGGuider": EMAGGuider,
    "EMASyncGuider": EMASyncGuider,
    "LTX2ScheduledEnhanceVideo": LTX2ScheduledEnhanceVideo,
    "FreqDecompTemporalGuidance": FreqDecompTemporalGuidance,
    "SARFSolverSampler": SARFSolverSamplerNode,
    "RFSolverSampler": RFSolverSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EMAGGuider": "EMAG Guider",
    "EMASyncGuider": "EMAG + SyncCFG Hybrid Guider",
    "LTX2ScheduledEnhanceVideo": "LTX-2 Scheduled Enhance-A-Video",
    "FreqDecompTemporalGuidance": "Frequency-Decomposed Temporal Guidance",
    "SARFSolverSampler": "SA-RF-Solver Sampler",
    "RFSolverSampler": "RF-Solver Sampler",
}
