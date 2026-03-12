import torch
import torch.nn.functional as F
import comfy.samplers

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