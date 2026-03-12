import torch
import torch.nn.functional as F
import comfy.samplers

class EMASyncGuiderImpl(comfy.samplers.CFGGuider):
    """
    Implementation of EMAG + SyncCFG hybrid guidance.
    
    SyncCFG Theory (from Harmony paper):
    Standard CFG: ε = ε_uncond + s*(ε_cond - ε_uncond)
    Problem: Amplifying (ε_cond - ε_uncond) improves quality but hurts sync
    
    SyncCFG decouples:
    - ε_v: video generation path
    - ε_a: audio generation path  
    - ε_sync: synchronization alignment path
    
    Final: ε = ε_uncond + s_v*(ε_v - ε_uncond) + s_a*(ε_a - ε_uncond) + s_sync*(ε_sync - ε_uncond)
    
    Where ε_sync is computed to preserve audio-video correspondence.
    """
    
    # Class-level keys for persistent state
    _EMA_STATE_KEY = '_emasync_ema_state'
    _EMA_STEP_KEY = '_emasync_step_counter'
    _EMA_FIRST_TIMESTEP_KEY = '_emasync_first_timestep'
    
    def __init__(self, model_patcher, mode, cfg, emag_scale, ema_decay,
                 sync_scale, video_scale, audio_scale,
                 start_percent, end_percent, adaptive_layers, perturb_img_to_text,
                 separate_audio_video_cond):
        super().__init__(model_patcher)
        
        # Mode configuration
        self.mode = mode
        self.separate_audio_video_cond = separate_audio_video_cond
        
        # EMAG parameters
        self.emag_scale = emag_scale
        self.ema_decay = ema_decay
        self.adaptive_layers = adaptive_layers
        self.perturb_img_to_text = perturb_img_to_text
        
        # SyncCFG parameters
        self.sync_scale = sync_scale
        self.video_scale = video_scale
        self.audio_scale = audio_scale
        
        # Scheduling
        self.start_percent = start_percent
        self.end_percent = end_percent
        
        # State
        self._hook_handles = []
        self.total_steps = None
        
        # Separate conditionings for SyncCFG
        self.audio_positive = None
        self.video_positive = None
        
    def _get_persistent_ema(self):
        """Get or create persistent EMA state on model_patcher"""
        if not hasattr(self.model_patcher, self._EMA_STATE_KEY):
            setattr(self.model_patcher, self._EMA_STATE_KEY, {})
        return getattr(self.model_patcher, self._EMA_STATE_KEY)
    
    def _get_persistent_step(self):
        """Get persistent step counter"""
        if not hasattr(self.model_patcher, self._EMA_STEP_KEY):
            setattr(self.model_patcher, self._EMA_STEP_KEY, 0)
        return getattr(self.model_patcher, self._EMA_STEP_KEY)
    
    def _set_persistent_step(self, step):
        """Set persistent step counter"""
        setattr(self.model_patcher, self._EMA_STEP_KEY, step)
        
    def _detect_new_generation(self, timestep):
        """Detect if this is a new generation and reset EMA if needed"""
        timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else float(timestep)
        
        if not hasattr(self.model_patcher, self._EMA_FIRST_TIMESTEP_KEY):
            setattr(self.model_patcher, self._EMA_FIRST_TIMESTEP_KEY, timestep_val)
            setattr(self.model_patcher, '_emasync_last_timestep', timestep_val)
            return True
        
        last_timestep = getattr(self.model_patcher, '_emasync_last_timestep', 0.0)
        
        # Diffusion timesteps decrease over generation
        if timestep_val > last_timestep + 0.01:
            print(f"[EMASync] New generation detected ({timestep_val:.4f} > {last_timestep:.4f}). Resetting state.")
            setattr(self.model_patcher, self._EMA_FIRST_TIMESTEP_KEY, timestep_val)
            setattr(self.model_patcher, '_emasync_last_timestep', timestep_val)
            return True
        
        setattr(self.model_patcher, '_emasync_last_timestep', timestep_val)
        return False
        
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """
        Main prediction method implementing EMAG + SyncCFG hybrid logic.
        """
        # Detect new generation
        is_new_gen = self._detect_new_generation(timestep)
        if is_new_gen:
            self._get_persistent_ema().clear()
            self._set_persistent_step(0)
        
        ema_attention = self._get_persistent_ema()
        current_step = self._get_persistent_step()
        
        if self.total_steps is None and 'sigmas' in model_options:
            self.total_steps = len(model_options['sigmas']) - 1
        
        apply_guidance = self._should_apply(current_step)
        
        print(f"[EMASync] Mode={self.mode}, Step={current_step}, Apply={apply_guidance}")
        
        if not apply_guidance:
            # Standard CFG fallback
            result = super().predict_noise(x, timestep, model_options, seed)
            self._set_persistent_step(current_step + 1)
            return result
        
        # Route to appropriate implementation
        if self.mode == "EMAG_ONLY":
            result = self._predict_emag_only(x, timestep, model_options, ema_attention)
        elif self.mode == "SYNCCFG_ONLY":
            result = self._predict_synccfg_only(x, timestep, model_options)
        else:  # HYBRID
            result = self._predict_hybrid(x, timestep, model_options, ema_attention)
        
        self._set_persistent_step(current_step + 1)
        return result
    
    def _predict_emag_only(self, x, timestep, model_options, ema_attention):
        """Original EMAG implementation"""
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        
        self._register_emag_hooks(ema_attention)
        
        try:
            out_perturbed = comfy.samplers.calc_cond_batch(
                self.inner_model, 
                [negative_cond, positive_cond], 
                x, 
                timestep, 
                model_options
            )
            cond_pred_perturbed = out_perturbed[1]
            uncond_pred = out_perturbed[0]
        finally:
            self._remove_emag_hooks()
        
        out_standard = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [positive_cond],
            x,
            timestep,
            model_options
        )
        cond_pred_standard = out_standard[0]
        
        # Eq. 15 & 16 from EMAG paper
        emag_guidance = cond_pred_standard + self.emag_scale * (cond_pred_standard - cond_pred_perturbed)
        noise_pred = uncond_pred + self.cfg * (emag_guidance - uncond_pred)
        
        return noise_pred
    
    def _predict_synccfg_only(self, x, timestep, model_options):
        """
        Pure SyncCFG implementation without EMA perturbation.
        
        Implements decoupled guidance:
        ε = ε_uncond + s_v*(ε_v - ε_uncond) + s_a*(ε_a - ε_uncond) + s_sync*(ε_sync - ε_uncond)
        
        For LTX-2, we assume the model has audio and video components that can be
        guided separately while maintaining synchronization.
        """
        negative_cond = self.conds.get("negative", None)
        
        if self.separate_audio_video_cond and self.audio_positive is not None and self.video_positive is not None:
            # Separate conditionings provided
            audio_cond = self.audio_positive
            video_cond = self.video_positive
            # For sync, we use the joint conditioning (positive) or compute a sync-specific one
            sync_cond = self.conds.get("positive", None)
        else:
            # Use same conditioning for all (degraded but functional)
            joint_cond = self.conds.get("positive", None)
            audio_cond = joint_cond
            video_cond = joint_cond
            sync_cond = joint_cond
        
        # Get unconditional prediction
        out_uncond = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [negative_cond],
            x,
            timestep,
            model_options
        )
        uncond_pred = out_uncond[0]
        
        # Get conditional predictions
        # Audio path
        out_audio = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [audio_cond],
            x,
            timestep,
            model_options
        )
        audio_pred = out_audio[0]
        
        # Video path
        out_video = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [video_cond],
            x,
            timestep,
            model_options
        )
        video_pred = out_video[0]
        
        # Sync path - for synchronization, we want the model to focus on alignment
        # This can be implemented as a special conditioning or as the joint prediction
        out_sync = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [sync_cond],
            x,
            timestep,
            model_options
        )
        sync_pred = out_sync[0]
        
        # Compute SyncCFG
        # ε = ε_uncond + s_v*(ε_v - ε_uncond) + s_a*(ε_a - ε_uncond) + s_sync*(ε_sync - ε_uncond)
        noise_pred = uncond_pred.clone()
        
        if self.video_scale > 0:
            noise_pred += self.video_scale * (video_pred - uncond_pred)
        
        if self.audio_scale > 0:
            noise_pred += self.audio_scale * (audio_pred - uncond_pred)
        
        if self.sync_scale > 0:
            # The sync term helps maintain alignment between audio and video
            # We compute it as the component that preserves correspondence
            sync_guidance = sync_pred - uncond_pred
            
            # Optional: Project sync guidance to be orthogonal to video/audio guidance
            # to ensure it only affects alignment, not quality
            if self.video_scale > 0 or self.audio_scale > 0:
                combined_quality = (self.video_scale * (video_pred - uncond_pred) + 
                                  self.audio_scale * (audio_pred - uncond_pred))
                combined_quality_norm = combined_quality.norm() + 1e-8
                # Remove component of sync that's parallel to quality guidance
                projection = (sync_guidance * combined_quality).sum() / (combined_quality_norm ** 2) * combined_quality
                sync_guidance = sync_guidance - projection
            
            noise_pred += self.sync_scale * sync_guidance
        
        return noise_pred
    
    def _predict_hybrid(self, x, timestep, model_options, ema_attention):
        """
        HYBRID mode: EMAG perturbation + SyncCFG guidance structure.
        
        First applies EMA perturbation to create hard negatives,
        then applies SyncCFG decoupled guidance on the perturbed predictions.
        """
        negative_cond = self.conds.get("negative", None)
        
        if self.separate_audio_video_cond and self.audio_positive is not None and self.video_positive is not None:
            audio_cond = self.audio_positive
            video_cond = self.video_positive
            sync_cond = self.conds.get("positive", None)
        else:
            joint_cond = self.conds.get("positive", None)
            audio_cond = joint_cond
            video_cond = joint_cond
            sync_cond = joint_cond
        
        # Register EMA hooks for perturbation
        self._register_emag_hooks(ema_attention)
        
        try:
            # Get perturbed predictions (with EMA hooks active)
            # We need perturbed versions of uncond, audio, video, and sync
            
            # Batch all conditionings for efficiency
            all_conds = [negative_cond, audio_cond, video_cond, sync_cond]
            out_perturbed = comfy.samplers.calc_cond_batch(
                self.inner_model,
                all_conds,
                x,
                timestep,
                model_options
            )
            
            uncond_perturbed = out_perturbed[0]
            audio_perturbed = out_perturbed[1]
            video_perturbed = out_perturbed[2]
            sync_perturbed = out_perturbed[3]
            
        finally:
            self._remove_emag_hooks()
        
        # Get standard (non-perturbed) predictions
        out_standard = comfy.samplers.calc_cond_batch(
            self.inner_model,
            all_conds,
            x,
            timestep,
            model_options
        )
        
        uncond_standard = out_standard[0]
        audio_standard = out_standard[1]
        video_standard = out_standard[2]
        sync_standard = out_standard[3]
        
        # Apply EMAG to each prediction (Eq. 15 from EMAG paper)
        # ε̃' = ε' + w_e * (ε - ε')
        def apply_emag(standard, perturbed):
            return standard + self.emag_scale * (standard - perturbed)
        
        uncond_emag = apply_emag(uncond_standard, uncond_perturbed)
        audio_emag = apply_emag(audio_standard, audio_perturbed)
        video_emag = apply_emag(video_standard, video_perturbed)
        sync_emag = apply_emag(sync_standard, sync_perturbed)
        
        # Now apply SyncCFG structure with EMAG-enhanced predictions
        # ε = ε_uncond + s_v*(ε_v - ε_uncond) + s_a*(ε_a - ε_uncond) + s_sync*(ε_sync - ε_uncond)
        noise_pred = uncond_emag.clone()
        
        if self.video_scale > 0:
            noise_pred += self.video_scale * (video_emag - uncond_emag)
        
        if self.audio_scale > 0:
            noise_pred += self.audio_scale * (audio_emag - uncond_emag)
        
        if self.sync_scale > 0:
            sync_guidance = sync_emag - uncond_emag
            
            # Orthogonal projection to isolate alignment component
            if self.video_scale > 0 or self.audio_scale > 0:
                combined_quality = (self.video_scale * (video_emag - uncond_emag) + 
                                  self.audio_scale * (audio_emag - uncond_emag))
                combined_quality_norm = combined_quality.norm() + 1e-8
                projection = (sync_guidance * combined_quality).sum() / (combined_quality_norm ** 2) * combined_quality
                sync_guidance = sync_guidance - projection
            
            noise_pred += self.sync_scale * sync_guidance
        
        return noise_pred
    
    def _should_apply(self, current_step):
        """Check if guidance should be applied based on step schedule"""
        if self.total_steps is None or self.total_steps == 0:
            return True
        
        start_step = int(self.start_percent * self.total_steps)
        end_step = int(self.end_percent * self.total_steps)
        
        # Apply between end_step and start_step (diffusion goes high->low noise)
        return end_step <= current_step <= start_step
    
    def _register_emag_hooks(self, ema_attention):
        """Register EMA perturbation hooks (same as original EMAG)"""
        self._remove_emag_hooks()
        
        try:
            model = self.model_patcher.model
            blocks = self._find_transformer_blocks(model)
            
            if blocks is None:
                print("[EMASync] Warning: Could not find transformer blocks")
                return
            
            if self.adaptive_layers:
                layers_to_perturb = self._select_layers_adaptive(blocks)
            else:
                layers_to_perturb = list(range(len(blocks)))
            
            for idx in layers_to_perturb:
                if idx >= len(blocks):
                    continue
                block = blocks[idx]
                self._hook_attention_modules(block, idx, ema_attention)
            
            print(f"[EMASync] Registered {len(self._hook_handles)} hooks on layers {layers_to_perturb}")
                        
        except Exception as e:
            print(f"[EMASync] Warning: Could not register hooks: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_transformer_blocks(self, model):
        """Find transformer blocks across different architectures"""
        if hasattr(model, 'diffusion_model'):
            dm = model.diffusion_model
            for attr in ('transformer_blocks', 'joint_blocks', 'blocks'):
                if hasattr(dm, attr):
                    return getattr(dm, attr)
        
        for attr in ('transformer_blocks', 'joint_blocks', 'blocks'):
            if hasattr(model, attr):
                return getattr(model, attr)
        
        return None
    
    def _hook_attention_modules(self, block, layer_idx, ema_attention):
        """Hook attention modules for EMA perturbation"""
        # Self-attention
        for attr_name in ('attn1', 'attn', 'self_attn'):
            if hasattr(block, attr_name):
                module = getattr(block, attr_name)
                handle = module.register_forward_hook(
                    self._make_emag_hook(layer_idx, 'self', ema_attention)
                )
                self._hook_handles.append(handle)
                break
        
        # Cross-attention if enabled
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
        """Remove all EMA hooks"""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except:
                pass
        self._hook_handles.clear()
    
    def _make_emag_hook(self, layer_idx, attn_type, ema_attention):
        """Create EMA perturbation hook"""
        decay = self.ema_decay
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None
            
            key = f"layer_{layer_idx}_{attn_type}"
            
            if key in ema_attention:
                ema_attn = ema_attention[key]
                
                if ema_attn.shape == attn_output.shape:
                    old_ema = ema_attn
                    new_ema = (decay * ema_attn + (1.0 - decay) * attn_output.detach())
                    ema_attention[key] = new_ema
                    
                    if rest is not None:
                        return (old_ema,) + rest
                    else:
                        return old_ema
                else:
                    print(f"[EMASync] Shape mismatch for {key}: {ema_attn.shape} vs {attn_output.shape}. Reinitializing.")
                    ema_attention[key] = attn_output.detach().clone()
                    return output
            
            ema_attention[key] = attn_output.detach().clone()
            return output
        
        return hook
    
    def _select_layers_adaptive(self, blocks):
        """Adaptive layer selection from EMAG paper"""
        n_layers = len(blocks)
        
        if n_layers <= 12:
            start = max(0, n_layers // 2 - 1)
            end = min(n_layers, n_layers // 2 + 2)
            return list(range(start, end))
        else:
            return list(range(12, min(n_layers, 16)))