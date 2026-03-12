import torch
import math
from tqdm import trange
import comfy.samplers


# ═══════════════════════════════════════════════════════════════════════════
# Core RF-SDE step
# ═══════════════════════════════════════════════════════════════════════════

def _rf_sde_step(z, denoised, sigma, sigma_next, eta, s_noise=1.0):
    """
    Single RF-SDE step: DDIM-eta formulation for rectified flow.
    
    This is the fundamental building block. Everything else is about
    getting a better `denoised` (x̂₀) estimate to feed into this.
    
    Args:
        z: Current latent at sigma
        denoised: Model's data prediction x̂₀
        sigma: Current noise level (tensor, batch-broadcastable)
        sigma_next: Next noise level (tensor)
        eta: Stochasticity (0=ODE, 1=full SDE/ancestral)
        s_noise: Noise multiplier (default 1.0)
    
    Returns:
        z_next: Latent at sigma_next
    """
    if torch.all(sigma_next == 0):
        return denoised
    
    # Predicted noise: ε̂ = (z - (1-σ)·x̂₀) / σ
    pred_noise = (z - (1.0 - sigma) * denoised) / sigma.clamp(min=1e-8)
    
    # Noise mixing coefficients
    det_coeff = math.sqrt(max(1.0 - eta * eta, 0.0))
    stoch_coeff = eta * s_noise
    
    # z_next = (1-σ_next)·x̂₀ + σ_next·[det·ε̂ + stoch·ε_new]
    z_next = (1.0 - sigma_next) * denoised + sigma_next * det_coeff * pred_noise
    
    if stoch_coeff > 0:
        noise = torch.randn_like(z)
        z_next = z_next + sigma_next * stoch_coeff * noise
    
    return z_next


# ═══════════════════════════════════════════════════════════════════════════
# Data prediction strategies
# ═══════════════════════════════════════════════════════════════════════════

def _predict_euler(model, z, sigma, sigma_next, extra_args, history):
    """
    Euler data prediction: single model evaluation.
    1 NFE. Matches SA-Solver at predictor_order=1.
    """
    denoised = model(z, sigma, **extra_args)
    return denoised, denoised, history


def _predict_rf_solver_2(model, z, sigma, sigma_next, extra_args, history):
    """
    RF-Solver-2 data prediction: Euler predictor + corrector average.
    2 NFE. Better x̂₀ estimate. Worth testing even at η=1.
    """
    denoised_cur = model(z, sigma, **extra_args)
    
    if torch.all(sigma_next == 0):
        return denoised_cur, denoised_cur, history
    
    # Euler predictor step (deterministic, just for evaluating at σ_next)
    ratio = sigma_next / sigma
    z_pred = ratio * z + (1.0 - ratio) * denoised_cur
    
    # Second evaluation at predicted point
    denoised_pred = model(z_pred, sigma_next, **extra_args)
    
    # Average data predictions (RF-Solver-2 correction)
    denoised_avg = 0.5 * (denoised_cur + denoised_pred)
    
    return denoised_avg, denoised_cur, history


def _predict_ab2(model, z, sigma, sigma_next, extra_args, history):
    """
    Adams-Bashforth-2 data prediction via velocity extrapolation.
    1 NFE after warmup (first step uses RF-Solver-2 at 2 NFE).
    """
    denoised = model(z, sigma, **extra_args)
    
    if torch.all(sigma_next == 0):
        return denoised, denoised, history
    
    v_cur = (z - denoised) / sigma.clamp(min=1e-8)
    h_cur = sigma_next - sigma
    
    v_prev = history.get('v_prev', None)
    h_prev = history.get('h_prev', None)
    
    if v_prev is None or h_prev is None:
        # Warmup: RF-Solver-2 (costs 1 extra NFE this step only)
        ratio = sigma_next / sigma
        z_pred = ratio * z + (1.0 - ratio) * denoised
        denoised_pred = model(z_pred, sigma_next, **extra_args)
        denoised_avg = 0.5 * (denoised + denoised_pred)
        
        history['v_prev'] = v_cur.clone()
        history['h_prev'] = h_cur
        return denoised_avg, denoised, history
    
    # AB2 velocity extrapolation
    r = h_cur / h_prev
    v_ab2 = (1.0 + r / 2.0) * v_cur - (r / 2.0) * v_prev
    
    # Convert back to data prediction: x̂₀ = z - σ·v
    denoised_ab2 = z - sigma * v_ab2
    
    history['v_prev'] = v_cur.clone()
    history['h_prev'] = h_cur
    return denoised_ab2, denoised, history


PREDICTORS = {
    "euler": _predict_euler,
    "rf_solver_2": _predict_rf_solver_2,
    "ab2": _predict_ab2,
}


# ═══════════════════════════════════════════════════════════════════════════
# Main sampling loop
# ═══════════════════════════════════════════════════════════════════════════

def sa_rf_sample(model, x, sigmas, extra_args=None, callback=None,
                 disable=None, eta=1.0, s_noise=1.0,
                 predictor="euler", sde_start=0.0, sde_end=1.0):
    """
    SA-RF-Solver sampling loop.
    
    Args:
        model: ComfyUI model wrapper
        x: Initial noisy latent
        sigmas: Sigma schedule (high → low, ending in 0)
        extra_args: Conditioning dict
        callback: Progress callback
        disable: Disable progress bar
        eta: Stochasticity (0=ODE, 1=full ancestral SDE)
        s_noise: Noise multiplier
        predictor: Data prediction method
        sde_start: Normalized sigma below which SDE is active (0.0=start)
        sde_end: Normalized sigma above which SDE is active (1.0=end)
    """
    extra_args = {} if extra_args is None else extra_args
    predict_fn = PREDICTORS[predictor]
    s_in = x.new_ones([x.shape[0]])
    history = {}
    
    sigma_max = float(sigmas[0])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i] * s_in
        sigma_next = sigmas[i + 1] * s_in
        
        # Get data prediction x̂₀
        denoised, denoised_raw, history = predict_fn(
            model, x, sigma, sigma_next, extra_args, history
        )
        
        # SDE range gating
        sigma_frac = float(sigma.flatten()[0]) / max(sigma_max, 1e-8)
        if sde_start <= sigma_frac <= sde_end:
            step_eta = eta
        else:
            step_eta = 0.0
        
        # Take the RF-SDE step
        x = _rf_sde_step(x, denoised, sigma, sigma_next, step_eta, s_noise)
        
        if callback is not None:
            callback({
                'x': x, 'i': i,
                'sigma': sigmas[i], 'sigma_next': sigmas[i + 1],
                'denoised': denoised_raw,
            })
    
    return x