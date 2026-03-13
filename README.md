# ZLD NODES FOR COMFYUI

## EMAG 
    EMAG (Exponential Moving Average Guidance) Guider Node
    A drop-in replacement for CFGGuider that applies EMA-based attention perturbation
    to create hard negatives for improved guidance.
    
    Based on: "EMAG: Exponential Moving Average Guidance for Diffusion Models"
    Paper equations implemented: Eq. 12 (EMA), Eq. 15 (EMAG update), Eq. 16 (CFG with EMAG)

## EMASync
    EMAG + SyncCFG Hybrid Guider Node
    Combines Exponential Moving Average Guidance with Synchronization-Enhanced CFG for  
    improved audio-video generation alignment.
      
    Modes:
    - EMAG_ONLY: Standard EMAG guidance (original behavior)
    - SYNCCFG_ONLY: Pure SyncCFG without EMA perturbation
    - HYBRID: EMAG perturbation + SyncCFG guidance structure
    
    Based on: 
    - "EMAG: Exponential Moving Average Guidance for Diffusion Models"
    - "Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy"

## Scheduled EAV LTX2
    Based on the node by Kijai, this expands this node by providing the ability to 
    schedule the Enhance-A-Video tau value. EAV can disrupt and suppress high frequency 
    (fine) details that are desired in later steps. This allows you to mitigate that 
    occurrence while gaining substantial coherence improvements with lower frequencies 
    earlier in the process.

## FDTG
    Frequency-Decomposed Temporal Guidance for video diffusion models applies wavelet 
    decomposition to the model's velocity prediction, then applies differentiated guidance  
    to low-vs-high frequency bands with optional temporal consistency enforcement on high-
    frequency content.

## RF-Solver 
    RF-Solver: Pure deterministic ODE sampler for rectified flow models.
    
    Use this when you need deterministic output (reproducibility, inversion). For best 
    generation quality on LTX-2, use SA-RF-Solver with eta=1.0.


## SA-RF-Solver
    SA-RF-Solver v2: Proper SDE sampler for Rectified Flow models.
    
    The key parameter is ETA:
      0.0 = Deterministic ODE (like Euler/RF-Solver)
      1.0 = Full ancestral SDE (fresh noise each step — best for LTX-2)
    
    The predictor controls how x̂₀ is estimated:
      euler = 1 NFE (matches SA-Solver predictor_order=1)
      rf_solver_2 = 2 NFE (better x̂₀ via second-order correction)
      ab2 = 1 NFE after warmup (reuses velocity history)
    
    Empirically, eta=1.0 + euler matches SA-Solver's best LTX-2 settings.

## LTXVImgToVideoInplaceNoCrop
    Drop-in replacement that resizes the input image to the target latent dimensions WITHOUT 
    center-cropping.  The only change from the stock node is  "center" → "disabled"  in the 
    common_upscale call, which forces a direct resize (stretch-to-fit) instead of cover-
    then-crop.

---------------------

## LTX Cinema Workflows

| Component | High | Medium | Low | Fast |
|-----------|------|--------|-----|------|
| **S2 Guider** | EMASyncGuider HYBRID | EMAGGuider | EMAGGuider | CFGGuider (cfg=1) |
| **S2 Sampler** | SA-RF-Solver (`rf_solver_2`, η=1.05) | SA-RF-Solver (`rf_solver_2`, η=1.05) | SA-Solver (τ=1.0) | SA-Solver (τ=1.0) |
| **S3/S4 Guider** | EMASyncGuider HYBRID | EMAGGuider | EMAGGuider | CFGGuider (cfg=1) |
| **S3/S4 Sampler** | SA-RF-Solver (`euler`, η=1.0) | SA-RF-Solver (`euler`, η=1.0) | SA-Solver (τ=0.2) | SA-Solver (τ=0.2) |
| **EMAG active** | Yes (via SyncCFG) | Yes (end=0.2) | Yes (end=0.2) | No (end=1.0 = disabled) |
| **Sync scheduling** | Yes (0.9→0.7) | No | No | No |
| **Duration (RTX3090)** | ~25m / 5s | ~16m / 5s | ~12m / 5s | ~6m / 5s |

---------------------

## Papers Referenced

| Technique | Paper | arXiv |
|-----------|-------|-------|
| RF-Solver | Wang et al., 2024 | [2411.04746](https://arxiv.org/abs/2411.04746) |
| SA-Solver | Xue et al., NeurIPS 2023 | — |
| EMAG | Yadav et al., 2025 | [2512.17303](https://arxiv.org/abs/2512.17303) |
| Enhance-A-Video | NUS HPC AI Lab, 2025 | [2502.07508](https://arxiv.org/abs/2502.07508) |
| CFG-Zero* | Fan et al., 2025 | [2503.18886](https://arxiv.org/abs/2503.18886) |
| FDG | 2025 | [2506.19713](https://arxiv.org/abs/2506.19713) |
| LTX-Video 2 | Lightricks, 2026 | [2601.03233](https://arxiv.org/abs/2601.03233) |