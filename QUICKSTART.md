# EMAG ComfyUI Custom Node - Quick Start

## Installation (3 steps)

1. **Create the folder**:
   ```
   ComfyUI/custom_nodes/comfyui-emag/
   ```

2. **Copy these 4 files** into that folder:
   - `__init__.py`
   - `node.py`
   - `README.md` (optional, for reference)
   - `requirements.txt` (optional, no extra deps needed)

3. **Restart ComfyUI**

## Finding the Node

After restart, search for **"EMAG Guider"** in the node menu, or navigate to:
```
Add Node → sampling → custom_sampling → guiders → EMAG Guider
```

## Basic Usage

### Replace CFGGuider with EMAG Guider

**Before (Standard CFG)**:
```
Model → CFGGuider → SamplerCustomAdvanced → Output
Positive →     ↑
Negative →     ↑
```

**After (With EMAG)**:
```
Model → EMAG Guider → SamplerCustomAdvanced → Output  
Positive →     ↑
Negative →     ↑
```

### Quick Settings

For best results with **SD3** or similar models:

| Parameter | Value | Why |
|-----------|-------|-----|
| cfg | 7.0 | Standard CFG scale |
| emag_scale | 1.75 | EMAG enhancement (from paper) |
| ema_decay | 0.9 | Smoothing factor |
| start_percent | 1.0 | Start at beginning |
| end_percent | 0.2 | Stop at 20% of steps |
| adaptive_layers | ✓ | Use recommended layers |
| perturb_img_to_text | ✓ | Full EMAG mode |

## What to Expect

**Benefits**:
- ✨ Better detail and refinement
- 📈 Higher quality scores (HPS)
- 🎯 Better prompt adherence
- 🔄 Works with existing workflows

**Trade-offs**:
- ⏱️ ~10-20% slower (extra forward pass)
- 💾 Slightly more memory (EMA storage)

## Troubleshooting

**No visible difference?**
→ Increase `emag_scale` to 2.0-2.5

**Over-sharpened images?**
→ Decrease `emag_scale` to 1.25-1.5

**Errors about transformer blocks?**
→ Your model may not be compatible (needs transformer-based architecture)

## Full Documentation

See **README.md** for:
- Detailed parameter explanations
- Architecture compatibility
- How EMAG works (algorithm details)
- Advanced usage (combining with APG/CADS)
- Paper citation

## File Structure

Your folder should look like:
```
ComfyUI/
└── custom_nodes/
    └── comfyui-emag/
        ├── __init__.py          ← Required
        ├── node.py              ← Required  
        ├── README.md            ← Optional
        └── requirements.txt     ← Optional
```

## Paper Reference

EMAG: Exponential Moving Average Guidance for Diffusion Models  
arXiv: https://arxiv.org/abs/2512.17303

The implementation follows equations 12, 15, and 16 from the paper.

---

**That's it!** You're ready to use EMAG guidance in ComfyUI.
