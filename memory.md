# Memory Notes — RecurrentGemma PyTorch Models

## 2026-07-04 — GALR Stereo Verification

**GALR implementation verified** (`aura/neural_acoustics/galr_teacher.py`):
- Time-domain masks: 2D conv → tanh/sigmoid beam-forming → ReLU (paper Eq. 12)
- Stereo support: `in_channels: int = 2` in `GALRTeacherConfig`
- PIT: Implemented via `itertools.permutations`
- Decoder: One transposed 1D conv per source output

**Stereo input contract**: `(B, 2, T)` → encoder → segment → blocks → mask head → `(B, n_sources, T)`

## Work Needed for Arbitrary Channel Loading (DEMAND)

DEMAND has **separate mono WAV files per channel** (`ch01.wav` through `ch16.wav`), not interleaved stereo.

### Current DEMAND Loader
```python
# demand.py line 47: channels: str = "ch01"  — loads single channel
# demand.py line 102: if f.endswith(f"{self.channels}.wav")
# demand.py line 133: data = data.mean(axis=1) — never hits (each WAV is mono)
```

### Work for GALR Stereo Training
1. **Multi-channel selection API**: Load 2+ specific channels and stack:
   ```python
   # channels="all" → stack all 16 channels → shape (16, T)
   # channels=[1,3,5] → select subset → shape (3, T)
   # channels="mean" → load all, mean across → shape (1, T)
   ```
2. **DEMAND → stereo loader**: Load any 2 channels and present as `(2, T)` for GALR

### Files to Modify
| File | Change |
|---|---|
| `shared/data/loaders/demand.py` | Multi-channel selection API (`channels` param accepts str/list) |
| `aura/neural_acoustics/galr_teacher.py` | Already stereo-capable via `in_channels=2` |

---

## 2026-07-04 — Audit: FFT/STFT Alignment with PNAS Paper

### FFT/STFT Status: Confirmed Time-Domain Aligned

**Files audited**:
- `recurrentgemma/torch/layers.py` — RGLRU, Conv1D, RMSNorm
- `recurrentgemma/torch/modules.py` — RecurrentBlock, MLPBlock, ResidualBlock
- `recurrentgemma/torch/griffin.py` — Griffin model assembly
- `recurrentgemma/jax/layers.py` — JAX reference implementations

**Finding**: All implementations operate in **time-domain only**. The RecurrentGemma
Griffin architecture uses:
- `RGLRU` — Real-Gated Linear Recurrent Unit (time-domain recurrent)
- `Conv1D` — 1D temporal convolution (time-domain)
- No STFT/FFT operations on audio signals anywhere in these modules

**Conclusion**: DeepONet/FiLM implementations in `aura/neural_acoustics/` are
time-domain aligned with the PNAS paper — no changes needed.

### PNAS Fourier Features Clarification

The PNAS paper (DOI 10.1073/pnas.2312159120) uses Fourier encoding on
**spatial-temporal coordinates** (x, y, z, t), not audio spectral transforms:

```python
# In fourierFeatureExpansion_f0 (feature_expansion.py):
# Input: (..., 4) for (x, y, z, t) coordinates
# Output: [..., 4 + 2*len(fs)] with cos/sin of frequencies fs
```

This is positional encoding for the DeepONet trunk net, not a spectral transform
on audio waveforms. The DeepONet operator learns acoustic wave propagation from
initial condition (pressure field on a spatial mesh) to query points.

### Wavelet Alternative Available

For multi-scale time-domain analysis, `aum/wavelet/analysis.py` provides:
- `wavelet_filter_bank()` — undecimated Db4 multi-scale decomposition
- `wavelet_coherence()` — signal coherence measure across scales

These can be used as alternatives to FFT-based feature extraction when frequency-domain
analysis is needed without the ROCm FFT instability issues.

### PyTorch vs JAX Source Verification

Both sources verified working:
1. **HF Transformers RecurrentGemma** — `recurrentgemma` pip package from kryptodogg fork
2. **In-repo `models/recurrentgemma/recurrentgemma/torch/`** — PyTorch implementations

Numerical parity tests (`layers_test.py`, `modules_test.py`, `griffin_test.py`) confirm
both paths produce equivalent outputs. This allows benchmarking HF Transformers variants
against Hawk and paper Griffin implementations.

---

## 2026-07-04 — DEMAND Multi-Channel Loader Refactor (COMPLETE)

**Implemented in `shared/data/loaders/demand.py`:**
- `_resolve_channels()` — parses `channels` param (str/list) → list of channel names
- `_discover_files()` — groups files by base key, collects paths per channel
- `_load_audio()` — loads and stacks multiple channels → `(n_channels, T)` shape

**API**:
```python
# channels="ch01" (default) → single channel, shape (1, T)
# channels=[1, 3, 5] → select channels, shape (3, T)
# channels="all" → stack all 16, shape (16, T)
# channels="mean" → load all, mean across, shape (1, T)
```

**Tests**: `tests/shared/data/test_demand_loader.py` validates all channel specs.

---

## Deferred Work

- `room_loader.py` stereo preservation — separate task