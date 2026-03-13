# DVD Model Resolution Limits

## Summary

**VDA-style smart resize**: Process at native resolution up to UHD (4096×2160). Downscale only if input exceeds UHD; otherwise use as-is. Output matches processed resolution (maximum quality).

---

## Dimension Constraints

| Constraint | Value | Source |
|------------|-------|--------|
| **Height** | Must be divisible by **16** | `WanVideoPipeline` (`height_division_factor=16`) |
| **Width** | Must be divisible by **16** | `WanVideoPipeline` (`width_division_factor=16`) |
| **Frames** | `num_frames % 4 == 1` (e.g. 1, 5, 9, 81) | `time_division_factor=4`, `time_division_remainder=1` |

`resize_for_training_scale()` in `test_single_video.py` aligns dimensions to 16 before inference.

---

## Recommended Resolutions

| Resolution | Dimensions | VRAM (approx) | Use Case |
|------------|------------|----------------|----------|
| **4K (default)** | 2176 × 3840 | ~24–30 GB | Colab A100 80GB |
| **1080p** | 1088 × 1920 | ~16+ GB | Colab T4/V100 |
| **720p** | 720 × 1280 | ~12–14 GB | Colab T4, faster |
| **480p** | 480 × 640 | ~8 GB | Fastest |

*Cinesculpt DVD cellcode uses 4K by default for 80GB Colab.*

---

## VRAM (Wan2.1 1.3B)

- **480×640**: ~8 GB
- **720×1280**: ~12–14 GB
- **4K (2176×3840)**: ~24–30 GB — Colab A100 80GB
- **Colab A100 80GB**: Use max resolution (4K)

---

## Default Inference Params

From `infer_bash/openworld.sh`:

```bash
HEIGHT=480
WIDTH=640
WINDOW_SIZE=81
OVERLAP=21
```

README: *"You could increase the resolution here but expect slower inference speed."*

---

## VDA-Style Smart Resize

Same logic as VDA engine:

- **UHD max**: 4096×2160
- **Input ≤ UHD**: Use native resolution, align to 16 (DVD requirement)
- **Input > UHD**: Scale down to fit UHD, aspect ratio preserved
- **Output**: Matches processed resolution (maximum quality)

Override via `job_data`: `dvd_window_size`, `dvd_overlap`
