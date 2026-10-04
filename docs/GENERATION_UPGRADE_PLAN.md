# Generation Upgrade Plan — "Generation Benchmark"

More realistic B-roll clips and sharper stills, at **no more than 5 minutes per
clip** on this PC (RTX 5060 Ti 16 GB, 32 GB RAM, ComfyUI 0.33.3, torch 2.10).
Written 2026-10-03.

**To start it:** close Buzzcaf Studio (let BuzzEdit finish), then double-click
**`BuzzEdit\RUN_GENERATION_BENCHMARK.bat`**. It takes about 45 minutes and opens
the results folder when done.

---

## Why

Today a clip takes ~150 s (`workflows/zimage_wan22_i2v.json`) and looks soft:

- its first frame is generated at only **832×480** (stills are 1584×896);
- Wan 2.2 runs at **832×480, 16 fps**, the most compressed model files
  (**Q4_K_S**) and the older **v1** 4-step speed LoRA;
- nothing upscales or smooths it, so it is stretched to 1080p/4K in the edit.

That leaves ~150 s of the 5-minute budget unused. The plan spends it on quality.

**Kept: Wan 2.2.** LTX-2.3 is ~5.7× faster but in a like-for-like I2V test it
moved the camera and broke hands; Wan held a steady camera and stable motion,
which a calm, realistic story channel needs. LTX also wants 64 GB RAM.

## The upgrade

| Stage | Now | Upgraded | Est. cost |
|---|---|---|---|
| First frame | Z-Image Turbo at 832×480 | Z-Image Turbo at clip size + 1.5× hi-res pass | +~10 s |
| Motion | Wan 2.2 I2V 14B **Q4_K_S**, LoRA **v1**, 832×480 | **Q5_K_M**, lightx2v **1022** LoRAs, **1024×576** | +~75 s |
| Upscale | none | **FlashVSR v1.1** tiny ×2 → 2048×1152 | +~35 s |
| Smoothness | 16 fps | **RIFE v4.26** ×2 → **32 fps** | +~20 s |
| **Clip total** | **~150 s** | **~270 s (estimate)** | ≤ 300 s |
| Still | Z-Image 1584×896, 8 steps (~20 s) | + 1.5× hi-res pass → 2376×1344 (~40 s) | +~20 s |

If the upgraded clip averages over ~290 s, the benchmark automatically re-tests
it at **960×544**.

## Already downloaded (2026-10-03, all size-checked)

| File | Where |
|---|---|
| `wan2.2_i2v_high_noise_14B_Q5_K_M.gguf`, `..._low_noise_...` (10.8 GB each) | `B:\ComfyUI_windows_portable...\ComfyUI\models\unet\` |
| `wan2.2_i2v_A14b_{high,low}_noise_lora_rank64_lightx2v_4step_1022.safetensors` | `...\models\loras\` |
| `rife_v4.26.safetensors` | `C:\Users\singh\Documents\ComfyUI\models\frame_interpolation\` |
| FlashVSR v1.1 (5 files, 6.6 GB) | `C:\Users\singh\Documents\ComfyUI\models\FlashVSR-v1.1\` |
| ComfyUI-FlashVSR_Ultra_Fast node (commit 4820b3f) | staged in `C:\Users\singh\Documents\ComfyUI\custom_nodes_staged\` |

The current Q4_K_S models and v1 LoRAs stay where they are.

## What the benchmark does

`backend\tools\run_generation_benchmark.ps1` (started by the .bat):

1. Refuses to run while Buzzcaf Studio, BuzzEdit or ComfyUI is open.
2. Checks that every downloaded file is present.
3. Installs `triton-windows>=3.6,<3.7` (+ `einops`) into ComfyUI's venv; that
   is the Triton build for torch 2.10 and Blackwell. It warns if torch changes.
4. Moves the FlashVSR node into `custom_nodes` (ComfyUI loads nodes at start).
5. Unloads LM Studio models (GPU rule: nothing else on the card).
6. Runs `backend\tools\bench_generation.py`: starts its own ComfyUI, warms
   up, then renders **5 real Raat3Baje shots** (`backend\tools\bench_prompts.json`:
   mist toward a door, a person walking into a dark corridor, trembling hands on
   a ledger, a torch beam through haze, a hand holding out a book) with the
   current and the upgraded clip workflow (same prompt, same seed), then the
   same for stills. Every job is timed submit → file. It stops its ComfyUI and
   frees the GPU at the end.

**Nothing BuzzEdit uses is changed.** Productions keep the current workflow
until the upgrade is adopted (below).

## Reading the results

`BuzzEdit\data\bench\<date>\`:

- `summary.md` — runs, mean and slowest seconds per variant, "Within 5 min?"
- `compare_clip0..4.png` — left current, right upgraded (frame at 2.5 s)
- `compare_still0..4.png` — left current, right upgraded
- `report.json` — every run, with the path of each full clip to watch
  (ComfyUI `output\bench\`)
- `run.log`, `comfyui.log` — if something failed

Decide:

| If | Then |
|---|---|
| Upgraded clips look better and the slowest is ≤ 300 s | adopt at 1024×576 |
| Better, but over 300 s; the 960×544 run is within | adopt at 960×544 |
| FlashVSR errored (see `summary.md` errors) | adopt without it (RIFE + rest); look at it separately |
| Not clearly better | keep the current workflow; nothing to undo but the files |

## Status: built, then reverted to the old workflows (2026-10-03, at the user's request)

Active again: `zimage_wan22_i2v.json` (manifest, no fallback) and `zimage1.json`
(setting), at 832×480 (`MAX_VIDEO_GEN_PIXELS`). Delivered clips are still fitted
to 1920×1080 by `_fit_clip`. The HQ files below stay on disk, unused. Samples:
`data/bench/2026-10-03_hq_samples/`. To re-enable: set `broll_video.file` to
`zimage_wan22_i2v_hq.json` with `fallback_file: zimage_wan22_i2v.json`, set
`MAX_VIDEO_GEN_PIXELS` to 1024×576, and set the image workflow to `zimage1_hq.json`.

### What was built

| Step | Done |
|---|---|
| 1. HQ clip graph | `workflows/zimage_wan22_i2v_hq.json` (hi-res first frame, Q5_K_M, 1022 LoRAs, 1024×576, RIFE ×2 → 32 fps). `manifest.json` routes `broll_video` to it with `"fallback_file": "zimage_wan22_i2v.json"`: if an HQ clip fails (missing node/model, error — not a timeout), that beat is retried on the old graph and the rest of the pass uses it (`video_phase.fallback` in the stats). |
| FlashVSR | Not installed yet (node only staged, Triton missing), so it is in a separate file, `workflows/zimage_wan22_i2v_hq_flashvsr.json`. After `RUN_GENERATION_BENCHMARK.bat` has enabled the node, pick it under Settings → workflow for B-roll video; it falls back the same way. |
| 2. Size cap | `MAX_VIDEO_GEN_PIXELS` = 1024×576 (`presentation/assets.py`). |
| 3. fps | Clip length is still counted at Wan's 16 fps; `ResolvedWorkflow.build` multiplies the saved fps by the graph's interpolation factor (`frame_multiplier`, read from `FrameInterpolate` nodes), so HQ saves at 32 and the old graph at 16. |
| 4. Stills | `workflows/zimage1_hq.json` (zimage1 + 1.5× hi-res pass → 2376×1344); the `workflow_broll_image` setting points at it. `zimage1.json` is unchanged — set it back in Settings to undo. |
| 5. Timing | See "Measured" below. |
| Benchmark fix | `bench_generation.py` fed RIFE `model=`; ComfyUI 0.33.3's `FrameInterpolate` takes `interp_model=`. Fixed. |

Unchanged and still selectable: `zimage_wan22_i2v.json`, `zimage1.json`, the Q4_K_S models and the v1 LoRAs.

### Measured (2026-10-03, live ComfyUI 0.33.3, LM Studio unloaded, no FlashVSR)

| Job | Seconds | Output |
|---|---|---|
| HQ still (`zimage1_hq.json`), cold | 51 | 2376×1344, clean hands and detail |
| HQ clip, cold | 426 / 431 | 1024×576, 32 fps, 161 frames, 5.03 s; steady push-in, no warping |
| HQ clip, second in a row | 431 | same |
| Upscale stage alone (RealESRGAN ×2 → 1920×1080 → RIFE, 81 frames) | 88 | 1920×1080, 32 fps, 161 frames; cleaner edges than Lanczos, not dramatically sharper |

**Full HD rule (2026-10-03):** every delivered clip is 1920×1080, or the media's
aspect at that pixel count (1080×1920 for Shorts). The HQ graph upscales with
RealESRGAN ×2 + `ImageScale` (`u_scale`, bound as `out_width`/`out_height`)
before RIFE. Any clip that comes back at another size (the old fallback graph,
the FlashVSR variant's 2048×1152) is fitted by `_fit_clip` in
`presentation/assets.py` (ffmpeg Lanczos, NVENC cq 16). Estimated HQ clip now:
~431 + ~88 ≈ **520 s**.

**Over the 300 s budget.** Warm takes as long as cold, so the Q5_K_M experts
(2 × 10.8 GB) are not staying cached in 32 GB RAM. They reload for every clip,
and free RAM fell to 1.8 GB during the run. Caveat: these test graphs made
their own first frame, so Z-Image and Qwen-4B loaded next to Wan. In a
production run the clip starts from the phase-1 still, so Z-Image is never
loaded during the video phase. Production may be faster, but that is not
measured yet. Options: keep HQ at a ~7 min clip, go back to the Q4_K_S
experts with the 1022 LoRAs + RIFE (smaller weights), try 960×544, or set
`broll_video` back to `zimage_wan22_i2v.json` until the benchmark has run.

## Adopting it (after you choose — ask Claude to do it)

1. Add `workflows/zimage_wan22_i2v_hq.json` (the graph `upgraded_clip()` builds)
   and point the `broll_video` role in `workflows/manifest.json` at it; keep the
   old file as a fallback.
2. Raise `MAX_VIDEO_GEN_PIXELS` in `backend/presentation/assets.py` from
   832×480 to the chosen size (1024×576 or 960×544).
3. Set the clip fps to 32 (RIFE doubles the frames) where BuzzEdit binds `fps`.
4. Add the hi-res pass to `workflows/zimage1.json` for stills.
5. Re-measure one real production; update the timing baselines.

## Undo

- Move `custom_nodes\ComfyUI-FlashVSR_Ultra_Fast` back out (or delete it).
- `"C:\Users\singh\Documents\ComfyUI\.venv\Scripts\python.exe" -m pip uninstall triton-windows`
  (only if it causes trouble; nothing else uses it today).
- Delete the downloaded files listed above (~28 GB) if the upgrade is not adopted.

## Sources

- LTX-2.3 vs Wan 2.2 I2V benchmark — https://zenn.dev/toki_mwc/articles/ltx23-vs-wan22-i2v-benchmark-rtx5090?locale=en
- lightx2v Wan 2.2 distill LoRAs — https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- Wan 2.2 I2V GGUF — https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF
- Video upscaling in ComfyUI (SeedVR2, FlashVSR, RIFE) — https://artokun.mintlify.app/blog/video-upscale-comfyui
- FlashVSR node — https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast
- triton-windows compatibility — https://github.com/woct0rdho/triton-windows
