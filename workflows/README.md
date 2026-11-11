# ComfyUI workflows

Every ComfyUI graph BuzzEdit runs -- and so every picture, clip, music bed and sound the Buzzcaf
Studio gets from ComfyUI, since the Studio sends all of that work through BuzzEdit -- is a file in
this folder, in ComfyUI's **API format** (`{node_id: {class_type, inputs}}`, not the UI's
"workflow" format). Code never invents a graph: it loads one of these files, writes the job's
values into the inputs named in `manifest.json` (prompt, size, seed, length, fps, file prefix...)
and posts it to ComfyUI's `/prompt`. The only nodes code adds are the two of the face swap (below).

Resolution: `backend/presentation/workflows.py` `resolve(role)` picks a file for a role from the
app setting (`workflow_<role>` in `backend/data/app_settings.json`, the dropdowns in BuzzEdit's
settings), else `manifest.json`, else a built-in default. For `broll_video` with no setting, the
clip-generation mode decides (`clip_generation_mode`: the Classic / HQ toggle on the Studio's
Projects page, sent to BuzzEdit before each production). `GET /api/settings/workflows` lists them.

## In use (2026-10-24)

| File | Role / what makes it | Who uses it | Models |
|---|---|---|---|
| `zimage1_esrgan.json` | `broll_image`: every still (Z-Image Turbo, RealESRGAN x2) | **live** (setting `workflow_broll_image`); every production's pictures, the character reference portrait, thumbnail scene | `unet/z_image_turbo_bf16`, `text_encoders/qwen_3_4b`, `vae/ae`, `upscale_models/RealESRGAN_x2plus` |
| `zimage1_fast.json` | `broll_image` (manifest default): the same plus SageAttention | when the setting is cleared | as above + **comfyui-kjnodes** |
| `zimage1_hq.json` | vibe-reel HQ still: Z-Image with a 1.5x latent hires pass | `presentation/vibe_reel.py` (Life3Baje vibe reels) | as zimage1 |
| `zimage_wan22_i2v_lite16.json` | `broll_video`: Z-Image first frame -> Wan 2.2 i2v 14B Q5_K_M + lightx2v 1022 LoRAs, 832x480 @ 16 fps, Lanczos to 1080p | **live** (setting `workflow_broll_video`); every production's clips | above + `unet/wan2.2_i2v_{high,low}_noise_14B_Q5_K_M.gguf`, `loras/wan2.2_i2v_A14b_{high,low}_noise_lora_rank64_lightx2v_4step_1022`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled`, `vae/wan_2.1_vae` |
| `zimage_wan22_i2v_fast16.json` | `broll_video`, clip mode **HQ**: lite16 + SageAttention | when the setting is cleared and the mode is HQ | as lite16 + **comfyui-kjnodes** |
| `zimage_wan22_i2v_reel_portrait.json` | vibe-reel portrait clip: Wan 2.2 Q5_K_M, FlashVSR x2, RIFE x2 to 32 fps | `vibe_reel.py` (clip mode `reel_hq`) | as lite16 + `FlashVSR-v1.1/`, `frame_interpolation/rife_v4.26` + **ComfyUI-FlashVSR_Ultra_Fast** |
| `zimage_wan22_i2v_hq16.json` | fallback for the portrait reel: 1024x576 + RealESRGAN on every frame | when FlashVSR's node is missing | as lite16 + RealESRGAN_x2plus |
| `ace_step15_music.json` | `music15`: ACE-Step 1.5 Turbo, real bpm / key conditioning | music beds that ask for a tempo, raga or slow/calm mood | `diffusion_models/acestep_v1.5_turbo`, `text_encoders/qwen_{0.6b,1.7b}_ace15`, `vae/ace_1.5_vae` |
| `ace_step_music.json` | `music`: ACE-Step 1.0 | **the Studio's default music engine** (`music_engine: "ace_step"`); every other music bed | `checkpoints/ace_step_v1_3.5b` |
| `stable_audio_sfx.json` | `audio`: Stable Audio Open 1.0 | scene sounds (`[sfx: ...]`), transition hits, short stingers | `checkpoints/stable-audio-open-1.0`, `text_encoders/t5_base` |
| `thumbnail.json` | `thumbnail`: SDXL Juggernaut XL v9 | thumbnail scene when the still graph cannot build one (`agents/thumbnail_agent.py`) | `checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2` |

### Built on top of a file: the main-character face swap

`backend/presentation/character.py` `with_face_swap()` takes whichever still graph `broll_image`
resolved to and adds two nodes before its `SaveImage`: `bz_face_ref` (`LoadImage`, the main
character's reference photo in ComfyUI's input folder) and `bz_face_swap` (`ReActorFaceSwap`:
`inswapper_128.onnx`, `retinaface_resnet50`, CodeFormer `codeformer-v0.1.0.pth` at weight 0.7,
largest face only, filtered by gender when the character's is known). The composed graph, as
ComfyUI receives it, is kept in `reference/zimage1_esrgan_faceswap.json` -- a reference copy
regenerated from the code, not a file BuzzEdit loads (see "Regenerating" below). Needs
**ComfyUI-ReActor**; BuzzEdit checks the node exists (`client.has_node`) and skips the swap without it.

## Kept but not used

| File | Why it is here |
|---|---|
| `zimage_wan22_i2v.json` | clip mode **Classic** (Wan 2.2 Q4_K_S + lightx2v v1 LoRAs). Its four model files were moved to `H:\ComfyUI_models_parked\B_portable_ComfyUI_models\` on 2026-10-24: move them back before choosing Classic. Also the manifest's `broll_video` entry (its bindings apply to every clip file) and `tools/bench_generation.py`'s base graph |
| `zimage_wan22_i2v_hq.json`, `zimage_wan22_i2v_hq_flashvsr.json` | earlier HQ graphs, selectable by hand in the settings dropdown only |
| `zimage1.json` | plain Z-Image still, `tools/bench_generation.py` only |
| `broll_generation.json` | the original SD 1.5 graph, the last-resort default; its `v1-5-pruned-emaonly.ckpt` is not installed |
| `*.bak`, `*.bak_upscale` | old copies; not `*.json`, so never listed |

There is no graph for the `graphic` role, and none for matte, eye contact, stabilise or upscaling
outside the clip files: those run in BuzzEdit's own Python, not in ComfyUI.

## What ComfyUI needs

**ComfyUI** Desktop 0.33.x (BuzzEdit's Electron starts it: code `C:\Users\singh\ComfyUI-Installs\ComfyUI\ComfyUI`,
data and venv `C:\Users\singh\Documents\ComfyUI`, port 8188). Core nodes cover everything except:

| Custom node pack | Nodes | Used by |
|---|---|---|
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (1.1.9) | `UnetLoaderGGUF` | every `zimage_wan22_i2v*.json` |
| [comfyui-kjnodes](https://github.com/kijai/ComfyUI-KJNodes) (1.3.7) | `PathchSageAttentionKJ` | `zimage1_fast.json`, `zimage_wan22_i2v_fast16.json` |
| [ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) | `FlashVSRNode` | `zimage_wan22_i2v_reel_portrait.json`, `_hq_flashvsr.json` |
| [ComfyUI-ReActor](https://github.com/Gourieff/ComfyUI-ReActor) | `ReActorFaceSwap` | the face swap |

`FrameInterpolate` / `FrameInterpolationModelLoader` (RIFE) and the ACE-Step / Z-Image / Wan nodes
are core in this ComfyUI version.

**Models.** ComfyUI reads two model roots (`C:\Users\singh\Documents\ComfyUI\extra_model_paths.yaml`):
`C:\Users\singh\Documents\ComfyUI\models` and the old portable tree
`B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI\models`. The
folder names in the table above are relative to either root. Unused models were parked on
2026-10-03 in `E:\ComfyUI_models_parked\` and on 2026-10-24 in `H:\ComfyUI_models_parked\`
(each with a `MOVED.txt` listing where every file came from): if ComfyUI reports a missing model,
look there first.

## Adding or changing a workflow

1. Build it in ComfyUI, then **Workflow -> Export (API)** and save it here as `<name>.json`.
2. Name the inputs BuzzEdit should fill in `manifest.json` under its role (`bindings`: node id,
   `inputs`, input name -- a list of such paths when several nodes take the same value).
3. Select it in BuzzEdit's settings (or put it in `manifest.json` as the role's `file`), run one job,
   and add a row to the tables above: role, who uses it, models, custom nodes.

## Regenerating `reference/`

    .venv\Scripts\python.exe backend\tools\export_reference_graphs.py

writes the composed graphs (currently the face swap over the live still graph) from the same code
BuzzEdit runs, so the reference copy cannot drift from what is sent.
