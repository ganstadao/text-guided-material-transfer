# Text-guided Material Transfer

An experimental extension of [ZeST: Zero-Shot Material Transfer from a Single Image](https://github.com/ttchengab/zest_code). It uses a **material exemplar image** and a text phrase identifying the desired material in that image. The prototype explores two additions to ZeST:

1. CLIPSeg-based localization of a material region in a potentially cluttered exemplar.
2. A custom cross-attention step between text and image features before IP-Adapter projection.

This is research-in-progress code, not a text-only material generator. The two additions have **not** been shown to improve results consistently. No benchmark claims are made here.

## Repository contents

| Path | Purpose |
| --- | --- |
| `demo.ipynb` | ZeST baseline notebook, with stored outputs removed. |
| `run_batch.py` | ZeST batch inference script; needs inputs and path review before use. |
| `demo_gradio.py` | Interactive prototype containing the CLIPSeg preprocessing function and custom adapter path. |
| `CustomIPAdapterXL.py`, `utils.py` | Experimental text/image feature fusion and attention utilities. |
| `experiments/` | Small exploratory scripts retained from development. |
| `requirements.txt` | Historical dependency list; not a validated lockfile. |

## Setup and assets

Use a CUDA environment with SDXL, IP-Adapter, a depth ControlNet, DPT and CLIPSeg. The [upstream ZeST instructions](https://github.com/ttchengab/zest_code#installation) describe the baseline dependencies and asset layout. This repository deliberately excludes pretrained weights, datasets, caches, and generated image batches.

The scripts currently expect the following paths relative to the repository root:

```text
DPT/weights/dpt_hybrid-midas-501f0c75.pt
ip_adapter/
models/image_encoder/
models/stable-diffusion-xl-base-1.0/
models/controlnet-depth-sdxl-1.0/
sdxl_models/ip-adapter_sdxl_vit-h.bin
demo_assets/input_imgs/
demo_assets/material_exemplars/
demo_assets/depths/
demo_assets/outputs/
demo_assets/CLIPSeg_masks/
```

The local `demo_gradio.py` also requests `CIDAS/clipseg-rd64-refined` through Transformers. Model licenses and access requirements belong to the respective providers.

## Current limitations

- The call to `preprocess_material()` in `demo_gradio.py` is currently commented out, so the default generation path **does not apply CLIPSeg to the exemplar**. The CLIPSeg function and exploratory script are included for further evaluation.
- `CustomIPAdapterXL.py` is an experimental modification; its attention behavior, memory use, and compatibility with dependency versions need validation.
- The baseline batch script assumes particular `.png` filenames and model locations. It is not a portable evaluation runner yet.
- Existing local outputs are exploratory examples, not a controlled benchmark. No dataset-level performance numbers are reported.

For a fair component study, compare baseline ZeST, ZeST + exemplar segmentation, ZeST + custom attention, and both additions on the same fixed input/exemplar pairs and seeds. Keep segmentation, material fidelity, target geometry, and background preservation separate in the analysis.

## Attribution

The baseline notebook and batch script derive from the official [ZeST implementation](https://github.com/ttchengab/zest_code) by Ta-Ying Cheng and collaborators (ECCV 2024). `LICENSE` retains the upstream MIT copyright notice. The `ip_adapter` and `DPT` packages are external dependencies and are not vendored here.

Please cite the [ZeST paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00144.pdf) when using the underlying method.
