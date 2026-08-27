# DrivoR + CEM (MPC) — portable bundle

Self-contained DrivoR trajectory scorer wired into the Cross-Entropy-Method
(CEM) refiner. The code in this folder is everything you need from the repo:
the minimal `navsim` source subset and the DrivoR config. No other part of the
original repo is required, and **no `nuplan` / `hydra` dependency** is pulled
in. The two weight files are not versioned here — fetch them once with
[Weights](#weights) below.

## Layout

```
drivor_cem_portable/
├── README.md
├── requirements.txt
├── checkpoints/                                 # (downloaded, see Weights)
│   └── epoch_29_54.6.ckpt                      # DrivoR weights (~292 MB)
├── weights/                                     # (downloaded, see Weights)
│   └── vit_small_patch14_reg4_dinov2.lvd142m/  # DINOv2 backbone (~85 MB)
│       ├── model.safetensors
│       └── config.json
└── navsim/                                      # minimal importable subset
    ├── agents/utils/
    │   ├── drivor_cem_standalone.py   # <-- entry point (load + score_fn + CEM)
    │   ├── drivor_mpc.py              # DrivoR loader + score_fn builder
    │   ├── cem_utils.py               # model-agnostic CEM optimizer
    │   └── inference_profiler.py
    ├── agents/drivoR/                 # DrivoR model + transformer + DINOv2-LoRA
    └── planning/script/config/common/agent/drivoR.yaml
```

## Weights

Neither weight file is tracked in git. Download both once, from this folder:

```bash
# DrivoR scorer checkpoint (~292 MB) — EPDMS 54.6, train 85k + simscale 134k, 30 epochs
mkdir -p checkpoints
wget -O checkpoints/epoch_29_54.6.ckpt \
  https://github.com/valeoai/DrivoR/releases/download/Scaling/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth

# DINOv2 backbone (~85 MB)
mkdir -p weights/vit_small_patch14_reg4_dinov2.lvd142m
huggingface-cli download timm/vit_small_patch14_reg4_dinov2.lvd142m \
  model.safetensors config.json \
  --local-dir weights/vit_small_patch14_reg4_dinov2.lvd142m
```

The release asset is published as `.pth`, but it is byte-for-byte the
checkpoint this bundle expects — a Lightning checkpoint loaded as
`torch.load(...)["state_dict"]` — so the `-O` rename above is all that is
needed. To keep the published name instead, point `load_drivor` at it
explicitly:

```python
model = dcs.load_drivor(ckpt_path="checkpoints/nav2_30_epochs_with_134k_simscale_85ktrain_54.6.pth")
```

This is the same checkpoint the main repo puts in `../weights/`; if you already
have it there, symlink rather than downloading twice.

## Install

```bash
pip install -r requirements.txt   # into a CUDA-enabled env (matches your torch build)
```

## Run the toy example

From this folder, after fetching the [weights](#weights):

```bash
python -m navsim.agents.utils.drivor_cem_standalone
# or
PYTHONPATH=. python navsim/agents/utils/drivor_cem_standalone.py
```

It loads DrivoR from the bundled config/checkpoint (seed=2), builds random
DrivoR-shaped inputs, and runs CEM refinement over a fan of toy proposals with
the DrivoR scorer + comfort + anchor costs, printing the refined trajectory and
scores. Flags: `--batch-size`, `--num-proposals`, `--cem-iterations`,
`--cem-samples`.

> The bundle resolves paths relative to its own location, so it runs from any
> working directory — including a shell that has `NAVSIM_DEVKIT_ROOT` exported
> for the main NAVSIM evaluation. To point at a different checkout instead, set
> `DRIVOR_CEM_ROOT`.

## Use it in your own code

```python
from navsim.agents.utils import drivor_cem_standalone as dcs

model = dcs.load_drivor()                 # default config/ckpt/overrides + seed=2

features = {
    "drivor_image":      image,           # (B, num_cams, 3, H, W), ImageNet-normalized
    "drivor_ego_status": ego_status,      # (B, T_hist, 11): [pose(3), vel(2), acc(2), cmd(4)]
}

out = dcs.cem_refine_with_drivor(
    model, features,
    proposals=proposals,                  # (B, N_p, P, 3) seed trajectories
    best_traj=best_traj,                  # (B, P, 3) anchor / argmax pick
)
refined = out["trajectory"]               # (B, P, 3)
```

Build `drivor_image` / `drivor_ego_status` from a real `agent_input` with
`drivor_mpc.build_drivor_image` / `drivor_mpc.build_drivor_ego_status` — these
produce exactly the tensors specified below.

## Input specification

### `drivor_image` — `(B, num_cams, 3, H, W)`, `float32`

With the bundled config `num_cams = 4`, `H = 672`, `W = 1148` (image_size is
stored as `[W, H] = [1148, 672]`).

- **Camera order (axis 1):** `[front, back, left, right]` = `[cam_f0, cam_b0,
  cam_l0, cam_r0]`. This is the order DrivoR's feature builder stacks them: it
  walks the fixed key list `(cam_f0, cam_b0, cam_l0, cam_l1, cam_l2, cam_r0,
  cam_r1, cam_r2)` and keeps each camera whose config list is non-empty and
  whose image is present. The bundled config enables exactly f0/b0/l0/r0, so
  the result is front, back, left, right. **Keep this order** — the model has
  no per-camera position embedding that would let it recover a permutation.
- **Channels:** RGB, channels-first (`permute(2, 0, 1)` of an `H×W×3` image).
- **Resize:** each camera image is resized to `(W, H) = (1148, 672)` (PIL
  `Image.resize`, which takes `(width, height)`).
- **Normalization:** scale to `[0, 1]` (`/255`), then ImageNet normalize per
  channel: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`,
  i.e. `x = (x/255 − mean) / std`.

This matches `DrivoRFeatureBuilder._get_camera_feature` in
`navsim/agents/drivoR/drivor_features.py`. The scorer path consumes **only**
this normalized image stack — camera intrinsics / extrinsics (`cam_K`,
`world_2_cam`) that the full feature builder also emits are not needed, so
`drivor_mpc.build_drivor_image` (the image-only subset) is sufficient.

### `drivor_ego_status` — `(B, T_hist, 11)`, `float32`

Ego state history in local (ego rear-axle) coordinates, **most recent frame
last** (the model uses `[:, -1]` since `full_history_status=false`). The 11
features per timestep are concatenated as `[pose(3), velocity(2),
acceleration(2), driving_command(4)]`:

| idx | field | meaning |
|-----|-------|---------|
| 0–2 | `ego_pose` | `(x, y, heading)` in local coords — x forward, y left, heading in **radians**. At the current frame this is `(0, 0, 0)`. |
| 3–4 | `ego_velocity` | `(vx, vy)` in m/s, local frame. `vx` is forward speed. |
| 5–6 | `ego_acceleration` | `(ax, ay)` in m/s², local frame. |
| 7–10 | `driving_command` | **one-hot**, length 4: `[left, straight, right, unknown]`. Encodes desired route intent only (not obstacles/signs); left/right cover turns, lane changes and sharp curves. Exactly one entry is 1 (use `unknown` = `[0,0,0,1]` if intent is unavailable). |

The CEM boundary kinematics (`init_speed`, `past_a`, `past_omega`) are derived
from this tensor using index 3 (`vx`), index 5 (`ax`) and the heading
difference at index 2 across the last two frames — so provide at least
`T_hist ≥ 2` real history frames for a meaningful `past_omega`.

### `proposals` — `(B, N_p, P, 3)` and `best_traj` — `(B, P, 3)`, `float32`

Candidate trajectories and the anchor, both in the same local (rear-axle)
frame as `ego_pose`: last-dim is `(x, y, heading)`, `heading` in radians.
DrivoR was trained on `P = 8` poses sampled every `0.5 s` (t = 0.5 s … 4 s).
`proposals` seeds the CEM Gaussian std; `best_traj` is the anchor the
regularizer pulls toward (typically the argmax pick). If your trajectories
have more than 8 poses, pass `subsample_indices` (e.g. `(4, 9, …, 39)` for a
40-pose, 0.1 s grid) to select the 8 timestamps DrivoR expects.

## CEM defaults

`num_iterations=5`, `num_samples=64`, `num_elites=num_samples//8`, objective =
DrivoR score + closed-form comfort cost + anchor regularizer (the un-ablated
`cem_utils.cem_optimize` baseline). Output `trajectory` is the converged CEM
mean where it beats the anchor on the combined score, else the anchor
(`vote=True`); pass `vote=False` to always return the CEM mean.
